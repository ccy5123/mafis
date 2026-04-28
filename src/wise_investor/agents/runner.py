"""Native Ollama agent loop — bypasses CrewAI's function-calling path.

Why: diagnose_tool_calling.py showed that both ollama.chat(tools=...) and
POST /v1/chat/completions return structured tool_calls correctly for Llama 3.1
8B, Qwen 2.5 7B, and their 16K variants. CrewAI 1.14 / LiteLLM evidently does
not surface those tool_calls to the agent loop, so both our Llama and Qwen
runs produced hallucinated numbers without any tool invocation.

This runner talks to Ollama directly, executes our Python calculation tools
when the model emits tool_calls, feeds the results back, and loops until the
model produces a final textual answer. Reports are saved by the caller.

Tools registered here are the same Python functions wrapped for CrewAI in
agents/tools.py; we expose them with OpenAI-compatible JSON schemas here so
the native Ollama call can forward them verbatim.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wise_investor.config import settings
from wise_investor.llm.base import SamplingConfig


FACTS_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "facts_cache"
from wise_investor.data.cross_validate import cross_validate_quote
from wise_investor.data.finnhub import FinnhubClient as FMPClient  # alias for minimal call-site change
from wise_investor.data.fred import (
    FredError,
    format_macro_snapshot,
    get_macro_snapshot,
)
from wise_investor.tools.dcf import reverse_dcf as reverse_dcf_impl
from wise_investor.tools.valuation import (
    calculate_ev_ebitda as calculate_ev_ebitda_impl,
    calculate_per as calculate_per_impl,
    get_peer_multiples as get_peer_multiples_impl,
)
from wise_investor.tools.verify import (
    fetch_source_value,
    list_supported_fields,
    verify_number as verify_number_impl,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers (mirror agents/tools.py)
# ---------------------------------------------------------------------------


def _fmt_num(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1e9:
        return f"{v / 1e9:,.{digits}f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:,.{digits}f}M"
    return f"{v:,.{digits}f}"


def _fmt_pct(v: float | None) -> str:
    return "N/A" if v is None else f"{v * 100:.2f}%"


def _fmt_warnings(warnings: list[str]) -> str:
    if not warnings:
        return "Warnings: none"
    return "Warnings:\n" + "\n".join(f"  - {w}" for w in warnings)


# ---------------------------------------------------------------------------
# Tool executors — each takes a parsed dict of arguments, returns a string
# ---------------------------------------------------------------------------


def _exec_cross_validate_quote(args: dict[str, Any]) -> str:
    symbol = str(args["symbol"]).upper()
    with FMPClient() as c:
        r = cross_validate_quote(symbol, fmp=c)
    lines = [f"Cross-validation for {symbol} (threshold {r.threshold_pct}%):"]
    for cmp in r.comparisons:
        flag = (
            "OK" if cmp.within_threshold is True
            else "DIVERGES" if cmp.within_threshold is False
            else "UNKNOWN"
        )
        lines.append(
            f"  - {cmp.field}: FMP={_fmt_num(cmp.fmp_value)} "
            f"| yfinance={_fmt_num(cmp.yf_value)} "
            f"| diff={cmp.diff_pct if cmp.diff_pct is not None else 'N/A'}% "
            f"| {flag}"
        )
        if cmp.note:
            lines.append(f"    note: {cmp.note}")
    lines.append("Flagged: yes" if r.any_flagged else "Flagged: no")
    return "\n".join(lines)


def _exec_calculate_per(args: dict[str, Any]) -> str:
    symbol = str(args["symbol"]).upper()
    with FMPClient() as c:
        r = calculate_per_impl(symbol, client=c)
    return "\n".join(
        [
            f"PER for {symbol}",
            f"Source: Python-computed from FMP data; fiscal year end {r.as_of or 'unknown'}",
            f"Computed PER: {_fmt_num(r.computed)}",
            "Inputs:",
            f"  - current price: {_fmt_num(r.inputs.get('price'))} (FMP /quote)",
            f"  - EPS diluted: {_fmt_num(r.inputs.get('eps_diluted_latest_annual'))} "
            f"(FMP /income-statement annual {r.as_of})",
            f"FMP-reported PER (same fiscal year): {_fmt_num(r.fmp_reported)}",
            f"Divergence vs FMP: "
            + (f"{r.diff_pct_vs_fmp:.2f}%" if r.diff_pct_vs_fmp is not None else "N/A"),
            _fmt_warnings(r.warnings),
        ]
    )


def _exec_calculate_ev_ebitda(args: dict[str, Any]) -> str:
    symbol = str(args["symbol"]).upper()
    with FMPClient() as c:
        r = calculate_ev_ebitda_impl(symbol, client=c)
    return "\n".join(
        [
            f"EV/EBITDA for {symbol}",
            f"Source: Python-computed from FMP data; fiscal year end {r.as_of or 'unknown'}",
            f"Computed EV/EBITDA: {_fmt_num(r.computed)}",
            "Inputs:",
            f"  - Enterprise Value: {_fmt_num(r.inputs.get('enterprise_value'))} "
            f"(FMP /enterprise-values annual {r.as_of})",
            f"  - EBITDA: {_fmt_num(r.inputs.get('ebitda_latest_annual'))} "
            f"(FMP /income-statement annual {r.as_of})",
            f"FMP-reported EV/EBITDA: {_fmt_num(r.fmp_reported)}",
            f"Divergence vs FMP: "
            + (f"{r.diff_pct_vs_fmp:.2f}%" if r.diff_pct_vs_fmp is not None else "N/A"),
            _fmt_warnings(r.warnings),
        ]
    )


def _exec_get_peer_multiples(args: dict[str, Any]) -> str:
    symbol = str(args["symbol"]).upper()
    max_peers = int(args.get("max_peers", 5))
    additional_peers = args.get("additional_peers") or None
    with FMPClient() as c:
        t = get_peer_multiples_impl(
            symbol,
            client=c,
            max_peers=max_peers,
            additional_peers=additional_peers,
        )
    lines = [
        f"Peer multiples table — target {t.target_symbol} (as of {t.as_of or 'unknown'})",
    ]
    if t.override_sources:
        lines.append(
            f"Manual peer overrides (from value chain brief): {', '.join(t.override_sources)}"
        )
    lines.append(
        f"{'Symbol':<8} {'Name':<28} {'MktCap':>10} {'PER':>8} {'EV/EBITDA':>10}"
    )
    lines.append("-" * 68)
    for row in t.rows:
        lines.append(
            f"{row.symbol:<8} {(row.name or '')[:28]:<28} "
            f"{_fmt_num(row.market_cap):>10} {_fmt_num(row.per):>8} "
            f"{_fmt_num(row.ev_ebitda):>10}"
        )
        if row.warnings:
            lines.append(f"         warnings: {'; '.join(row.warnings)}")

    if t.excluded:
        lines.append("")
        lines.append(
            "Excluded peers (data quality — do NOT use for comparison):"
        )
        for row in t.excluded:
            reason = next(
                (w for w in row.warnings if w.startswith("excluded from comparison:")),
                "excluded (reason unknown)",
            )
            reason_short = reason.replace("excluded from comparison: ", "")
            lines.append(
                f"  - {row.symbol} ({row.name or '?'}): {reason_short}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Peer override parsing from value chain brief
# ---------------------------------------------------------------------------


_PEER_OVERRIDE_HEADING_PATTERN = re.compile(
    r"^##\s*Peer\s*Override\b", re.IGNORECASE | re.MULTILINE
)
# Line like: "- SMNEY — Siemens Energy ADR" → capture "SMNEY"
_PEER_TICKER_PATTERN = re.compile(r"^[-*]\s+([A-Z][A-Z0-9.\-]{0,9})\b")


def parse_peer_override(value_chain_text: str) -> list[str]:
    """Extract ticker symbols from the '## Peer Override' section of a value
    chain brief. Returns empty list if the section is absent, empty, or marks
    "no override needed".

    Format expected:
        ## Peer Override
        Any prose intro...
        - TICKER — description
        - OTHER — description

    Case-insensitive heading match. Lines without a leading ticker are ignored.
    """
    m = _PEER_OVERRIDE_HEADING_PATTERN.search(value_chain_text)
    if not m:
        return []

    # Slice from the heading to the next heading or end of text.
    start = m.end()
    tail = value_chain_text[start:]
    next_heading = re.search(r"^##\s", tail, re.MULTILINE)
    section = tail[: next_heading.start()] if next_heading else tail

    tickers: list[str] = []
    for line in section.splitlines():
        tm = _PEER_TICKER_PATTERN.match(line.strip())
        if tm:
            ticker = tm.group(1).upper()
            # Skip obvious non-ticker words like "NONE" or "NOTES"
            if ticker in {"NONE", "NOTES", "SEE", "NO"}:
                continue
            if ticker not in tickers:
                tickers.append(ticker)
    return tickers


def _exec_reverse_dcf(args: dict[str, Any]) -> str:
    symbol = str(args["symbol"]).upper()
    discount_rate = float(args.get("discount_rate", 0.10))
    terminal_growth = float(args.get("terminal_growth", 0.025))
    high_growth_years = int(args.get("high_growth_years", 10))
    with FMPClient() as c:
        r = reverse_dcf_impl(
            symbol,
            client=c,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            high_growth_years=high_growth_years,
        )
    return "\n".join(
        [
            f"Reverse DCF for {symbol}",
            f"Source: Python-solved; fiscal year end {r.as_of or 'unknown'}",
            f"Implied annual FCF growth ({high_growth_years}y stage 1): "
            f"{_fmt_pct(r.implied_growth_rate)}",
            "Inputs:",
            f"  - market cap: {_fmt_num(r.current_market_cap)} (FMP /quote)",
            f"  - latest annual FCF: {_fmt_num(r.inputs.get('fcf_latest_annual'))} "
            f"(FMP /cash-flow-statement {r.as_of})",
            f"  - FCF source: {r.inputs.get('fcf_source', 'N/A')}",
            f"  - discount rate: {_fmt_pct(discount_rate)}",
            f"  - terminal growth: {_fmt_pct(terminal_growth)}",
            f"  - high-growth years: {high_growth_years}",
            _fmt_warnings(r.warnings),
        ]
    )


def _exec_verify_number(args: dict[str, Any]) -> str:
    claim = float(args["claim"])
    field = str(args["field"])
    symbol = str(args["symbol"]).upper()
    tolerance_pct = float(args.get("tolerance_pct", 1.0))
    with FMPClient() as c:
        r = verify_number_impl(
            claim=claim,
            field=field,
            symbol=symbol,
            client=c,
            tolerance_pct=tolerance_pct,
        )
    match_str = (
        "MATCH" if r.matches is True
        else "MISMATCH" if r.matches is False
        else "UNKNOWN (source unavailable)"
    )
    return "\n".join(
        [
            f"Verification for {symbol}.{r.field}",
            f"  claim: {_fmt_num(r.claim, 4)}",
            f"  source value: {_fmt_num(r.source_value, 4)}",
            f"  source: {r.source_citation}",
            f"  diff: {r.diff_pct if r.diff_pct is not None else 'N/A'}% "
            f"(tolerance {r.tolerance_pct}%)",
            f"  result: {match_str}",
            _fmt_warnings(r.warnings),
        ]
    )


def _exec_fetch_field(args: dict[str, Any]) -> str:
    """Read-only field fetch for pre-gather. Avoids MATCH/MISMATCH confusion."""
    field = str(args["field"])
    symbol = str(args["symbol"]).upper()
    with FMPClient() as c:
        r = fetch_source_value(field=field, symbol=symbol, client=c)
    return "\n".join(
        [
            f"Field: {r.field}",
            f"Symbol: {symbol}",
            f"Source value: {_fmt_num(r.value, 4)}",
            f"Source: {r.source_citation}",
            _fmt_warnings(r.warnings),
        ]
    )


# ---------------------------------------------------------------------------
# Tool registry (OpenAI-style JSON schemas + Python executor)
# ---------------------------------------------------------------------------


@dataclass
class ToolEntry:
    name: str
    spec: dict[str, Any]
    executor: Callable[[dict[str, Any]], str]


TOOLS: list[ToolEntry] = [
    ToolEntry(
        name="cross_validate_quote",
        spec={
            "type": "function",
            "function": {
                "name": "cross_validate_quote",
                "description": (
                    "Compare FMP and yfinance quotes for a US ticker and flag any field "
                    "whose values diverge by more than 5%. Run this first, before quoting prices."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Uppercase US ticker, e.g. NVDA"}
                    },
                    "required": ["symbol"],
                },
            },
        },
        executor=_exec_cross_validate_quote,
    ),
    ToolEntry(
        name="calculate_per",
        spec={
            "type": "function",
            "function": {
                "name": "calculate_per",
                "description": (
                    "Compute PER = current price / latest annual diluted EPS. Returns "
                    "Python-computed value, FMP-reported value, inputs, and any warnings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"}
                    },
                    "required": ["symbol"],
                },
            },
        },
        executor=_exec_calculate_per,
    ),
    ToolEntry(
        name="calculate_ev_ebitda",
        spec={
            "type": "function",
            "function": {
                "name": "calculate_ev_ebitda",
                "description": (
                    "Compute EV / EBITDA for the latest annual fiscal year. Returns "
                    "Python-computed and FMP-reported values with inputs and warnings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"}
                    },
                    "required": ["symbol"],
                },
            },
        },
        executor=_exec_calculate_ev_ebitda,
    ),
    ToolEntry(
        name="get_peer_multiples",
        spec={
            "type": "function",
            "function": {
                "name": "get_peer_multiples",
                "description": (
                    "Return a peer comparison table of PER and EV/EBITDA for the target "
                    "ticker and its top peers from FMP /stock-peers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "max_peers": {"type": "integer", "default": 5},
                    },
                    "required": ["symbol"],
                },
            },
        },
        executor=_exec_get_peer_multiples,
    ),
    ToolEntry(
        name="reverse_dcf",
        spec={
            "type": "function",
            "function": {
                "name": "reverse_dcf",
                "description": (
                    "Solve for the annual FCF growth rate the market is pricing in given "
                    "current market cap and latest FCF. Two-stage DCF with explicit "
                    "assumptions (discount_rate, terminal_growth, high_growth_years)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "discount_rate": {"type": "number", "default": 0.10},
                        "terminal_growth": {"type": "number", "default": 0.025},
                        "high_growth_years": {"type": "integer", "default": 10},
                    },
                    "required": ["symbol"],
                },
            },
        },
        executor=_exec_reverse_dcf,
    ),
    ToolEntry(
        name="verify_number",
        spec={
            "type": "function",
            "function": {
                "name": "verify_number",
                "description": (
                    "Check whether a numeric claim matches the authoritative source "
                    "for that field on the ticker. Pass claim=0.0 to use this as a "
                    '"get source value" call; the tool returns the source value and '
                    "you use that verbatim. Supported fields: "
                    + ", ".join(list_supported_fields())
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "number"},
                        "field": {"type": "string"},
                        "symbol": {"type": "string"},
                        "tolerance_pct": {"type": "number", "default": 1.0},
                    },
                    "required": ["claim", "field", "symbol"],
                },
            },
        },
        executor=_exec_verify_number,
    ),
]


TOOL_SPECS: list[dict[str, Any]] = [t.spec for t in TOOLS]
TOOL_LOOKUP: dict[str, ToolEntry] = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


@dataclass
class AgentRunResult:
    final_text: str
    tool_calls_made: int
    iterations: int
    elapsed_sec: float
    tool_trace: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-gather path — Python runs all tools up front, no LLM tool decisions
# ---------------------------------------------------------------------------


# Fields the Analyst should have raw source values for in Section 3.
# (Revenue, profitability, cash flow, leverage — the standard picture.)
_VERIFY_FIELDS: tuple[str, ...] = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "free_cash_flow",
    "operating_cash_flow",
    "total_debt",
    "total_stockholders_equity",
)


def _facts_cache_path(symbol: str) -> Path:
    stamp = dt.date.today().isoformat()
    return FACTS_CACHE_DIR / f"{symbol.upper()}_{stamp}.json"


def _load_peer_overrides_for(symbol: str) -> list[str]:
    """Read docs/value_chains/<SYMBOL>.md and extract Peer Override tickers.

    Returns [] if the file or section is missing. Tolerant: never raises.
    """
    vc_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "value_chains"
        / f"{symbol.upper()}.md"
    )
    if not vc_path.exists():
        return []
    try:
        text = vc_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read value chain file %s: %s", vc_path, e)
        return []
    return parse_peer_override(text)


def pre_gather_facts(symbol: str, use_cache: bool = True) -> dict[str, str]:
    """Run every Phase 1A tool up front and return a dict of tool-name → output.

    This bypasses the LLM's tool-selection step, which empirically fails on
    small open-source models when asked to produce structured multi-section
    reports. Instead, Python deterministically gathers the full Phase 1A
    fact set and hands it to the LLM as context for narrative synthesis only.

    Facts are cached to data/fmp_cache/{SYMBOL}_{YYYY-MM-DD}.json so prompt
    iterations within the same day do not re-hit FMP's 250/day free-tier
    quota. Pass use_cache=False to force a fresh fetch.
    """
    symbol = symbol.upper()
    cache_path = _facts_cache_path(symbol)

    if use_cache and cache_path.exists():
        logger.info("Loading cached facts from %s", cache_path)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # Korean-ticker dispatch: detect 6-digit KRX codes (with or without
    # .KS / .KQ suffix) and route to the DART facts builder instead of
    # the Finnhub path. Output schema is intentionally compatible so the
    # agents and quality metrics don't need to know which source
    # supplied the numbers.
    from wise_investor.data.dart_facts import (
        is_korean_ticker,
        pre_gather_dart_facts,
    )
    if is_korean_ticker(symbol):
        logger.info("Korean ticker detected (%s); routing to DART.", symbol)
        # Pull the FRED KRW/USD rate once so dollar equivalents are
        # available. Fail-soft — if FRED is down, facts are KRW-only.
        # Use the module-level `get_macro_snapshot` already imported
        # at the top of this file (no local re-import to avoid Python
        # treating the name as a function-local elsewhere in the
        # function body).
        usd_krw_rate: float | None = None
        try:
            snap = get_macro_snapshot()
            if snap.usd_krw_rate and snap.usd_krw_rate.value:
                usd_krw_rate = float(snap.usd_krw_rate.value)
        except Exception as e:
            logger.warning("FRED rate fetch failed: %s", e)

        facts = pre_gather_dart_facts(symbol, usd_krw_rate=usd_krw_rate)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved DART facts to cache: %s", cache_path)
        return facts

    peer_overrides = _load_peer_overrides_for(symbol)
    if peer_overrides:
        logger.info(
            "Peer overrides for %s from value chain brief: %s",
            symbol,
            peer_overrides,
        )

    # Partial-failure tolerant: if an individual tool errors (e.g. FMP quota
    # exhaustion mid-run), record the error text in place of the output so the
    # LLM sees which fields are missing and can flag them in Section 6. The
    # remaining tools still run.
    def _safe(name: str, fn: Callable[[], str]) -> str:
        try:
            return fn()
        except Exception as e:
            logger.warning("pre-gather %s failed: %s", name, e)
            return f"ERROR: {e}"

    facts: dict[str, str] = {}
    facts["cross_validate_quote"] = _safe(
        "cross_validate_quote", lambda: _exec_cross_validate_quote({"symbol": symbol})
    )
    facts["calculate_per"] = _safe(
        "calculate_per", lambda: _exec_calculate_per({"symbol": symbol})
    )
    facts["calculate_ev_ebitda"] = _safe(
        "calculate_ev_ebitda", lambda: _exec_calculate_ev_ebitda({"symbol": symbol})
    )
    facts["get_peer_multiples"] = _safe(
        "get_peer_multiples",
        lambda: _exec_get_peer_multiples(
            {
                "symbol": symbol,
                "max_peers": 5,
                "additional_peers": peer_overrides,
            }
        ),
    )
    facts["reverse_dcf"] = _safe(
        "reverse_dcf", lambda: _exec_reverse_dcf({"symbol": symbol})
    )

    # Read-only field fetch: avoids verify_number's MATCH/MISMATCH verdicts
    # contaminating the pre-gathered facts (we have no claim to verify at
    # this stage, we just want authoritative source values).
    for fld in _VERIFY_FIELDS:
        key = f"fetch.{fld}"
        facts[key] = _safe(
            key,
            lambda fld=fld: _exec_fetch_field({"field": fld, "symbol": symbol}),
        )

    # Phase 2: FRED macro snapshot for the Economist agent. Gracefully
    # degrades if the FRED key is missing — the Economist prompt handles
    # the "N/A" case by saying snapshot unavailable.
    facts["fred.macro_snapshot"] = _safe(
        "fred.macro_snapshot",
        lambda: format_macro_snapshot(get_macro_snapshot()),
    )

    # Phase 3D: 10-K RAG excerpts for qualitative narrative grounding.
    # Four fixed queries (business / moat / risk_factors / mdna) retrieve
    # the most relevant passages from the indexed filing and feed them to
    # every agent downstream. Fails soft: if the ticker is not in SEC
    # EDGAR (e.g. Korean listings) all four entries become uniform ERROR
    # strings so the schema is stable.
    try:
        from wise_investor.rag.integration import gather_and_format_for_pre_gather
        edgar_bodies = gather_and_format_for_pre_gather(symbol)
    except Exception as e:
        logger.warning("RAG gather wrapper failed for %s: %s", symbol, e)
        edgar_bodies = {}
    facts.update(edgar_bodies)

    # Phase 3E-2: geopolitical context for the Economist.
    # GDELT themes (ECON_TRADE_SANCTIONS, TRADE_WAR, EPU_POLICY) + Google
    # News keyword feed give the Economist event-level context on top of
    # FRED's numerical indicators. Fails soft per-source: each failure
    # turns into an "ERROR: ..." string in its facts entry; the Economist
    # prompt renders these as data-gap notes.
    try:
        from wise_investor.geopolitics.snapshot import (
            get_geopolitics_snapshot,
            format_geopolitics_snapshot,
        )
        geo_snapshot = get_geopolitics_snapshot(symbol)
        facts["geo.snapshot"] = format_geopolitics_snapshot(geo_snapshot)
    except Exception as e:
        logger.warning("geopolitics snapshot failed for %s: %s", symbol, e)
        facts["geo.snapshot"] = f"ERROR: geopolitics snapshot unavailable: {e}"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved facts to cache: %s", cache_path)
    return facts


def render_facts_block(facts: dict[str, str]) -> str:
    """Format the gathered facts as a single user-message block for the LLM.

    Uses XML-style tagging per Anthropic's prompt-engineering guidance — models
    lock onto tagged source blocks more reliably than markdown-heading sections,
    and the tag name becomes a canonical citation key (e.g. "<tool_output
    name='calculate_per'>" → "[Source: calculate_per]").
    """
    sections = []
    for name, output in facts.items():
        sections.append(
            f'<tool_output name="{name}">\n{output}\n</tool_output>'
        )
    return "\n\n".join(sections)


def _run_synthesis_once(
    system_prompt: str,
    user_prompt: str,
    model: str,
    sampling: "SamplingConfig | None" = None,
    keep_alive: str | int | None = None,
    log_fn: Callable[[str], None] | None = None,
    agent_for_config: str = "analyst",
) -> tuple[str, float]:
    """Single LLM call for synthesis. Returns (final_text, elapsed_sec).

    Routed through the active LLMBackend so MAFIS stays portable
    across Ollama / MLX / llama.cpp / OpenAI-compat. `sampling` is
    optional — when omitted we resolve the recommended config for
    `agent_for_config` so callers that haven't been migrated yet
    still get sensible defaults.

    `keep_alive` is forwarded to the backend's `chat` as a kwarg.
    Only Ollama's backend acts on it (model swap optimization);
    other backends ignore unknown kwargs.
    """
    from wise_investor.llm import get_agent_config, get_backend

    log = log_fn or (lambda m: logger.info(m))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    backend = get_backend()
    if sampling is None:
        sampling = get_agent_config(agent_for_config, backend=backend.name).sampling

    chat_kwargs: dict[str, Any] = {}
    if keep_alive is not None:
        chat_kwargs["keep_alive"] = keep_alive

    t0 = time.perf_counter()
    log(f"[synthesis] {model} (keep_alive={keep_alive}) on {backend.name}")
    response = backend.chat(
        messages=messages, model=model, sampling=sampling, **chat_kwargs
    )
    elapsed = time.perf_counter() - t0
    text = response.content or ""
    log(f"[synthesis] {model} done in {elapsed:.1f}s ({len(text)} chars)")
    return text, elapsed


def run_analyst_synthesis(
    system_prompt: str,
    task_prompt: str,
    facts: dict[str, str],
    model: str | None = None,
    sampling: "SamplingConfig | None" = None,
    log_fn: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Synthesis-only LLM call: no tool calls, just narrative over given facts.

    The LLM is told the facts are already gathered by Python; its only job is
    to compose the seven-section report citing those facts. This removes the
    "decide when to call tools" step that small local models are unreliable at.
    """
    from wise_investor.llm import get_agent_config, get_backend

    log = log_fn or (lambda m: logger.info(m))

    backend = get_backend()
    if model is None or sampling is None:
        cfg = get_agent_config("analyst", backend=backend.name)
        model = model or cfg.model
        sampling = sampling or cfg.sampling

    # Anthropic research: long source documents belong at the TOP of the prompt,
    # with the question / instructions last (up to 30% quality lift). We put
    # tool facts first, value chain next (already inside task_prompt), and the
    # report template instructions last.
    facts_block = render_facts_block(facts)
    combined_user = (
        "<pre_gathered_tool_outputs>\n"
        "These are the complete, authoritative numeric facts for this report. "
        "Python has already run all six Phase 1A calculation tools. You MUST NOT "
        "state any number that does not appear in a <tool_output> block below. "
        "When you cite a number, name its tool: [Source: calculate_per] or "
        "[Source: fetch.revenue]. A 'Warnings:' line is only meaningful when "
        "it says something OTHER than 'Warnings: none'. If every tool_output "
        "says 'Warnings: none', report 'No tool warnings' in Section 6; do NOT "
        "invent warnings from unrelated fields like 'Source value' or diff.\n\n"
        + facts_block
        + "\n</pre_gathered_tool_outputs>\n\n"
        + task_prompt
        + "\n\n"
        "--- Final output instructions ---\n"
        "Compose the seven-section markdown report using ONLY facts from "
        "<pre_gathered_tool_outputs> above and the value chain brief in the "
        "task prompt. Do not attempt any tool calls — they are not available "
        "and all needed data is already gathered."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": combined_user},
    ]

    t0 = time.perf_counter()
    log(f"[synthesis] calling {model} on {backend.name}")
    response = backend.chat(messages=messages, model=model, sampling=sampling)
    elapsed = time.perf_counter() - t0
    final = response.content or ""
    log(f"[synthesis] done in {elapsed:.1f}s")

    # Represent the pre-gather as fake tool trace entries so downstream tooling
    # (meta reporting, audit) sees which tools contributed data.
    trace = [
        {"name": name, "args": "pre-gathered", "ok": True, "output_preview": out[:300]}
        for name, out in facts.items()
    ]

    return AgentRunResult(
        final_text=final,
        tool_calls_made=len(facts),
        iterations=1,
        elapsed_sec=elapsed,
        tool_trace=trace,
    )


# ---------------------------------------------------------------------------
# Phase 1C: full crew synthesis (Analyst -> Valuer -> Skeptic)
# ---------------------------------------------------------------------------


def _wrap_user_prompt_with_facts(
    task_prompt: str,
    facts: dict[str, str],
    is_skeptic: bool = False,
    tips_block: str = "",
) -> str:
    """Prepend the pre-gathered facts block to any task-specific user prompt.

    `tips_block` is the output of `data.tip_feed.format_tips_block` — when
    non-empty, it's inserted between the facts block and the task prompt
    (after the citation rules, so agents see the user-tip context as
    an additional input rather than as part of the citable tool corpus).
    """
    facts_block = render_facts_block(facts)
    intro = (
        "<pre_gathered_tool_outputs>\n"
        "These are the complete, authoritative numeric facts for this report. "
        "Python has already run all Phase 1A calculation tools. You MUST NOT "
        "state any number that does not appear in a <tool_output> block below.\n\n"
        "=== UNIVERSAL CITATION RULE — applies to EVERY section ===\n"
        "Every single line of text that contains a numeric value (dollar "
        "amount, percentage, multiple, growth rate, ratio) MUST end with a "
        "[Source: <tool_name>] citation on the same line. This rule applies "
        "to:\n"
        "  - Bullet lists of financial metrics\n"
        "  - Narrative prose sentences with embedded numbers\n"
        "  - Comparison sentences ('X trades at 40 vs Y at 60')\n"
        "  - Questions for Skeptic (claim + number)\n"
        "  - Warnings and data-gap notes\n"
        "  - Skeptic rebuttals where a number appears\n"
        "One citation at the end of a line covers every number on that line. "
        "Multiple tools can be cited as [Source: calculate_per, get_peer_multiples].\n\n"
        "Bad example (DO NOT emit): 'NVIDIA trades at a PER of 40.73 and "
        "EV/EBITDA of 35.01, compared to Broadcom's PER of 75.70.'\n"
        "Good example (EMIT THIS): 'NVIDIA trades at a PER of 40.73 and "
        "EV/EBITDA of 35.01, compared to Broadcom's PER of 75.70 "
        "[Source: calculate_per, calculate_ev_ebitda, get_peer_multiples].'\n\n"
        "Code-fenced peer tables (```plaintext ... ```) are self-labelled by "
        "column headers; they do NOT need per-row citations. Tables outside "
        "code fences DO need citations.\n\n"
        "=== 10-K EXCERPT CITATIONS ===\n"
        "Qualitative claims sourced from `edgar.*` tool_output blocks "
        "(business segments, moat, risk factors, MD&A) must carry the "
        "[Source: 10-K <section>, filed <YYYY-MM-DD>] hint printed under "
        "each passage — copy it VERBATIM including the square brackets. "
        "Never paraphrase a 10-K excerpt without its citation. If no "
        "`edgar.*` passage supports a qualitative claim, either omit the "
        "claim or mark it as '[from earnings call transcript / public "
        "reporting — no 10-K passage]' so the Skeptic can audit.\n\n"
        "=== GEOPOLITICAL / NEWS CITATIONS ===\n"
        "Claims about recent events (export controls, tariffs, sanctions, "
        "M&A, regulatory actions) may cite the `geo.snapshot` tool_output "
        "block. Use the format: '[Source: Google News, <outlet>, <YYYY-MM-DD>]' "
        "for Google News headlines, or '[Source: GDELT <theme>, <domain>, "
        "<YYYY-MM-DD>]' for GDELT articles. The outlet/date fields come "
        "directly from the snapshot text — copy verbatim, do not invent. "
        "Geopolitical context should be used when it directly affects the "
        "target (supply chain, regulation, demand); generic macro events "
        "belong to the Economist's Rate/Inflation sections, not here.\n\n"
        + facts_block
        + "\n</pre_gathered_tool_outputs>\n\n"
    )
    if tips_block:
        intro += tips_block + "\n\n"
    return intro + task_prompt


@dataclass
class CrewRunResult:
    """Output of the full 6-agent crew synthesis pipeline (Phase 2 debate):
    Economist -> Analyst -> Valuer -> Skeptic -> Defender -> Steward.
    """

    symbol: str
    economist_text: str
    analyst_text: str
    valuer_text: str
    skeptic_text: str
    steward_text: str
    combined_markdown: str
    economist_elapsed: float
    analyst_elapsed: float
    valuer_elapsed: float
    skeptic_elapsed: float
    steward_elapsed: float
    pre_gather_elapsed: float
    total_elapsed: float
    facts_used: dict[str, str]
    economist_model: str
    analyst_model: str
    valuer_model: str
    skeptic_model: str
    steward_model: str
    # Phase 2 debate round — optional for backwards compat with any test
    # or script that instantiates CrewRunResult without the defender.
    defender_text: str = ""
    defender_elapsed: float = 0.0
    defender_model: str = ""


def run_crew_synthesis(
    symbol: str,
    value_chain_text: str,
    facts: dict[str, str],
    economist_system: str,
    economist_user_prompt_builder: Callable[[str, str], str],
    analyst_system: str,
    analyst_task: str,
    valuer_system: str,
    valuer_user_prompt_builder: Callable[[str, str, str], str],
    skeptic_system: str,
    skeptic_user_prompt_builder: Callable[[str, str, str, str], str],
    steward_system: str,
    steward_user_prompt_builder: Callable[..., str],
    defender_system: str | None = None,
    defender_user_prompt_builder: Callable[[str, str, str, str], str] | None = None,
    economist_model_name: str | None = None,
    analyst_model_name: str | None = None,
    valuer_model_name: str | None = None,
    skeptic_model_name: str | None = None,
    steward_model_name: str | None = None,
    defender_model_name: str | None = None,
    run_tag: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> CrewRunResult:
    """Run the Phase 2 full pipeline:
    Economist -> Analyst -> Valuer -> Skeptic -> Defender -> Steward.

    Debate structure: Skeptic produces 5 attacks. Defender (optional,
    skipped if defender_system is None) responds DEFENDED/CONCEDED to
    each. Steward translates Defender labels into NEUTRALIZED/SURVIVED
    for the discipline matrix — no independent judgment.

    Model swap strategy: Economist/Analyst/Valuer/Defender share Qwen
    (no swap). keep_alive="0" after Valuer so Skeptic's Llama can load.
    keep_alive="0" after Skeptic so Qwen can load back for Defender +
    Steward.
    """
    log = log_fn or (lambda m: logger.info(m))

    e_model = economist_model_name or settings.analyst_model  # Qwen default
    a_model = analyst_model_name or settings.analyst_model
    v_model = valuer_model_name or settings.valuer_model
    s_model = skeptic_model_name or settings.skeptic_model
    st_model = steward_model_name or settings.steward_model
    d_model = defender_model_name or settings.analyst_model  # share with Analyst

    t_start = time.perf_counter()

    # -- User tips: NOT injected into agent prompts. Per
    # docs/constitution.md §7, the tip channel is decoupled from
    # analysis triggering: tips are still logged via the bot
    # (TipStore / classifier), but they NEVER reach any LLM in any
    # stage. Universe membership and screening must remain a function
    # of objective criteria + the rubric (Commitment 1), not of which
    # messages the user happens to forward.
    #
    # Step 8 of the v2 work order will rebuild the tip surface as a
    # post-Stage-4 *annotation* (the user sees "you mentioned this N
    # days ago" alongside the system's verdict) — purely metadata,
    # never prompt context. Until that surface is built, this block
    # stays empty.
    tips_block = ""
    tips_bundle = None

    # -- Economist (macro backdrop, reads value chain + macro snapshot)
    log(f"[crew] Economist on {e_model}")
    economist_task = economist_user_prompt_builder(symbol, value_chain_text)
    economist_user = _wrap_user_prompt_with_facts(
        economist_task, facts, tips_block=tips_block
    )
    economist_text, economist_elapsed = _run_synthesis_once(
        system_prompt=economist_system,
        user_prompt=economist_user,
        model=e_model,
        log_fn=log_fn,
    )

    # -- Analyst (reads Economist context via prepended macro section)
    log(f"[crew] Analyst on {a_model}")
    analyst_task_with_macro = (
        "The Economist has already written the macro-environment section "
        "below. Reference it when it matters to your business or financial "
        "analysis (e.g. 'per the Economist, the rate cycle is HOLDING'). "
        "Do not repeat macro numbers the Economist already cited.\n\n"
        "<economist_section>\n"
        + economist_text
        + "\n</economist_section>\n\n"
        + analyst_task
    )
    analyst_user = _wrap_user_prompt_with_facts(
        analyst_task_with_macro, facts, tips_block=tips_block
    )
    analyst_text, analyst_elapsed = _run_synthesis_once(
        system_prompt=analyst_system,
        user_prompt=analyst_user,
        model=a_model,
        log_fn=log_fn,
    )

    # -- Valuer (reads Analyst output)
    log(f"[crew] Valuer on {v_model}")
    valuer_task = valuer_user_prompt_builder(symbol, value_chain_text, analyst_text)
    valuer_user = _wrap_user_prompt_with_facts(
        valuer_task, facts, tips_block=tips_block
    )
    valuer_unload = "0" if s_model != v_model else None
    valuer_text, valuer_elapsed = _run_synthesis_once(
        system_prompt=valuer_system,
        user_prompt=valuer_user,
        model=v_model,
        keep_alive=valuer_unload,
        log_fn=log_fn,
    )

    # -- Skeptic (reads Analyst + Valuer)
    log(f"[crew] Skeptic on {s_model}")
    skeptic_task = skeptic_user_prompt_builder(
        symbol, value_chain_text, analyst_text, valuer_text
    )
    skeptic_user = _wrap_user_prompt_with_facts(
        skeptic_task, facts, is_skeptic=True, tips_block=tips_block
    )
    # Unload Skeptic model if the next stage (Defender or Steward) runs on
    # a different model (typical: Skeptic=Llama, Defender/Steward=Qwen).
    next_after_skeptic = d_model if defender_system else st_model
    skeptic_unload = "0" if next_after_skeptic != s_model else None
    skeptic_text, skeptic_elapsed = _run_synthesis_once(
        system_prompt=skeptic_system,
        user_prompt=skeptic_user,
        model=s_model,
        keep_alive=skeptic_unload,
        log_fn=log_fn,
    )

    # -- Defender (optional Phase 2 debate round; reads Analyst + Valuer + Skeptic)
    defender_text = ""
    defender_elapsed = 0.0
    if defender_system and defender_user_prompt_builder:
        log(f"[crew] Defender on {d_model}")
        defender_task = defender_user_prompt_builder(
            symbol, analyst_text, valuer_text, skeptic_text
        )
        defender_user = _wrap_user_prompt_with_facts(
            defender_task, facts, tips_block=tips_block
        )
        # Unload Defender model if Steward runs on a different model.
        defender_unload = "0" if st_model != d_model else None
        defender_text, defender_elapsed = _run_synthesis_once(
            system_prompt=defender_system,
            user_prompt=defender_user,
            model=d_model,
            keep_alive=defender_unload,
            log_fn=log_fn,
        )

    # -- Steward (reads Analyst + Valuer + Skeptic + Defender)
    log(f"[crew] Steward on {st_model}")
    steward_task = steward_user_prompt_builder(
        symbol,
        value_chain_text,
        analyst_text,
        valuer_text,
        skeptic_text,
        defender_text,
    )
    steward_user = _wrap_user_prompt_with_facts(
        steward_task, facts, tips_block=tips_block
    )
    steward_text, steward_elapsed = _run_synthesis_once(
        system_prompt=steward_system,
        user_prompt=steward_user,
        model=st_model,
        log_fn=log_fn,
    )

    # -- Steward discipline audit (Python post-check)
    # Empirically the 7B model emits SURVIVED/NEUTRALIZED labels correctly
    # but then picks a Verdict that violates the label→verdict matrix.
    # This check parses the labels deterministically and appends a System
    # Audit note when the verdict is too optimistic. Narrative is left
    # verbatim; only the appended note carries the corrected verdict.
    #
    # CRITICAL: feed the audit a document that INCLUDES the Defender
    # section. The Defender-aware audit path only fires when it can
    # find `# Part N · Defender` in the input; passing only
    # `steward_text` hides the Defender labels and collapses the audit
    # to Steward's self-reporting (which is exactly the failure mode
    # we shipped the audit to prevent — see commit 04b4c0a).
    from wise_investor.agents.steward_audit import (
        audit_steward_section,
        apply_audit_to_section,
    )
    steward_text_for_audit = steward_text
    if defender_text:
        # Give the audit enough context to find the Defender heading.
        # The Part number ("Part 5") only needs to match the regex
        # `# Part \d+ · Defender`, so the exact value is unimportant.
        steward_text_for_audit = (
            "# Part 5 · Defender\n\n"
            + defender_text
            + "\n\n---\n\n# Part 6 · Steward\n\n"
            + steward_text
        )
    audit = audit_steward_section(steward_text_for_audit)
    if audit.violation:
        log(
            f"[crew] Steward AUDIT VIOLATION: {audit.verdict} C{audit.conviction} "
            f"→ {audit.corrected_verdict} C{audit.corrected_conviction} "
            f"(N={audit.neutralized_count}, S={audit.survived_count}, "
            f"defender={audit.defender_defended_count}D/{audit.defender_conceded_count}C, "
            f"mistranslated={audit.steward_mistranslated})"
        )
        steward_text = apply_audit_to_section(steward_text, audit)

    # -- Mark injected tips as consumed by this run so the next crew
    # for the same ticker doesn't re-inject them. We do this AFTER the
    # audit so a failed audit (which doesn't currently abort, but might
    # in future) doesn't leave tips unmarked. Failure is non-fatal —
    # the report has already been produced; a missed mark just means
    # one duplicate injection on the next run.
    if run_tag and tips_bundle is not None and not tips_bundle.is_empty:
        try:
            from wise_investor.data.tip_feed import mark_consumed_for_run
            n_marked = mark_consumed_for_run(tips_bundle.all_tips(), run_tag)
            log(f"[crew] tips: marked {n_marked} as consumed by {run_tag}")
        except Exception as e:
            logger.warning(
                "tip_feed mark_consumed failed (%s); next run may "
                "re-inject these tips.",
                e,
            )

    total_elapsed = time.perf_counter() - t_start

    # -- Citation grounding audit (Python post-check over the whole report)
    # Scans every `[Source: edgar.*]` citation for numeric claims and
    # verifies each claim appears in the indexed 10-K passages. Flags
    # hallucinations where the LLM attached an edgar citation to a
    # fabricated number.
    from wise_investor.quality.citation_audit import (
        audit_edgar_citations,
        render_citation_audit_section,
    )
    try:
        full_text_for_audit = (
            economist_text + "\n" + analyst_text + "\n" +
            valuer_text + "\n" + skeptic_text + "\n" + steward_text
        )
        citation_audit_result = audit_edgar_citations(
            full_text_for_audit, symbol=symbol
        )
        citation_audit_markdown = render_citation_audit_section(
            citation_audit_result
        )
        if citation_audit_result.ungrounded:
            log(
                f"[crew] Citation audit: "
                f"{len(citation_audit_result.ungrounded)} ungrounded claim(s) "
                f"across {citation_audit_result.citations_checked} "
                f"edgar.* citation(s)"
            )
    except Exception as e:
        logger.warning("citation audit crashed: %s", e)
        citation_audit_markdown = ""

    combined = _compose_combined_report(
        symbol=symbol,
        economist_text=economist_text,
        analyst_text=analyst_text,
        valuer_text=valuer_text,
        skeptic_text=skeptic_text,
        defender_text=defender_text,
        steward_text=steward_text,
        economist_model=e_model,
        analyst_model=a_model,
        valuer_model=v_model,
        skeptic_model=s_model,
        defender_model=d_model if defender_text else "",
        steward_model=st_model,
    )
    # Append citation audit section (only if violations were found).
    if citation_audit_markdown:
        combined = combined.rstrip() + "\n" + citation_audit_markdown

    return CrewRunResult(
        symbol=symbol.upper(),
        economist_text=economist_text,
        analyst_text=analyst_text,
        valuer_text=valuer_text,
        skeptic_text=skeptic_text,
        defender_text=defender_text,
        steward_text=steward_text,
        combined_markdown=combined,
        economist_elapsed=economist_elapsed,
        analyst_elapsed=analyst_elapsed,
        valuer_elapsed=valuer_elapsed,
        skeptic_elapsed=skeptic_elapsed,
        defender_elapsed=defender_elapsed,
        steward_elapsed=steward_elapsed,
        pre_gather_elapsed=0.0,
        total_elapsed=total_elapsed,
        facts_used=facts,
        economist_model=e_model,
        analyst_model=a_model,
        valuer_model=v_model,
        skeptic_model=s_model,
        defender_model=d_model if defender_text else "",
        steward_model=st_model,
    )


def _compose_combined_report(
    symbol: str,
    economist_text: str,
    analyst_text: str,
    valuer_text: str,
    skeptic_text: str,
    steward_text: str,
    economist_model: str,
    analyst_model: str,
    valuer_model: str,
    skeptic_model: str,
    steward_model: str,
    defender_text: str = "",
    defender_model: str = "",
) -> str:
    """Assemble the five-or-six agent outputs into a single markdown document.

    The Defender stage is optional for backwards compat with pre-Phase-2
    test runs; when present, Steward becomes Part 6 and the report
    title says "Full 6-Agent Crew" instead of 5.
    """
    has_defender = bool(defender_text)
    agent_count = 6 if has_defender else 5
    roster = (
        "Economist + Analyst + Valuer + Skeptic + Defender + Steward"
        if has_defender
        else "Economist + Analyst + Valuer + Skeptic + Steward"
    )
    model_line = (
        f"_Models: Economist/{economist_model} · Analyst/{analyst_model} · "
        f"Valuer/{valuer_model} · Skeptic/{skeptic_model} · "
    )
    if has_defender:
        model_line += f"Defender/{defender_model} · "
    model_line += f"Steward/{steward_model}_\n\n"

    header = (
        f"# {symbol} — Equity Research Note (Phase 2 — Full {agent_count}-Agent Crew)\n\n"
        f"_Generated by the Wise Investor System: {roster}._\n"
        + model_line
        + "---\n\n"
    )
    divider = "\n\n---\n\n"
    body = (
        f"# Part 1 · Economist\n\n{economist_text.strip()}"
        + divider
        + f"# Part 2 · Analyst\n\n{analyst_text.strip()}"
        + divider
        + f"# Part 3 · Valuer\n\n{valuer_text.strip()}"
        + divider
        + f"# Part 4 · Skeptic\n\n{skeptic_text.strip()}"
    )
    if has_defender:
        body += divider + f"# Part 5 · Defender\n\n{defender_text.strip()}"
        body += divider + f"# Part 6 · Steward\n\n{steward_text.strip()}\n"
    else:
        body += divider + f"# Part 5 · Steward\n\n{steward_text.strip()}\n"
    return header + body


# ---------------------------------------------------------------------------
# Legacy tool-calling agent loop (kept for compatibility, not used in 1C)
# ---------------------------------------------------------------------------


def _execute_tool_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Execute a single tool_call dict; return (output_text, trace_entry)."""
    fn = call["function"]
    name = fn["name"]
    raw_args = fn.get("arguments", {})
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = dict(raw_args)

    entry: dict[str, Any] = {"name": name, "args": args}

    tool = TOOL_LOOKUP.get(name)
    if tool is None:
        out = f"ERROR: unknown tool '{name}'. Available: {list(TOOL_LOOKUP)}"
        entry["ok"] = False
        entry["output_preview"] = out
        return out, entry

    try:
        out = tool.executor(args)
        entry["ok"] = True
    except Exception as e:
        out = f"ERROR executing {name}: {e}"
        entry["ok"] = False

    entry["output_preview"] = out[:300]
    return out, entry


def run_agent(
    system_prompt: str,
    user_prompt: str,
    model: str | None = None,
    sampling: "SamplingConfig | None" = None,
    max_iterations: int = 30,
    log_fn: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Run a single-agent tool-using loop through the active LLMBackend.

    Stops when the model returns a message with no tool_calls (final answer) or
    when max_iterations is exceeded. Tool-calling support is backend-dependent:
    Ollama and OpenAI-compat carry it through; MLX/llama.cpp would need a
    server (mlx_lm.server) plus the openai_compat backend.
    """
    from wise_investor.llm import get_agent_config, get_backend

    backend = get_backend()
    if model is None or sampling is None:
        cfg = get_agent_config("analyst", backend=backend.name)
        model = model or cfg.model
        sampling = sampling or cfg.sampling

    log = log_fn or (lambda m: logger.info(m))

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    t0 = time.perf_counter()
    tool_calls_made = 0
    trace: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        log(f"[iter {iteration}] calling {model}")
        response = backend.chat(
            messages=messages,
            model=model,
            sampling=sampling,
            tools=TOOL_SPECS,
        )
        # The runner downstream still operates on a "message dict".
        # Reconstruct one from LLMResponse so all the indexing logic
        # below stays unchanged.
        tool_calls = response.extra.get("tool_calls") or []
        msg = {
            "content": response.content or "",
            "tool_calls": tool_calls,
        }

        # Record the assistant turn (with whatever content / tool_calls it produced).
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content", "") or "",
                "tool_calls": tool_calls,
            }
        )

        if not tool_calls:
            elapsed = time.perf_counter() - t0
            final = msg.get("content", "") or ""
            log(
                f"[done] iters={iteration} tool_calls={tool_calls_made} "
                f"elapsed={elapsed:.1f}s"
            )
            return AgentRunResult(
                final_text=final,
                tool_calls_made=tool_calls_made,
                iterations=iteration,
                elapsed_sec=elapsed,
                tool_trace=trace,
            )

        # Execute every tool call in this turn; feed results back.
        for call in tool_calls:
            tool_calls_made += 1
            log(f"[iter {iteration}] tool_call: {call['function']['name']}")
            output, entry = _execute_tool_call(call)
            trace.append(entry)
            messages.append(
                {
                    "role": "tool",
                    "name": call["function"]["name"],
                    "content": output,
                }
            )

    elapsed = time.perf_counter() - t0
    log(f"[abort] hit max_iterations={max_iterations}")
    return AgentRunResult(
        final_text=(
            "ERROR: Analyst agent exceeded max_iterations without producing a "
            "final report. Inspect tool_trace for what it was looping on."
        ),
        tool_calls_made=tool_calls_made,
        iterations=max_iterations,
        elapsed_sec=elapsed,
        tool_trace=trace,
    )
