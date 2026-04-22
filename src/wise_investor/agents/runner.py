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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import ollama

from wise_investor.config import settings


FACTS_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "facts_cache"
from wise_investor.data.cross_validate import cross_validate_quote
from wise_investor.data.finnhub import FinnhubClient as FMPClient  # alias for minimal call-site change
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
    with FMPClient() as c:
        t = get_peer_multiples_impl(symbol, client=c, max_peers=max_peers)
    lines = [
        f"Peer multiples table — target {t.target_symbol} (as of {t.as_of or 'unknown'})",
        f"{'Symbol':<8} {'Name':<28} {'MktCap':>10} {'PER':>8} {'EV/EBITDA':>10}",
        "-" * 68,
    ]
    for row in t.rows:
        lines.append(
            f"{row.symbol:<8} {(row.name or '')[:28]:<28} "
            f"{_fmt_num(row.market_cap):>10} {_fmt_num(row.per):>8} "
            f"{_fmt_num(row.ev_ebitda):>10}"
        )
        if row.warnings:
            lines.append(f"         warnings: {'; '.join(row.warnings)}")
    return "\n".join(lines)


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
        lambda: _exec_get_peer_multiples({"symbol": symbol, "max_peers": 5}),
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


def run_analyst_synthesis(
    system_prompt: str,
    task_prompt: str,
    facts: dict[str, str],
    model: str | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Synthesis-only LLM call: no tool calls, just narrative over given facts.

    The LLM is told the facts are already gathered by Python; its only job is
    to compose the seven-section report citing those facts. This removes the
    "decide when to call tools" step that small local models are unreliable at.
    """
    model = model or settings.analyst_model
    log = log_fn or (lambda m: logger.info(m))

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
    log(f"[synthesis] calling {model}")
    resp = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
    )
    elapsed = time.perf_counter() - t0
    final = resp["message"].get("content", "") or ""
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
    max_iterations: int = 30,
    log_fn: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Run a single-agent tool-using loop against Ollama.

    Stops when the model returns a message with no tool_calls (final answer) or
    when max_iterations is exceeded.
    """
    model = model or settings.analyst_model
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
        resp = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOL_SPECS,
            options={
                "temperature": settings.llm_temperature,
                "seed": settings.llm_seed,
            },
        )
        msg = resp["message"]
        tool_calls = msg.get("tool_calls") or []

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
