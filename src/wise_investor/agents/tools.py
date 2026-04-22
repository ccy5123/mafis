"""CrewAI Tool wrappers around the Phase 1A calculation layer.

Each wrapper:
- Accepts only primitive args (string ticker, optional floats) so CrewAI + Ollama
  can fill them in reliably.
- Constructs its own FMPClient per call (httpx.Client is cheap; this keeps the
  tool stateless and safe across sequential agent calls).
- Returns a plain-text block with explicit source citations so the LLM can quote
  values verbatim into reports without re-deriving numbers (design-v2.2 §7).

Docstrings are written for the LLM — keep them tight, action-oriented, and with
one canonical example.
"""

from __future__ import annotations

from typing import Any

from crewai.tools import tool

from wise_investor.data.cross_validate import cross_validate_quote
from wise_investor.data.fmp import FMPClient
from wise_investor.tools.dcf import reverse_dcf as reverse_dcf_impl
from wise_investor.tools.valuation import (
    calculate_ev_ebitda as calculate_ev_ebitda_impl,
    calculate_per as calculate_per_impl,
    get_peer_multiples as get_peer_multiples_impl,
)
from wise_investor.tools.verify import (
    list_supported_fields,
    verify_number as verify_number_impl,
)


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
# Tool 1: cross-validate
# ---------------------------------------------------------------------------


@tool("cross_validate_quote")
def tool_cross_validate_quote(symbol: str) -> str:
    """Compare FMP and yfinance quotes for a US ticker and flag any field whose
    values diverge by more than 5%.

    Use this at the start of an analysis to confirm the market data you will
    rely on is consistent across sources.

    Args:
        symbol: Uppercase US ticker (e.g. "NVDA", "AAPL").

    Returns a plain-text comparison table citing both sources.
    """
    with FMPClient() as c:
        r = cross_validate_quote(symbol, fmp=c)
    lines = [
        f"Cross-validation for {symbol} (threshold {r.threshold_pct}%):",
    ]
    for cmp in r.comparisons:
        flag = (
            "OK"
            if cmp.within_threshold is True
            else "DIVERGES"
            if cmp.within_threshold is False
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
    lines.append(
        "Flagged: yes" if r.any_flagged else "Flagged: no"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2: PER
# ---------------------------------------------------------------------------


@tool("calculate_per")
def tool_calculate_per(symbol: str) -> str:
    """Compute PER (Price / EPS diluted) from the latest annual income statement
    and compare against FMP's reported PER for the same fiscal year.

    Use this when the report needs a PER figure. Do NOT state a PER that was not
    returned by this tool — always cite the output.

    Args:
        symbol: Uppercase US ticker.
    """
    with FMPClient() as c:
        r = calculate_per_impl(symbol, client=c)
    lines = [
        f"PER for {symbol}",
        f"Source: Python-computed from FMP data; fiscal year end {r.as_of or 'unknown'}",
        f"Computed PER: {_fmt_num(r.computed)}",
        "Inputs:",
        f"  - current price: {_fmt_num(r.inputs.get('price'))} (FMP /quote)",
        f"  - EPS diluted: {_fmt_num(r.inputs.get('eps_diluted_latest_annual'), 2)} "
        f"(FMP /income-statement annual {r.as_of})",
        f"FMP-reported PER (same fiscal year): {_fmt_num(r.fmp_reported)}",
        f"Divergence vs FMP: "
        + (f"{r.diff_pct_vs_fmp:.2f}%" if r.diff_pct_vs_fmp is not None else "N/A")
        + " (5%+ divergence usually reflects price movement since fiscal year end)",
        _fmt_warnings(r.warnings),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3: EV/EBITDA
# ---------------------------------------------------------------------------


@tool("calculate_ev_ebitda")
def tool_calculate_ev_ebitda(symbol: str) -> str:
    """Compute EV/EBITDA from FMP enterprise value and annual EBITDA, then
    cross-check against FMP's reported ev_to_ebitda for the same fiscal year.

    Args:
        symbol: Uppercase US ticker.
    """
    with FMPClient() as c:
        r = calculate_ev_ebitda_impl(symbol, client=c)
    lines = [
        f"EV/EBITDA for {symbol}",
        f"Source: Python-computed from FMP data; fiscal year end {r.as_of or 'unknown'}",
        f"Computed EV/EBITDA: {_fmt_num(r.computed)}",
        "Inputs:",
        f"  - Enterprise Value: {_fmt_num(r.inputs.get('enterprise_value'))} "
        f"(FMP /enterprise-values annual {r.as_of})",
        f"  - EBITDA: {_fmt_num(r.inputs.get('ebitda_latest_annual'))} "
        f"(FMP /income-statement annual {r.as_of})",
        f"FMP-reported EV/EBITDA (same fiscal year): {_fmt_num(r.fmp_reported)}",
        f"Divergence vs FMP: "
        + (f"{r.diff_pct_vs_fmp:.2f}%" if r.diff_pct_vs_fmp is not None else "N/A"),
        _fmt_warnings(r.warnings),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4: peer multiples
# ---------------------------------------------------------------------------


@tool("get_peer_multiples")
def tool_get_peer_multiples(symbol: str, max_peers: int = 5) -> str:
    """Return a peer-comparison table of PER and EV/EBITDA for the target and
    its top peers from FMP /stock-peers.

    Rows with missing values mean that peer's underlying data was unavailable;
    EXCLUDE such rows from median/mean/percentile calculations and list them
    under a Data Gaps note in the report.

    Args:
        symbol: Uppercase US ticker for the target company.
        max_peers: How many peers to include beyond the target (default 5).
    """
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


# ---------------------------------------------------------------------------
# Tool 5: reverse DCF
# ---------------------------------------------------------------------------


@tool("reverse_dcf")
def tool_reverse_dcf(
    symbol: str,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    high_growth_years: int = 10,
) -> str:
    """Solve for the annual FCF growth rate the market is currently pricing in,
    given the current market cap and latest annual free cash flow.

    This is a reverse DCF: instead of picking assumptions to compute a price,
    we take the price and recover the implied growth. Report the result
    alongside a sanity check: is this growth plausible vs history and peers?

    Args:
        symbol: Uppercase US ticker.
        discount_rate: WACC assumption (default 0.10 = 10%).
        terminal_growth: Long-run growth assumption (default 0.025 = 2.5%).
        high_growth_years: Years of explicit stage-1 growth (default 10).
    """
    with FMPClient() as c:
        r = reverse_dcf_impl(
            symbol,
            client=c,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            high_growth_years=high_growth_years,
        )
    lines = [
        f"Reverse DCF for {symbol}",
        f"Source: Python-solved; fiscal year end {r.as_of or 'unknown'}",
        f"Implied annual FCF growth (stage 1, {high_growth_years}y): "
        f"{_fmt_pct(r.implied_growth_rate)}",
        "Inputs:",
        f"  - market cap: {_fmt_num(r.current_market_cap)} (FMP /quote)",
        f"  - latest annual FCF: {_fmt_num(r.inputs.get('fcf_latest_annual'))} "
        f"(FMP /cash-flow-statement {r.as_of})",
        f"  - FCF source: {r.inputs.get('fcf_source', 'N/A')}",
        f"  - discount rate (WACC assumption): {_fmt_pct(discount_rate)}",
        f"  - terminal growth: {_fmt_pct(terminal_growth)}",
        f"  - high-growth years: {high_growth_years}",
        _fmt_warnings(r.warnings),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 6: verify_number
# ---------------------------------------------------------------------------


_SUPPORTED_FIELDS_BLURB = ", ".join(list_supported_fields())


@tool("verify_number")
def tool_verify_number(
    claim: float,
    field: str,
    symbol: str,
    tolerance_pct: float = 1.0,
) -> str:
    """Check whether a numeric claim matches the authoritative source for that
    field on the given ticker. Use this when you need to confirm a number that
    another agent or document has stated.

    Args:
        claim: The number being verified (numeric, not a string).
        field: One of the supported field names (aliases like "pe", "fcf" also
            accepted). See the list in the tool description.
        symbol: Uppercase US ticker.
        tolerance_pct: Percentage tolerance; default 1.0 means claim within
            ±1% of the source value counts as a match.
    """
    with FMPClient() as c:
        r = verify_number_impl(
            claim=claim, field=field, symbol=symbol, client=c, tolerance_pct=tolerance_pct
        )
    match_str = (
        "MATCH"
        if r.matches is True
        else "MISMATCH"
        if r.matches is False
        else "UNKNOWN (source unavailable)"
    )
    lines = [
        f"Verification for {symbol}.{r.field}",
        f"  claim: {_fmt_num(r.claim, 4)}",
        f"  source value: {_fmt_num(r.source_value, 4)}",
        f"  source: {r.source_citation}",
        f"  diff: {r.diff_pct if r.diff_pct is not None else 'N/A'}%"
        f" (tolerance {r.tolerance_pct}%)",
        f"  result: {match_str}",
        _fmt_warnings(r.warnings),
    ]
    return "\n".join(lines)


# Append the dynamic supported-fields list to the description so the LLM
# always sees the current set. Done via string concatenation because the
# decorator-generated description contains JSON schema braces that would
# confuse str.format.
tool_verify_number.description = (
    tool_verify_number.description
    + f"\n\nSupported fields: {_SUPPORTED_FIELDS_BLURB}"
)


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


ALL_TOOLS: list[Any] = [
    tool_cross_validate_quote,
    tool_calculate_per,
    tool_calculate_ev_ebitda,
    tool_get_peer_multiples,
    tool_reverse_dcf,
    tool_verify_number,
]
