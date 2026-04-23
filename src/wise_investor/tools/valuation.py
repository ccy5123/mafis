"""Valuation calculation tools — Finnhub-backed.

Implements the "LLM is judgment, Python is calculation" principle: every numeric
ratio consumed by agents in reports is computed here from raw Finnhub data,
never inferred by the LLM (design-v2.2 §7).

Post Phase 1B migration, the data provider is Finnhub. Per-annual dollar
values come from /stock/financials-reported XBRL extraction; pre-computed
ratios come from /stock/metric; market cap from /stock/profile2.

Each tool returns a CalculationResult with:
- `computed`: our Python-calculated value
- `fmp_reported`: provider's pre-computed value for cross-verification
  (name retained for stability; now sourced from Finnhub /stock/metric)
- `diff_pct_vs_fmp`: percentage divergence between the two
- `inputs`: raw numbers used so reports can cite them verbatim
- `warnings`: any caveats
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from wise_investor.data.finnhub import (
    FinnhubClient,
    derive_ebitda,
    derive_free_cash_flow,
    extract_field,
    total_debt as fn_total_debt,
)


class CalculationResult(BaseModel):
    symbol: str
    metric: str
    computed: float | None
    fmp_reported: float | None  # retained field name; value now from Finnhub metric
    diff_pct_vs_fmp: float | None
    inputs: dict[str, Any]
    as_of: str | None = None
    warnings: list[str] = Field(default_factory=list)


class PeerMultipleRow(BaseModel):
    symbol: str
    name: str | None = None
    market_cap: float | None = None
    per: float | None = None
    ev_ebitda: float | None = None
    warnings: list[str] = Field(default_factory=list)


class PeerMultiplesTable(BaseModel):
    target_symbol: str
    as_of: str | None
    rows: list[PeerMultipleRow]  # peers that passed the sanity filter
    excluded: list[PeerMultipleRow] = Field(default_factory=list)
    override_sources: list[str] = Field(default_factory=list)


# Sanity filter thresholds. Peers whose headline multiples exceed these are
# excluded from the comparative table because they come from near-zero
# denominators (tiny earnings / EBITDA) and distort any median/quartile.
SANITY_MAX_PER = 300.0
SANITY_MAX_EV_EBITDA = 500.0


def _is_sane_peer(row: PeerMultipleRow) -> tuple[bool, str]:
    """Return (keep, reason). If keep=False, reason describes why excluded."""
    per = row.per
    ev = row.ev_ebitda
    if per is None and ev is None:
        return False, "both PER and EV/EBITDA unavailable"
    if per is not None and per > SANITY_MAX_PER:
        return False, f"PER={per:.2f} exceeds sanity threshold {SANITY_MAX_PER:.0f}x"
    if per is not None and per < 0:
        return False, f"PER={per:.2f} is negative (loss-making) — not comparable"
    if ev is not None and ev > SANITY_MAX_EV_EBITDA:
        return (
            False,
            f"EV/EBITDA={ev:.2f} exceeds sanity threshold {SANITY_MAX_EV_EBITDA:.0f}x",
        )
    if ev is not None and ev < 0:
        return False, f"EV/EBITDA={ev:.2f} is negative (loss-making) — not comparable"
    return True, ""


def _diff_pct(computed: float | None, reference: float | None) -> float | None:
    if computed is None or reference is None or reference == 0:
        return None
    return round(abs(computed - reference) / abs(reference) * 100.0, 3)


def _with_client(client: FinnhubClient | None) -> tuple[FinnhubClient, bool]:
    if client is not None:
        return client, False
    return FinnhubClient(), True


# ---------------------------------------------------------------------------
# calculate_per
# ---------------------------------------------------------------------------


def calculate_per(symbol: str, client: FinnhubClient | None = None) -> CalculationResult:
    """Compute PER = current price / latest annual diluted EPS.

    Cross-checks against Finnhub's peAnnual (same fiscal-year-end basis). Uses
    current price from /quote, EPS from latest 10-K via financials-reported.
    """
    fmp, owned = _with_client(client)
    warnings: list[str] = []
    inputs: dict[str, Any] = {}
    try:
        quote = fmp.quote(symbol)
        latest = fmp.latest_annual_financials(symbol)
        metric = fmp.metric(symbol)
    finally:
        if owned:
            fmp.close()

    price = quote.price
    inputs["price"] = price

    if latest is None:
        return CalculationResult(
            symbol=symbol.upper(),
            metric="PER",
            computed=None,
            fmp_reported=metric.metric.pe_annual,
            diff_pct_vs_fmp=None,
            inputs=inputs,
            warnings=["no annual financials available"],
        )

    eps = extract_field(latest, "eps_diluted")
    inputs["eps_diluted_latest_annual"] = eps
    inputs["fiscal_date"] = str(latest.end_date) if latest.end_date else None

    if eps is None:
        warnings.append("EPS diluted not found in latest annual filing (concept mapping missed)")
        computed: float | None = None
    elif eps <= 0:
        warnings.append(
            f"EPS <= 0 ({eps}); PER is not meaningful for unprofitable periods"
        )
        computed = None
    else:
        computed = round(price / eps, 3)

    fmp_reported = metric.metric.pe_annual
    return CalculationResult(
        symbol=symbol.upper(),
        metric="PER",
        computed=computed,
        fmp_reported=fmp_reported,
        diff_pct_vs_fmp=_diff_pct(computed, fmp_reported),
        inputs=inputs,
        as_of=str(latest.end_date) if latest.end_date else None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# calculate_ev_ebitda
# ---------------------------------------------------------------------------


def calculate_ev_ebitda(
    symbol: str, client: FinnhubClient | None = None
) -> CalculationResult:
    """Compute EV / EBITDA from Finnhub enterprise value (millions) and derived
    EBITDA (Operating Income + D&A) from the latest 10-K.

    Cross-checks against Finnhub's evEbitdaTTM. Note the TTM vs annual
    mismatch — the fmp_reported value uses TTM, our computed value uses
    latest annual; divergences of a few percent are expected.
    """
    fmp, owned = _with_client(client)
    warnings: list[str] = []
    inputs: dict[str, Any] = {}
    try:
        metric = fmp.metric(symbol)
        latest = fmp.latest_annual_financials(symbol)
    finally:
        if owned:
            fmp.close()

    ev_usd = metric.metric.enterprise_value_usd
    ebitda = derive_ebitda(latest) if latest is not None else None

    inputs["enterprise_value"] = ev_usd
    inputs["ebitda_latest_annual"] = ebitda
    inputs["fiscal_date"] = str(latest.end_date) if latest and latest.end_date else None

    if ev_usd is None or ebitda is None:
        missing = []
        if ev_usd is None:
            missing.append("enterprise_value")
        if ebitda is None:
            missing.append("ebitda (OperatingIncome + D&A)")
        warnings.append(f"{', '.join(missing)} unavailable")
        computed: float | None = None
    elif ebitda <= 0:
        warnings.append(
            f"EBITDA <= 0 ({ebitda}); EV/EBITDA is not meaningful for loss periods"
        )
        computed = None
    else:
        computed = round(ev_usd / ebitda, 3)

    fmp_reported = metric.metric.ev_ebitda_ttm
    if fmp_reported is not None and computed is not None:
        warnings.append(
            "Finnhub fmp_reported is TTM basis; computed is latest annual — "
            "divergence up to ~5% is expected."
        )

    return CalculationResult(
        symbol=symbol.upper(),
        metric="EV/EBITDA",
        computed=computed,
        fmp_reported=fmp_reported,
        diff_pct_vs_fmp=_diff_pct(computed, fmp_reported),
        inputs=inputs,
        as_of=str(latest.end_date) if latest and latest.end_date else None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# get_peer_multiples
# ---------------------------------------------------------------------------


def get_peer_multiples(
    symbol: str,
    client: FinnhubClient | None = None,
    max_peers: int = 5,
    additional_peers: list[str] | None = None,
) -> PeerMultiplesTable:
    """Build a peer-comparison table of PER and EV/EBITDA using Finnhub.

    Finnhub /stock/peers returns symbol strings only, so we fetch profile
    (for name, market cap) and metric (for pre-computed peTTM, evEbitdaTTM)
    for each peer. Uses TTM values for peer comparison because they are the
    most comparable across companies with different fiscal year ends.

    `additional_peers` lets the caller inject hand-picked comparables (for
    example, from a value chain brief's Peer Override section). These are
    merged with Finnhub's auto-peers and de-duplicated. This matters for
    newly-spun-off companies where Finnhub's auto-peer algorithm has not
    yet latched onto the right cohort.

    A sanity filter (see `_is_sane_peer` and SANITY_MAX_* constants) moves
    peers with garbage multiples into a separate `excluded` list so the
    LLM sees what was rejected without using those numbers for comparison.
    """
    fmp, owned = _with_client(client)
    override_sources: list[str] = []
    try:
        peer_list = fmp.peers(symbol)
        symbols: list[str] = [symbol.upper()]

        # Auto-peers first (they get priority up to max_peers).
        for s in peer_list:
            s_up = s.upper()
            if s_up != symbol.upper() and s_up not in symbols:
                symbols.append(s_up)
            if len(symbols) >= max_peers + 1:
                break

        # Then merge in additional overrides (always included, even if they
        # push total count beyond max_peers — the whole point of overrides is
        # to guarantee the hand-picked cohort shows up).
        if additional_peers:
            for s in additional_peers:
                s_up = s.upper()
                if s_up != symbol.upper() and s_up not in symbols:
                    symbols.append(s_up)
                    override_sources.append(s_up)

        rows: list[PeerMultipleRow] = []
        target_as_of: str | None = None

        for sym in symbols:
            row_warnings: list[str] = []
            name: str | None = None
            mkt_cap: float | None = None
            per: float | None = None
            ev_eb: float | None = None

            try:
                profile = fmp.profile(sym)
                name = profile.name
                mkt_cap = profile.market_cap_usd
            except Exception as e:
                row_warnings.append(f"profile failed: {e}")

            try:
                m = fmp.metric(sym)
                # Prefer peTTM, fall back to peAnnual
                per = m.metric.pe_ttm or m.metric.pe_annual
                ev_eb = m.metric.ev_ebitda_ttm
            except Exception as e:
                row_warnings.append(f"metric failed: {e}")

            if sym == symbol.upper() and target_as_of is None:
                # Prefer fiscal date from financials for the target
                try:
                    latest = fmp.latest_annual_financials(sym)
                    if latest and latest.end_date:
                        target_as_of = str(latest.end_date)
                except Exception:
                    pass

            if per is None:
                row_warnings.append("PER unavailable — peTTM and peAnnual both missing")
            if ev_eb is None:
                row_warnings.append("EV/EBITDA unavailable — evEbitdaTTM missing")

            rows.append(
                PeerMultipleRow(
                    symbol=sym,
                    name=name,
                    market_cap=mkt_cap,
                    per=per,
                    ev_ebitda=ev_eb,
                    warnings=row_warnings,
                )
            )
    finally:
        if owned:
            fmp.close()

    # Apply sanity filter. The target (row 0) is never excluded regardless of
    # its own multiples — the report is about analysing it.
    kept: list[PeerMultipleRow] = []
    excluded: list[PeerMultipleRow] = []
    for i, row in enumerate(rows):
        if i == 0:
            kept.append(row)
            continue
        keep, reason = _is_sane_peer(row)
        if keep:
            kept.append(row)
        else:
            row.warnings.append(f"excluded from comparison: {reason}")
            excluded.append(row)

    return PeerMultiplesTable(
        target_symbol=symbol.upper(),
        as_of=target_as_of,
        rows=kept,
        excluded=excluded,
        override_sources=override_sources,
    )


# Re-export the debt/fcf derivators so verify.py can use them without a
# separate import path.
__all__ = [
    "CalculationResult",
    "PeerMultipleRow",
    "PeerMultiplesTable",
    "calculate_per",
    "calculate_ev_ebitda",
    "get_peer_multiples",
    "fn_total_debt",
    "derive_free_cash_flow",
]
