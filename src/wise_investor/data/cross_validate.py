"""Lightweight cross-validation of Finnhub vs yfinance snapshots.

Phase 1B: compares current price and market cap from Finnhub (/quote and
/profile2) against yfinance's snapshot. A 5% threshold flags divergence.
PE is not in Finnhub /quote directly; use calculate_per for that separately.

EDGAR XBRL-based deep validation remains Phase 3 work.
"""

from __future__ import annotations

from pydantic import BaseModel

from wise_investor.data.finnhub import FinnhubClient
from wise_investor.data.yf import YFQuote, get_quote_snapshot


DEFAULT_THRESHOLD_PCT = 5.0


class FieldComparison(BaseModel):
    field: str
    fmp_value: float | None  # retained field name; now the Finnhub value
    yf_value: float | None
    diff_pct: float | None
    within_threshold: bool | None
    note: str | None = None


class CrossValidationResult(BaseModel):
    symbol: str
    threshold_pct: float
    comparisons: list[FieldComparison]

    @property
    def any_flagged(self) -> bool:
        return any(c.within_threshold is False for c in self.comparisons)


def compare_value(
    field: str,
    fmp_value: float | None,
    yf_value: float | None,
    threshold_pct: float,
) -> FieldComparison:
    if fmp_value is None and yf_value is None:
        return FieldComparison(
            field=field,
            fmp_value=None,
            yf_value=None,
            diff_pct=None,
            within_threshold=None,
            note="both sources missing",
        )
    if fmp_value is None or yf_value is None:
        return FieldComparison(
            field=field,
            fmp_value=fmp_value,
            yf_value=yf_value,
            diff_pct=None,
            within_threshold=None,
            note="one source missing — cannot compare",
        )
    if fmp_value == 0:
        same = yf_value == 0
        return FieldComparison(
            field=field,
            fmp_value=fmp_value,
            yf_value=yf_value,
            diff_pct=None,
            within_threshold=same,
            note="primary value is zero; compared by absolute equality",
        )

    diff_pct = abs(yf_value - fmp_value) / abs(fmp_value) * 100.0
    return FieldComparison(
        field=field,
        fmp_value=fmp_value,
        yf_value=yf_value,
        diff_pct=round(diff_pct, 3),
        within_threshold=diff_pct <= threshold_pct,
    )


def cross_validate_quote(
    symbol: str,
    fmp: FinnhubClient | None = None,
    yf_quote: YFQuote | None = None,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
) -> CrossValidationResult:
    """Compare Finnhub and yfinance snapshots across the Phase 1 fields.

    Parameter name `fmp` retained for call-site stability; it now accepts a
    FinnhubClient. The fmp_value field in each comparison similarly now
    refers to the Finnhub value.
    """
    owned_fmp = False
    if fmp is None:
        fmp = FinnhubClient()
        owned_fmp = True

    try:
        quote = fmp.quote(symbol)
        profile = fmp.profile(symbol)
    finally:
        if owned_fmp:
            fmp.close()

    if yf_quote is None:
        yf_quote = get_quote_snapshot(symbol)

    # Phase 1 cross-check covers price + market_cap. PE is not in Finnhub
    # /quote; use calculate_per for PE verification instead.
    comparisons = [
        compare_value("price", quote.price, yf_quote.price, threshold_pct),
        compare_value(
            "market_cap", profile.market_cap_usd, yf_quote.market_cap, threshold_pct
        ),
    ]
    return CrossValidationResult(
        symbol=symbol.upper(),
        threshold_pct=threshold_pct,
        comparisons=comparisons,
    )
