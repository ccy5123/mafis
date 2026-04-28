"""Finnhub-backed historical fundamentals adapter for back-validation.

The yfinance-based `historical_adapter.py` was built first because
yfinance is dependency-free. In practice, yfinance's free tier only
returns ~4 years of annual fundamentals — too short for the
constitution's 5-year back-validation horizon (§22).

Finnhub's `/stock/financials-reported` endpoint returns 15+ years of
annual filings WITH filed_date metadata, which is exactly what
back-validation needs to apply the filing-lag filter correctly: a
filing is only "publicly known" on its filed_date, not its
fiscal-year-end date.

This module is the Finnhub-backed counterpart to
`historical_adapter.py`. The live adapter for current-state screening
already uses Finnhub via `live_adapter.py`; this adapter adds the
as-of-date filtering that back-validation needs but live screening
doesn't.

Filing-lag rule: include a filing if its `filed_date` (Finnhub's
actual filing date, not fiscal year end) is on or before the
calibration date. Finnhub provides exact filed_date, so we don't need
the 90-day approximation that yfinance forced.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from wise_investor.screening.live_adapter import (
    DEFAULT_EFFECTIVE_TAX_RATE,
    FinancialsClient,
    IndustryAggregates,
    _build_annual,
    _build_quarterly_margin,
)
from wise_investor.screening.segments import single_segment_default
from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    TickerFundamentals,
)

logger = logging.getLogger(__name__)


def fetch_historical_fundamentals_finnhub(
    symbol: str,
    as_of_date: dt.date,
    *,
    client: FinancialsClient | None = None,
    industry_aggregates: IndustryAggregates | None = None,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE,
) -> TickerFundamentals:
    """Pull historical filings from Finnhub, filtered to those public
    on `as_of_date`.

    Filtering uses Finnhub's exact `filed_date`. A filing is included
    iff `filed_date <= as_of_date`. Filings still in the lag window
    (filed but the report's contents weren't yet broadly known) are
    NOT excluded — Finnhub's filed_date matches SEC EDGAR's, which
    is the moment the document becomes publicly available.

    Args:
        symbol: Ticker. Korean .KS tickers will fall through but
            Finnhub coverage is partial; use DART-based historical
            adapter (Step 5d+) for those.
        as_of_date: The point-in-time date for back-validation. Only
            filings publicly known on this date are returned.
        client: Anything satisfying FinancialsClient. None constructs
            a default FinnhubClient (lazy import).
        industry_aggregates, effective_tax_rate: Same semantics as the
            live adapter.
    """
    if client is None:
        from wise_investor.data.finnhub import FinnhubClient
        client = FinnhubClient()

    sym = symbol.upper()

    # --- annual: pull all, filter by filed_date ---
    annual_resp = client.financials(sym, freq="annual")
    raw_entries = list(getattr(annual_resp, "data", []) or [])
    public_entries = [e for e in raw_entries if _is_public_by(e, as_of_date)]

    annual: list[AnnualFinancials] = []
    for entry in public_entries:
        ann = _build_annual(entry, effective_tax_rate)
        if ann is not None:
            annual.append(ann)
    annual.sort(key=lambda a: a.fiscal_year)
    annual_t = tuple(annual)

    # --- quarterly margins (last 12 public-by-date) ---
    quarterly: list[QuarterlyMargin] = []
    try:
        q_resp = client.financials(sym, freq="quarterly")
    except Exception as e:
        logger.warning("quarterly financials fetch failed for %s: %s", sym, e)
        q_resp = None

    if q_resp is not None:
        q_raw = list(getattr(q_resp, "data", []) or [])
        q_public = [e for e in q_raw if _is_public_by(e, as_of_date)]
        # Most recent first — take up to 12.
        for entry in q_public[:12]:
            qm = _build_quarterly_margin(entry)
            if qm is not None:
                quarterly.append(qm)
    quarterly_t = tuple(quarterly)

    # --- profile + industry classification (current snapshot;
    #     Finnhub doesn't expose historical industry reclassifications) ---
    industry_classification = "Unknown"
    try:
        profile = client.profile(sym)
        industry_classification = (
            getattr(profile, "finnhub_industry", None) or "Unknown"
        )
    except Exception as e:
        logger.warning("profile fetch failed for %s: %s", sym, e)

    # --- segment default (Finnhub free tier doesn't expose segments) ---
    latest_fy = annual_t[-1].fiscal_year if annual_t else 0
    segments_history = (single_segment_default(sym, fiscal_year=latest_fy),)

    # --- industry aggregates ---
    if industry_aggregates is not None:
        roic_median = industry_aggregates.industry_roic_3y_median
        gm_std = industry_aggregates.industry_gross_margin_3y_std
    else:
        roic_median = None
        gm_std = None

    return TickerFundamentals(
        symbol=sym,
        industry_classification=industry_classification,
        annual=annual_t,
        quarterly_margins=quarterly_t,
        segments_history=segments_history,
        top5_customer_share=None,
        diversification_attempt_signals=0,
        industry_roic_3y_median=roic_median,
        industry_gross_margin_3y_std=gm_std,
    )


# ---------------------------------------------------------------------------
# Filing-lag filter
# ---------------------------------------------------------------------------


def _is_public_by(entry: Any, as_of_date: dt.date) -> bool:
    """True iff Finnhub's `filed_date` for this entry is on/before as_of_date.

    When filed_date is missing we fall back to the more conservative
    `end_date + 90 days` (matching the yfinance adapter's heuristic).
    Pure Pythonic-style: missing year/dates → exclude rather than
    include, so back-validation doesn't accidentally use a filing
    that wasn't actually public yet.
    """
    filed_raw = getattr(entry, "filed_date", None)
    filed: dt.date | None = _parse_finnhub_date(filed_raw)
    if filed is not None:
        return filed <= as_of_date

    # Fallback: end_date + 90 days lag
    end_raw = getattr(entry, "end_date", None)
    end: dt.date | None = _parse_finnhub_date(end_raw)
    if end is None:
        return False  # no date info → can't prove public; exclude
    return (end + dt.timedelta(days=90)) <= as_of_date


def _parse_finnhub_date(value: Any) -> dt.date | None:
    """Finnhub returns dates as 'YYYY-MM-DD HH:MM:SS' strings via
    pydantic, or sometimes pre-parsed datetime objects. Tolerate both."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    s = str(value).strip()
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s[:10])
    except ValueError:
        return None


__all__ = ["fetch_historical_fundamentals_finnhub"]
