"""Live fundamentals adapter — populates `TickerFundamentals` from Finnhub.

Unlike `historical_adapter.py` (which filters by 90-day filing-lag
against a historical as-of date for back-validation), this module pulls
the latest available filings and uses them as point-in-time-now inputs
for the Stage 2 prefilter — the live screening path.

What this adapter populates today:
  - `annual` fiscal-year history from Finnhub `/stock/financials-reported`
    (revenue, gross profit, operating income, NOPAT, invested capital)
  - `quarterly_margins` from quarterly filings (gross margin only)
  - `industry_classification` from `/stock/profile2`
  - `segments_history` falls back to single-segment-default; Finnhub's
    free tier doesn't expose business-segment disclosures
  - `industry_roic_3y_median` / `industry_gross_margin_3y_std` are None
    unless an `IndustryAggregates` is supplied — peer aggregation is a
    separate (cost-bearing) concern, not auto-fired here

What stays None / 0 by design (Commitment 3 — precision over recall):
  - `top5_customer_share` — requires 10-K Risk Factors RAG; not yet wired
  - `diversification_attempt_signals` — same source

The Stage 2 prefilter handles all these Nones gracefully: missing data
yields NEED_LLM (or the corresponding axis-FAIL when the constitution
demands it), not PASS. So the prefilter never quietly upgrades a
data-missing ticker into a buy candidate — the LLM tier or the human
reviewer makes the call instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from wise_investor.screening.segments import single_segment_default
from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    TickerFundamentals,
)

logger = logging.getLogger(__name__)


# Default effective tax rate when the filing doesn't include income-tax
# expense in a parseable form. 21% mirrors the post-TCJA US federal
# corporate rate. Calibration runs can refine per-industry if needed.
DEFAULT_EFFECTIVE_TAX_RATE: float = 0.21


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------


class FinancialsClient(Protocol):
    """Minimal duck-typed interface this adapter expects from the upstream
    fundamentals client. The Finnhub `FinnhubClient` satisfies it; tests
    pass a stub with the same shape."""

    def financials(self, symbol: str, freq: str = "annual") -> Any: ...

    def profile(self, symbol: str) -> Any: ...


@dataclass(frozen=True)
class IndustryAggregates:
    """Pre-computed peer aggregates (optional input)."""

    industry_roic_3y_median: float | None
    industry_gross_margin_3y_std: float | None


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def fetch_live_fundamentals(
    symbol: str,
    *,
    client: FinancialsClient | None = None,
    industry_aggregates: IndustryAggregates | None = None,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE,
) -> TickerFundamentals:
    """Pull latest filings and shape them into `TickerFundamentals`.

    Args:
        symbol: Ticker (e.g. "NVDA"). Korean .KS tickers pass through but
            Finnhub coverage is partial; DART integration is Step 5d.
        client: Anything satisfying `FinancialsClient`. When None, a
            default `FinnhubClient` is constructed (lazy import, so this
            module stays importable without the Finnhub API key set).
        industry_aggregates: Pre-computed peer median/std. None leaves
            the comparison fields None — the prefilter handles that.
        effective_tax_rate: Used to derive NOPAT from operating income.
    """
    if client is None:
        # Lazy: importing FinnhubClient triggers settings access (API key
        # validation). Tests that pass a stub never hit this branch.
        from wise_investor.data.finnhub import FinnhubClient
        client = FinnhubClient()

    sym = symbol.upper()

    # --- annual ---
    annual_resp = client.financials(sym, freq="annual")
    annual: list[AnnualFinancials] = []
    for entry in getattr(annual_resp, "data", []) or []:
        ann = _build_annual(entry, effective_tax_rate)
        if ann is not None:
            annual.append(ann)
    annual.sort(key=lambda a: a.fiscal_year)  # newest LAST, per type contract
    annual_t = tuple(annual)

    # --- quarterly margins (last 12) ---
    quarterly: list[QuarterlyMargin] = []
    try:
        q_resp = client.financials(sym, freq="quarterly")
    except Exception as e:
        logger.warning("quarterly financials fetch failed for %s: %s", sym, e)
        q_resp = None

    if q_resp is not None:
        for entry in (getattr(q_resp, "data", []) or [])[:12]:
            qm = _build_quarterly_margin(entry)
            if qm is not None:
                quarterly.append(qm)
    quarterly_t = tuple(quarterly)

    # --- profile + industry classification ---
    industry_classification = "Unknown"
    try:
        profile = client.profile(sym)
        industry_classification = (
            getattr(profile, "finnhub_industry", None) or "Unknown"
        )
    except Exception as e:
        logger.warning("profile fetch failed for %s: %s", sym, e)

    # --- segment default ---
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
        top5_customer_share=None,            # Commitment 3 — see module docstring
        diversification_attempt_signals=0,    # Commitment 3 — see module docstring
        industry_roic_3y_median=roic_median,
        industry_gross_margin_3y_std=gm_std,
    )


def fetch_live_universe(
    symbols: list[str],
    *,
    client: FinancialsClient | None = None,
    industry_aggregates_by_symbol: dict[str, IndustryAggregates] | None = None,
) -> list[TickerFundamentals]:
    """Apply `fetch_live_fundamentals` to a list of symbols.

    Per-ticker exceptions are caught and logged — partial runs are
    survivable. Result list may be shorter than input on errors.
    """
    if client is None:
        from wise_investor.data.finnhub import FinnhubClient
        client = FinnhubClient()

    out: list[TickerFundamentals] = []
    for s in symbols:
        try:
            aggs = (
                industry_aggregates_by_symbol.get(s.upper())
                if industry_aggregates_by_symbol is not None
                else None
            )
            funds = fetch_live_fundamentals(
                s, client=client, industry_aggregates=aggs
            )
            out.append(funds)
        except Exception as e:
            logger.warning("live fundamentals fetch failed for %s: %s", s, e)
    return out


# ---------------------------------------------------------------------------
# Per-entry projection helpers
# ---------------------------------------------------------------------------


def _build_annual(entry: Any, tax_rate: float) -> AnnualFinancials | None:
    """Project one Finnhub annual filing into `AnnualFinancials`.

    Returns None when the entry doesn't carry a fiscal year or has no
    extractable income-statement values — those rows would only pollute
    downstream proxy computations.
    """
    # Importing here keeps the module importable even when finnhub.py's
    # settings dependency is uninitialized (e.g. tests that don't touch
    # the real client). `extract_field` and `total_debt` are pure helpers
    # over the FinancialsEntry shape.
    from wise_investor.data.finnhub import extract_field, total_debt

    if getattr(entry, "year", None) is None:
        return None

    revenue = extract_field(entry, "revenue")
    gross_profit = extract_field(entry, "gross_profit")
    operating_income = extract_field(entry, "operating_income")

    # NOPAT — the simplest derivation that keeps live screening operating
    # consistently with historical_adapter.py. Effective-tax derivation
    # from the filing itself is a refinement for Step 5d.
    if operating_income is not None:
        nopat: float | None = operating_income * (1.0 - tax_rate)
    else:
        nopat = None

    debt = total_debt(entry)
    equity = extract_field(entry, "total_stockholders_equity")
    cash = extract_field(entry, "cash_and_cash_equivalents")
    if equity is not None:
        invested_capital: float | None = (debt or 0.0) + equity - (cash or 0.0)
    else:
        invested_capital = None

    return AnnualFinancials(
        fiscal_year=entry.year,
        revenue=revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        nopat=nopat,
        invested_capital=invested_capital,
        rd_expense=None,  # not in CONCEPT_CANDIDATES yet
    )


def _build_quarterly_margin(entry: Any) -> QuarterlyMargin | None:
    """Project one quarterly filing into `QuarterlyMargin`.

    Returns None when revenue/gross profit are missing or revenue is
    zero — division by zero would otherwise propagate as inf into the
    Stage 2 std-dev computation.
    """
    from wise_investor.data.finnhub import extract_field

    revenue = extract_field(entry, "revenue")
    gross_profit = extract_field(entry, "gross_profit")
    if revenue is None or revenue == 0:
        return None
    if gross_profit is None:
        return None

    year = getattr(entry, "year", None)
    quarter = getattr(entry, "quarter", None)
    if year is None or quarter is None:
        return None

    return QuarterlyMargin(
        quarter_id=f"{year}Q{quarter}",
        gross_margin=gross_profit / revenue,
    )


__all__ = [
    "DEFAULT_EFFECTIVE_TAX_RATE",
    "FinancialsClient",
    "IndustryAggregates",
    "fetch_live_fundamentals",
    "fetch_live_universe",
]
