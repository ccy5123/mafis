"""Live fundamentals adapter — dispatcher + Finnhub (US) implementation.

Two public entry points:

  - `fetch_live_fundamentals(symbol, ...)` — the dispatcher. Inspects
    the symbol pattern, routes Korean tickers (`.KS`, bare 6-digit, ...)
    to `live_adapter_kr.fetch_live_fundamentals_kr` (DART-backed) and
    everything else to `fetch_live_fundamentals_us` (Finnhub-backed).
    Most callers should use this one — it abstracts the data-source
    choice from the screening pipeline.

  - `fetch_live_fundamentals_us(symbol, ...)` — explicit Finnhub adapter.
    Use directly only when you need to bypass the dispatcher (e.g.
    ADR-listed Korean tickers that Finnhub does cover, where you'd
    rather use Finnhub's normalized data).

What `fetch_live_fundamentals_us` populates today:
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
    finnhub_client: FinancialsClient | None = None,
    dart_client: Any = None,
    industry_aggregates: IndustryAggregates | None = None,
    effective_tax_rate: float | None = None,
) -> TickerFundamentals:
    """Dispatcher: route by symbol pattern to the right per-source adapter.

    Korean tickers (.KS / .KQ / bare 6-digit / KRX:) → DART-backed adapter.
    Everything else → Finnhub-backed adapter.

    Args:
        symbol: Any of the formats either adapter accepts.
        finnhub_client: Used by the US adapter when applicable.
        dart_client: Used by the KR adapter when applicable.
        industry_aggregates: Forwarded as-is. Same shape works for both.
        effective_tax_rate: When None, each adapter uses its own default
            (US: 21%, KR: 22%). Override here only if you want a uniform
            rate across mixed-market batches.
    """
    from wise_investor.screening.live_adapter_kr import (
        DEFAULT_EFFECTIVE_TAX_RATE_KR,
        fetch_live_fundamentals_kr,
        is_korean_symbol,
    )

    if is_korean_symbol(symbol):
        rate = effective_tax_rate if effective_tax_rate is not None else DEFAULT_EFFECTIVE_TAX_RATE_KR
        return fetch_live_fundamentals_kr(
            symbol,
            client=dart_client,
            industry_aggregates=industry_aggregates,
            effective_tax_rate=rate,
        )

    rate = effective_tax_rate if effective_tax_rate is not None else DEFAULT_EFFECTIVE_TAX_RATE
    return fetch_live_fundamentals_us(
        symbol,
        client=finnhub_client,
        industry_aggregates=industry_aggregates,
        effective_tax_rate=rate,
    )


def fetch_live_fundamentals_us(
    symbol: str,
    *,
    client: FinancialsClient | None = None,
    industry_aggregates: IndustryAggregates | None = None,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE,
) -> TickerFundamentals:
    """Pull latest filings from Finnhub and shape them into `TickerFundamentals`.

    Args:
        symbol: Ticker (e.g. "NVDA"). For Korean tickers prefer the
            top-level `fetch_live_fundamentals` dispatcher, which routes
            to DART for richer coverage.
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
    finnhub_client: FinancialsClient | None = None,
    dart_client: Any = None,
    industry_aggregates_by_symbol: dict[str, IndustryAggregates] | None = None,
) -> list[TickerFundamentals]:
    """Apply the dispatcher to a list of mixed-market symbols.

    Per-ticker exceptions are caught and logged — partial runs are
    survivable. Result list may be shorter than input on errors.
    Clients are lazy-constructed: a US-only batch never imports the
    DART module, and vice versa. A real client is only built for
    markets that actually appear in `symbols`.
    """
    from wise_investor.screening.live_adapter_kr import is_korean_symbol

    out: list[TickerFundamentals] = []
    for s in symbols:
        try:
            aggs = (
                industry_aggregates_by_symbol.get(s.upper())
                if industry_aggregates_by_symbol is not None
                else None
            )
            if is_korean_symbol(s):
                if dart_client is None:
                    from wise_investor.data.dart import DartClient
                    dart_client = DartClient()
                funds = fetch_live_fundamentals(
                    s, dart_client=dart_client, industry_aggregates=aggs,
                )
            else:
                if finnhub_client is None:
                    from wise_investor.data.finnhub import FinnhubClient
                    finnhub_client = FinnhubClient()
                funds = fetch_live_fundamentals(
                    s, finnhub_client=finnhub_client, industry_aggregates=aggs,
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
    "fetch_live_fundamentals_us",
    "fetch_live_universe",
]
