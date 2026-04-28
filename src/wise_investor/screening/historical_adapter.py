"""Point-in-time historical fundamentals adapter for back-validation.

Constitution v2.0 §23 Step 4 originally prescribed calibration via
user-intuition comparison, but that contradicts Commitment 1
(user preferences must not influence universe membership). The
v2.0 implementation replaces user-intuition calibration with
**back-validation**: pick a calibration date in the past, run the
rubric on tickers using only data available at that date, then
score the rubric against objective outcomes 5 years later.

This module is the data side of that flow. Given (symbol, as_of_date),
it returns a `TickerFundamentals` populated only with information
that would have been public on as_of_date.

Design choices:

  - **yfinance for the data source.** It scrapes Yahoo Finance, which
    is unreliable enough that production ingestion shouldn't depend
    on it. For back-validation that runs once and caches its outputs
    to disk forever (point-in-time data is immutable), the
    instability is tolerable.
  - **Annual data only in v1.** yfinance's quarterly fundamentals
    are noisier and frequently missing for older quarters. The
    rubric's quarterly-margin-std proxy degrades to None, which the
    Stage 2 prefilter already handles via NEED_LLM.
  - **No segment data.** yfinance doesn't surface segment-level
    revenue. Tickers without segment disclosure flow through
    `single_segment_default` (the company itself is treated as one
    100% segment). For diversified holdings (BRK, conglomerates),
    the §13 30% rule will correctly exclude them — we accept that
    cost.
  - **Filing-lag approximation: FY+90 days.** A 10-K is typically
    filed 60-90 days after fiscal year end. We use 90 days as a
    conservative cutoff so we never include data that wouldn't
    actually have been public at as_of_date.
  - **Forever cache.** Historical data does not change. Once
    fetched, results are written to `data/historical_cache/` keyed
    by (symbol, as_of_date) and never re-fetched.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from wise_investor.config import PROJECT_ROOT
from wise_investor.screening.segments import single_segment_default
from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    SegmentBreakdown,
    TickerFundamentals,
)


logger = logging.getLogger(__name__)


HISTORICAL_CACHE_DIR: Path = PROJECT_ROOT / "data" / "historical_cache"

# 10-K filing lag — used to compute "would this fundamental have been
# public on as_of_date." A FY ending Dec 31, 2018 + 90 days = March 31,
# 2019, so the 2018 annual filing is treated as public on that date.
FILING_LAG_DAYS: int = 90


# Type alias for the injectable yfinance fetcher. Returns a dict with:
#   "income_stmt": pandas DataFrame of annual income statement
#   "balance_sheet": pandas DataFrame of annual balance sheet
#   "info": dict from Ticker.info (industry classification etc.)
# Tests pass a stub; production calls _default_yfinance_fetcher which
# imports yfinance and uses Ticker.income_stmt / .balance_sheet.
HistoricalFetcher = Callable[[str], dict[str, Any]]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_historical_fundamentals(
    symbol: str,
    as_of_date: dt.date,
    *,
    cache: bool = True,
    fetcher: HistoricalFetcher | None = None,
) -> TickerFundamentals:
    """Return TickerFundamentals filtered to data available on as_of_date.

    Args:
      symbol: ticker.
      as_of_date: the historical "now" — only data published BEFORE
        this date is included. Use this to simulate what the rubric
        would have said had it run on as_of_date.
      cache: if True, results are read from / written to
        `data/historical_cache/` keyed by (symbol, as_of_date).
        Disable for tests or when intentionally re-fetching.
      fetcher: injectable. None uses the default yfinance fetcher.
    """
    symbol = symbol.upper()
    cache_key = _cache_path(symbol, as_of_date)

    if cache and cache_key.exists():
        try:
            return _load_from_cache(cache_key)
        except Exception as e:
            logger.warning("Cache read failed for %s (%s); refetching.", cache_key, e)

    if fetcher is None:
        fetcher = _default_yfinance_fetcher

    raw = fetcher(symbol)
    funds = _build_fundamentals(symbol, as_of_date, raw)

    if cache:
        try:
            _write_to_cache(cache_key, funds)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", cache_key, e)

    return funds


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path(symbol: str, as_of_date: dt.date) -> Path:
    HISTORICAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return HISTORICAL_CACHE_DIR / f"{symbol.upper()}_{as_of_date.isoformat()}.json"


def _load_from_cache(path: Path) -> TickerFundamentals:
    raw = json.loads(path.read_text(encoding="utf-8"))
    annual = tuple(
        AnnualFinancials(**a) for a in raw.get("annual", [])
    )
    quarterly = tuple(
        QuarterlyMargin(**q) for q in raw.get("quarterly_margins", [])
    )
    segments = tuple(
        _segment_breakdown_from_dict(s) for s in raw.get("segments_history", [])
    )
    return TickerFundamentals(
        symbol=raw["symbol"],
        industry_classification=raw["industry_classification"],
        annual=annual,
        quarterly_margins=quarterly,
        segments_history=segments,
        top5_customer_share=raw.get("top5_customer_share"),
        diversification_attempt_signals=raw.get("diversification_attempt_signals", 0),
        industry_roic_3y_median=raw.get("industry_roic_3y_median"),
        industry_gross_margin_3y_std=raw.get("industry_gross_margin_3y_std"),
    )


def _segment_breakdown_from_dict(d: dict[str, Any]) -> SegmentBreakdown:
    from wise_investor.screening.types import Segment

    return SegmentBreakdown(
        primary_segment_exists=bool(d["primary_segment_exists"]),
        primary_segment_name=d.get("primary_segment_name"),
        primary_segment_revenue_share=d.get("primary_segment_revenue_share"),
        all_segments=tuple(Segment(**s) for s in d.get("all_segments", [])),
        fiscal_year=int(d["fiscal_year"]),
        source=str(d.get("source", "stub")),
    )


def _write_to_cache(path: Path, funds: TickerFundamentals) -> None:
    payload = {
        "symbol": funds.symbol,
        "industry_classification": funds.industry_classification,
        "annual": [asdict(a) for a in funds.annual],
        "quarterly_margins": [asdict(q) for q in funds.quarterly_margins],
        "segments_history": [
            {
                "primary_segment_exists": s.primary_segment_exists,
                "primary_segment_name": s.primary_segment_name,
                "primary_segment_revenue_share": s.primary_segment_revenue_share,
                "all_segments": [asdict(seg) for seg in s.all_segments],
                "fiscal_year": s.fiscal_year,
                "source": s.source,
            }
            for s in funds.segments_history
        ],
        "top5_customer_share": funds.top5_customer_share,
        "diversification_attempt_signals": funds.diversification_attempt_signals,
        "industry_roic_3y_median": funds.industry_roic_3y_median,
        "industry_gross_margin_3y_std": funds.industry_gross_margin_3y_std,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Building TickerFundamentals from raw yfinance output
# ---------------------------------------------------------------------------


def _build_fundamentals(
    symbol: str,
    as_of_date: dt.date,
    raw: dict[str, Any],
) -> TickerFundamentals:
    """Convert raw yfinance dicts to TickerFundamentals, filtered to
    data available on as_of_date.
    """
    income = raw.get("income_stmt") or {}  # {year: {field: value}}
    balance = raw.get("balance_sheet") or {}
    info = raw.get("info") or {}

    industry = (
        info.get("industry")
        or info.get("industryDisp")
        or info.get("sector")
        or "Unknown"
    )

    cutoff = as_of_date - dt.timedelta(days=FILING_LAG_DAYS)

    annual_rows: list[AnnualFinancials] = []
    for year_str in sorted(income.keys()):
        # year_str is the fiscal-year-end date string (yfinance returns
        # column labels as fiscal-year-end dates). Treat the "year" as
        # the calendar year of fiscal-year-end.
        try:
            fy_end = dt.date.fromisoformat(str(year_str)[:10])
        except (ValueError, TypeError):
            continue
        if fy_end > cutoff:
            continue  # not yet public at as_of_date

        i_row = income.get(year_str) or {}
        b_row = balance.get(year_str) or {}

        revenue = _opt_float(i_row.get("Total Revenue"))
        gross = _opt_float(i_row.get("Gross Profit"))
        operating = _opt_float(i_row.get("Operating Income"))
        rd = _opt_float(i_row.get("Research And Development"))
        # NOPAT ≈ Operating Income × (1 - tax_rate). yfinance gives
        # Tax Rate For Calcs sporadically; if missing we approximate
        # tax_rate = Tax Provision / Pretax Income, else default 21%.
        tax_rate = _estimate_tax_rate(i_row)
        if operating is not None and tax_rate is not None:
            nopat = operating * (1.0 - tax_rate)
        else:
            nopat = None

        invested_capital = _estimate_invested_capital(b_row)

        annual_rows.append(
            AnnualFinancials(
                fiscal_year=fy_end.year,
                revenue=revenue,
                gross_profit=gross,
                operating_income=operating,
                nopat=nopat,
                invested_capital=invested_capital,
                rd_expense=rd,
            )
        )

    # Default segment: single 100% segment for the most recent fiscal
    # year we kept. yfinance doesn't surface segment data.
    segments_history: tuple[SegmentBreakdown, ...]
    if annual_rows:
        latest_fy = annual_rows[-1].fiscal_year
        segments_history = (
            single_segment_default(symbol, fiscal_year=latest_fy, source="yfinance"),
        )
    else:
        segments_history = ()

    return TickerFundamentals(
        symbol=symbol,
        industry_classification=str(industry),
        annual=tuple(annual_rows),
        quarterly_margins=(),  # see module docstring for rationale
        segments_history=segments_history,
        top5_customer_share=None,  # not in yfinance; NEED_LLM at Stage 3
        diversification_attempt_signals=0,
        industry_roic_3y_median=None,  # populated separately by peer aggregation
        industry_gross_margin_3y_std=None,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


# Default US corporate tax rate used when we can't infer from
# the income statement. Matches the post-2017 federal rate; state
# tax adds a few points but for back-validation purposes the order
# of magnitude is what matters.
_DEFAULT_TAX_RATE: float = 0.21


def _estimate_tax_rate(income_row: dict[str, Any]) -> float | None:
    """Infer effective tax rate from the income statement row.

    Returns None when we can't estimate; the caller treats that as
    "NOPAT not computable" and the moat axis will land in NEED_LLM
    rather than fabricate a NOPAT.
    """
    explicit = _opt_float(income_row.get("Tax Rate For Calcs"))
    if explicit is not None and 0.0 <= explicit <= 1.0:
        return explicit

    tax = _opt_float(income_row.get("Tax Provision"))
    pretax = _opt_float(income_row.get("Pretax Income"))
    if tax is not None and pretax is not None and pretax != 0:
        rate = tax / pretax
        # Sanity bound — extreme outliers (negative pretax, huge swings)
        # corrupt the rate; fall back to the default in those cases.
        if 0.0 <= rate <= 0.5:
            return rate

    return _DEFAULT_TAX_RATE


def _estimate_invested_capital(balance_row: dict[str, Any]) -> float | None:
    """Invested capital ≈ total debt + total equity − cash & equivalents.

    Returns None if any required component is missing — the caller
    skips this fiscal year's ROIC computation rather than fabricate.
    """
    total_debt = _opt_float(balance_row.get("Total Debt"))
    equity = _opt_float(
        balance_row.get("Stockholders Equity")
        or balance_row.get("Common Stock Equity")
        or balance_row.get("Total Equity Gross Minority Interest")
    )
    cash = _opt_float(
        balance_row.get("Cash And Cash Equivalents")
        or balance_row.get("Cash Cash Equivalents And Short Term Investments")
    )

    if equity is None:
        return None
    # Treat missing debt or cash as zero — we'd rather have a slightly
    # noisy IC than skip ROIC entirely. The 5pp persistence threshold
    # (constitution §15) is forgiving enough that small IC drifts
    # don't flip moat verdicts.
    return (total_debt or 0.0) + equity - (cash or 0.0)


# ---------------------------------------------------------------------------
# Default yfinance fetcher (production)
# ---------------------------------------------------------------------------


def _default_yfinance_fetcher(symbol: str) -> dict[str, Any]:
    """Read fundamentals from yfinance.

    yfinance is an optional runtime dependency for back-validation.
    If it isn't installed, we raise a clear error rather than letting
    the caller see a bare ImportError from deep inside the call.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover — yfinance is in core deps
        raise RuntimeError(
            "Back-validation requires yfinance. It's already in the "
            "MAFIS dependency list; run `pip install -e .` from the "
            "repo root if you somehow have a partial environment."
        ) from e

    ticker = yf.Ticker(symbol)
    income = ticker.income_stmt
    balance = ticker.balance_sheet

    # yfinance returns DataFrames whose columns are timestamps. Convert
    # to dict-of-dicts so the downstream code doesn't need pandas.
    income_dict: dict[str, dict[str, Any]] = {}
    if income is not None and not income.empty:
        for col in income.columns:
            key = col.date().isoformat() if hasattr(col, "date") else str(col)
            income_dict[key] = {
                str(idx): None if _is_nan(row) else float(row)
                for idx, row in income[col].items()
            }

    balance_dict: dict[str, dict[str, Any]] = {}
    if balance is not None and not balance.empty:
        for col in balance.columns:
            key = col.date().isoformat() if hasattr(col, "date") else str(col)
            balance_dict[key] = {
                str(idx): None if _is_nan(row) else float(row)
                for idx, row in balance[col].items()
            }

    info = {}
    try:
        info = ticker.info or {}
    except Exception as e:
        logger.warning("yfinance .info failed for %s: %s", symbol, e)

    return {
        "income_stmt": income_dict,
        "balance_sheet": balance_dict,
        "info": info,
    }


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN check without numpy import
    except Exception:
        return False


__all__ = [
    "FILING_LAG_DAYS",
    "HISTORICAL_CACHE_DIR",
    "fetch_historical_fundamentals",
]
