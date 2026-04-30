"""DART-based live fundamentals adapter for Korean tickers.

OpenDART is the authoritative source for KRX-listed entities and
covers what Finnhub's free tier doesn't reach for Korean symbols (.KS
/ .KQ). This adapter shapes DART's per-year `fnlttSinglAcntAll.json`
responses into the same `TickerFundamentals` contract the Stage 2
prefilter consumes — so callers don't need to know which data source
produced the numbers.

Symbol normalization handled here so the dispatcher can pass anything
the manifest carries:
  - "005930.KS"  → "005930" (Yahoo-style suffix)
  - "005930.KQ"  → "005930"
  - "KRX:005930" → "005930"
  - "005930"     → "005930"
  - "5930"       → "005930" (zero-pad)

Account extraction strategy (per row in the DART response):
  1. Try IFRS XBRL `account_id` (e.g. "ifrs-full_Revenue").
  2. Fall back to Korean `account_nm` (e.g. "수익(매출액)", "매출액").
  3. `sj_div` narrows the search to the right statement (BS / IS / CF).

KRW values are returned as-is. Currency normalization is the consumer's
responsibility — the prefilter compares ratios (ROIC, gross margin),
not absolute amounts, so KRW vs USD doesn't affect the verdict.

Limitations (called out per Commitment 3):
  - Quarterly margins were deferred until P1c (2026-04). DART's per-
    quarter `reprt_code` (11013, 11012, 11014) returns cumulative-to-
    date amounts; the standalone quarter values are recovered by
    subtraction (see `_build_quarterly_margins_from_dart`). Korean
    moat axis evaluation now matches the US adapter's quarterly_margins
    contract.
  - Industry classification is a static "Korean Equity (DART)" string;
    DART doesn't expose GICS sub-industries.
  - Segments default to single-segment; DART has segment disclosures
    but they're filed as separate `사업의 내용` documents, not via the
    fnlttSinglAcntAll endpoint.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Protocol

from wise_investor.screening.segments import single_segment_default
from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    TickerFundamentals,
)

logger = logging.getLogger(__name__)


DEFAULT_EFFECTIVE_TAX_RATE_KR: float = 0.22  # 한국 법인세 기본세율 ~22% (2024 기준)
DEFAULT_HISTORY_YEARS: int = 5


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------


class DartLikeClient(Protocol):
    """Minimal duck-typed interface this adapter expects from a DART client."""

    def corp_code_from_stock_code(self, stock_code: str) -> str | None: ...

    def financials(
        self,
        corp_code: str,
        year: int | str,
        reprt_code: str = ...,
        fs_div: str = ...,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fetch_live_fundamentals_kr(
    symbol: str,
    *,
    client: DartLikeClient | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    industry_aggregates: Any = None,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE_KR,
    today: dt.date | None = None,
) -> TickerFundamentals:
    """Pull last `history_years` annual filings from DART → TickerFundamentals.

    Args:
        symbol: Korean ticker in any common form (see module docstring).
        client: Anything satisfying DartLikeClient. None constructs a
            real DartClient (lazy import).
        history_years: How many fiscal years back to attempt. The
            adapter is forgiving — years with no data are silently
            dropped, so this is an upper bound, not a requirement.
        industry_aggregates: Same `IndustryAggregates` shape the US
            adapter accepts; pass None to leave the comparison fields
            unfilled.
        effective_tax_rate: Used to derive NOPAT from operating income.
            Korean default is 22% (post-2018 corporate rate baseline).
        today: Override for the "current" date used to determine the
            most-recent fiscal year to query. Tests inject a fixed date
            so they don't drift across runs.
    """
    if client is None:
        from wise_investor.data.dart import DartClient  # lazy: needs API key
        client = DartClient()

    stock_code = _normalize_kr_symbol(symbol)
    corp_code = client.corp_code_from_stock_code(stock_code)
    if corp_code is None:
        raise ValueError(
            f"DART corp_code not found for {symbol} "
            f"(normalized: {stock_code}). Possible causes: not KRX-listed, "
            f"recently delisted, or corpCode cache stale."
        )

    today = today or dt.date.today()
    # Korean filings disclose annual results around March of year+1, so
    # the latest reliably-available fiscal year is current_year - 1 if
    # we're past Q1, else current_year - 2. Be conservative: always
    # start at current_year - 1.
    most_recent_fy = today.year - 1
    earliest_fy = most_recent_fy - history_years + 1

    annual: list[AnnualFinancials] = []
    for fy in range(earliest_fy, most_recent_fy + 1):
        try:
            resp = client.financials(corp_code, year=fy)
        except Exception as e:
            logger.warning("DART financials %s/%s failed: %s", symbol, fy, e)
            continue
        if not getattr(resp, "ok", True):
            # status != "000" on the DART response
            continue
        ann = _build_annual_from_dart(resp, fy, effective_tax_rate)
        if ann is not None:
            annual.append(ann)

    annual.sort(key=lambda a: a.fiscal_year)
    annual_t = tuple(annual)

    # P1c (2026-04): quarterly margins from DART cumulative subtraction.
    # We need 12 quarters (3 years × 4) for the moat-axis GM std proxy.
    # Limit the lookup window to the most recent 3 fiscal years to keep
    # API call count bounded (3 × 3 = 9 extra calls per ticker; 11011
    # is already fetched above).
    quarterly_window = list(range(max(earliest_fy, most_recent_fy - 2), most_recent_fy + 1))
    quarterly_margins = _build_quarterly_margins_from_dart(
        client, corp_code, quarterly_window,
    )

    latest_fy = annual_t[-1].fiscal_year if annual_t else 0
    segments_history = (single_segment_default(symbol.upper(), fiscal_year=latest_fy),)

    if industry_aggregates is not None:
        roic_median = industry_aggregates.industry_roic_3y_median
        gm_std = industry_aggregates.industry_gross_margin_3y_std
    else:
        roic_median = None
        gm_std = None

    return TickerFundamentals(
        symbol=symbol.upper(),
        industry_classification="Korean Equity (DART)",
        annual=annual_t,
        quarterly_margins=quarterly_margins,
        segments_history=segments_history,
        top5_customer_share=None,
        diversification_attempt_signals=0,
        industry_roic_3y_median=roic_median,
        industry_gross_margin_3y_std=gm_std,
    )


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


def fetch_historical_fundamentals_kr(
    symbol: str,
    as_of_date: dt.date,
    *,
    client: DartLikeClient | None = None,
    history_years: int = DEFAULT_HISTORY_YEARS,
    industry_aggregates: Any = None,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE_KR,
) -> TickerFundamentals:
    """Historical Korean fundamentals at a point-in-time `as_of_date`.

    Wrapper around `fetch_live_fundamentals_kr` that synthesizes a
    "today" date so the live adapter's history-window logic resolves
    to filings publicly available on `as_of_date`.

    Korean 사업보고서 (annual) is filed within ~3 months of fiscal
    year-end. Quarterly reports take ~45 days. The conservative
    rule used here:

      most_recent_safe_fy = as_of_date.year - 1

    so a calibration_date of 2018-06-30 yields fy=2017 as the latest
    fully-disclosed annual. The adapter's quarterly window
    (most_recent_fy − 2 .. most_recent_fy) then covers 2015-2017.

    `industry_aggregates` is accepted for API symmetry with the US
    historical adapter but currently ignored — peer aggregation isn't
    wired for KR (no DART /peers endpoint and Finnhub returns 403 on
    KRX symbols).
    """
    most_recent_safe_fy = as_of_date.year - 1
    # Pick a synthetic mid-year date in the year *after* the safe fy
    # so the live adapter's `most_recent_fy = today.year - 1` resolves
    # to most_recent_safe_fy.
    fake_today = dt.date(most_recent_safe_fy + 1, 6, 1)
    return fetch_live_fundamentals_kr(
        symbol,
        client=client,
        history_years=history_years,
        industry_aggregates=industry_aggregates,
        effective_tax_rate=effective_tax_rate,
        today=fake_today,
    )


def _normalize_kr_symbol(symbol: str) -> str:
    """Normalize a Korean ticker to the 6-digit DART stock code.

    Accepts: "005930", "005930.KS", "005930.KQ", "KRX:005930",
    bare integers ("5930"). Returns 6-digit zero-padded.
    """
    s = str(symbol).strip().upper()
    for suffix in (".KS", ".KQ", ".KRX"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    for prefix in ("KRX:", "KS:", "KQ:"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.zfill(6)


# ---------------------------------------------------------------------------
# Per-filing projection
# ---------------------------------------------------------------------------


# Account lookup table — XBRL `account_id` candidates first, Korean
# display-name fallbacks second. Order within each list matters: most
# common first. The adapter walks the table per logical field.
#
# `sj_divs` is a tuple because Korean filers split between two valid
# DART labels for the income statement: standalone-IFRS filers use
# "IS"; consolidated filers (Samsung, NAVER, Hyundai) use "IS"; many
# others (SK Hynix, Kakao, Samsung Bio) use "CIS" (Comprehensive
# Income Statement). Both are schema-valid; the adapter tries each
# in order so we don't bias toward one filing convention.
_ACCOUNT_CANDIDATES: dict[str, dict[str, list[str] | tuple[str, ...]]] = {
    "revenue": {
        "ids": ["ifrs-full_Revenue", "dart_Revenue"],
        "names": ["수익(매출액)", "매출액", "수익", "영업수익"],
        "sj_divs": ("IS", "CIS"),
    },
    "gross_profit": {
        "ids": ["ifrs-full_GrossProfit"],
        "names": ["매출총이익", "매출총이익(손실)"],
        "sj_divs": ("IS", "CIS"),
    },
    "operating_income": {
        "ids": [
            "dart_OperatingIncomeLoss",
            "ifrs-full_ProfitLossFromOperatingActivities",
        ],
        "names": ["영업이익", "영업이익(손실)"],
        "sj_divs": ("IS", "CIS"),
    },
    "total_equity": {
        "ids": ["ifrs-full_Equity"],
        "names": ["자본총계"],
        "sj_divs": ("BS",),
    },
    "cash_and_equivalents": {
        "ids": ["ifrs-full_CashAndCashEquivalents"],
        "names": ["현금및현금성자산"],
        "sj_divs": ("BS",),
    },
    "short_term_borrowings": {
        "ids": ["ifrs-full_ShorttermBorrowings", "dart_ShorttermBorrowings"],
        "names": ["단기차입금"],
        "sj_divs": ("BS",),
    },
    "long_term_borrowings": {
        "ids": ["ifrs-full_LongtermBorrowings", "dart_LongtermBorrowings"],
        "names": ["장기차입금"],
        "sj_divs": ("BS",),
    },
}


def _build_quarterly_margins_from_dart(
    client: DartLikeClient,
    corp_code: str,
    years: list[int],
) -> tuple[QuarterlyMargin, ...]:
    """Recover per-quarter standalone gross margin from DART cumulatives.

    DART exposes four `reprt_code` values that all carry cumulative-to-
    date numbers:

      11013 (1분기) = Q1 standalone (already a single-quarter view)
      11012 (반기) = H1 cumulative = Q1 + Q2
      11014 (3분기) = 9M cumulative = Q1 + Q2 + Q3
      11011 (사업)  = FY cumulative = Q1 + Q2 + Q3 + Q4

    Standalone quarter values are reconstructed by subtraction:

      Q1 = 11013
      Q2 = 11012 − 11013
      Q3 = 11014 − 11012
      Q4 = 11011 − 11014

    Each (revenue, gross_profit) pair where both inputs are present and
    revenue > 0 contributes one `QuarterlyMargin`. Quarters with any
    missing input are silently dropped (Commitment 3: surface absence,
    don't fabricate). The result is capped at the last 12 quarters so
    callers can compute a 3-year std without further trimming.

    API cost (per ticker): up to 3 extra DART calls per fiscal year in
    `years` (11013/11012/11014; 11011 is already fetched by
    `fetch_live_fundamentals_kr`). With a 3-year window this adds 9
    calls per ticker — well under the free-tier 10k/day budget.
    """
    from wise_investor.data.dart import (
        REPORT_ANNUAL,
        REPORT_H1,
        REPORT_Q1,
        REPORT_Q3,
    )

    # Each entry stores the cumulative (revenue, gross_profit) for one
    # reprt_code. None when the fetch failed or the report wasn't filed.
    Pair = tuple[float | None, float | None]

    def _diff(later: Pair, earlier: Pair) -> Pair:
        """Subtract two cumulative (rev, gp) pairs. Either input None
        means the standalone quarter is unrecoverable."""
        if (
            later[0] is None
            or earlier[0] is None
            or later[1] is None
            or earlier[1] is None
        ):
            return (None, None)
        return (later[0] - earlier[0], later[1] - earlier[1])

    out: list[QuarterlyMargin] = []
    for year in sorted(years):
        cumulatives: dict[str, Pair] = {}
        for code, label in (
            (REPORT_Q1, "Q1"),
            (REPORT_H1, "H1"),
            (REPORT_Q3, "9M"),
            (REPORT_ANNUAL, "FY"),
        ):
            try:
                resp = client.financials(
                    corp_code, year=year, reprt_code=code,
                )
            except Exception as e:
                logger.debug(
                    "DART quarterly fetch failed for %s/%s/%s: %s",
                    corp_code, year, code, e,
                )
                cumulatives[label] = (None, None)
                continue
            if not getattr(resp, "ok", True):
                cumulatives[label] = (None, None)
                continue
            cumulatives[label] = (
                _extract(resp, "revenue"),
                _extract(resp, "gross_profit"),
            )

        # Rebuild standalone quarters in chronological order so the
        # final tuple is also chronological.
        q1: Pair = cumulatives.get("Q1", (None, None))
        q2: Pair = _diff(
            cumulatives.get("H1", (None, None)),
            cumulatives.get("Q1", (None, None)),
        )
        q3: Pair = _diff(
            cumulatives.get("9M", (None, None)),
            cumulatives.get("H1", (None, None)),
        )
        q4: Pair = _diff(
            cumulatives.get("FY", (None, None)),
            cumulatives.get("9M", (None, None)),
        )

        for q_id, (rev, gp) in (
            (f"{year}Q1", q1),
            (f"{year}Q2", q2),
            (f"{year}Q3", q3),
            (f"{year}Q4", q4),
        ):
            if rev is None or gp is None or rev <= 0:
                continue
            gm = gp / rev
            out.append(QuarterlyMargin(quarter_id=q_id, gross_margin=gm))

    return tuple(out[-12:])


def _build_annual_from_dart(
    resp: Any, fiscal_year: int, tax_rate: float
) -> AnnualFinancials | None:
    """Project one DART annual response into AnnualFinancials."""
    revenue = _extract(resp, "revenue")
    gross_profit = _extract(resp, "gross_profit")
    operating_income = _extract(resp, "operating_income")

    if (
        revenue is None
        and gross_profit is None
        and operating_income is None
    ):
        # Empty response — DART had no rows or no recognizable accounts.
        return None

    nopat: float | None = (
        operating_income * (1.0 - tax_rate)
        if operating_income is not None
        else None
    )

    equity = _extract(resp, "total_equity")
    cash = _extract(resp, "cash_and_equivalents")
    short_debt = _extract(resp, "short_term_borrowings")
    long_debt = _extract(resp, "long_term_borrowings")

    if short_debt is None and long_debt is None:
        total_debt: float | None = None
    else:
        total_debt = (short_debt or 0.0) + (long_debt or 0.0)

    if equity is not None:
        invested_capital: float | None = (
            (total_debt or 0.0) + equity - (cash or 0.0)
        )
    else:
        invested_capital = None

    return AnnualFinancials(
        fiscal_year=fiscal_year,
        revenue=revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        nopat=nopat,
        invested_capital=invested_capital,
        rd_expense=None,
    )


def _extract(resp: Any, logical_field: str) -> float | None:
    """Walk the candidate IDs / Korean names / sj_divs for one field.

    Iteration order: outer loop = sj_div (income filers split between
    "IS" and "CIS"); inner loop = id-first then name-fallback. First
    non-None hit wins.
    """
    from wise_investor.data.dart import extract_account_value

    spec = _ACCOUNT_CANDIDATES[logical_field]
    sj_divs: tuple[str, ...] = tuple(spec["sj_divs"])
    for sj_div in sj_divs:
        for aid in spec["ids"]:
            v = extract_account_value(resp, account_id=aid, sj_div=sj_div)
            if v is not None:
                return v
        for nm in spec["names"]:
            v = extract_account_value(resp, account_nm=nm, sj_div=sj_div)
            if v is not None:
                return v
    return None


# ---------------------------------------------------------------------------
# Symbol detection — used by the dispatcher in live_adapter.py
# ---------------------------------------------------------------------------


def is_korean_symbol(symbol: str) -> bool:
    """Heuristic: does this look like a Korean equity ticker?

    Patterns:
      - Suffix .KS, .KQ, .KRX
      - Prefix KRX:, KS:, KQ:
      - Bare 6-digit numeric (KRX listing convention)
      - Bare 1-5 digit numeric (zero-pads to 6)
    """
    s = str(symbol).strip().upper()
    if any(s.endswith(suf) for suf in (".KS", ".KQ", ".KRX")):
        return True
    if any(s.startswith(pfx) for pfx in ("KRX:", "KS:", "KQ:")):
        return True
    # Pure digits: only treat as KR when 1-6 digits long. Longer pure
    # numeric strings probably aren't tickers at all.
    return bool(s.isdigit() and 1 <= len(s) <= 6)


__all__ = [
    "DEFAULT_EFFECTIVE_TAX_RATE_KR",
    "DEFAULT_HISTORY_YEARS",
    "DartLikeClient",
    "fetch_historical_fundamentals_kr",
    "fetch_live_fundamentals_kr",
    "is_korean_symbol",
]
