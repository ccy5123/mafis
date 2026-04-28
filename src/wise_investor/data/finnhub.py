"""Finnhub /api/v1 client — primary fundamentals source after Phase 1B migration.

Replaces FMP following the 2025-08-31 FMP endpoint restructuring and the daily
quota constraint of the free tier. Finnhub's free tier offers 60 calls/minute
with no daily cap.

Key differences from FMP:
- Dollar values in /profile2 and /metric are denominated in USD **millions**.
- Financial statements come via /stock/financials-reported as raw XBRL-tagged
  line items grouped by report.ic (income), report.bs (balance sheet), and
  report.cf (cash flow). We extract fields by matching XBRL `concept` names.
- Pre-computed ratios (PE, EV/EBITDA) live under /stock/metric?metric=all.
- Peers endpoint returns a flat list of symbol strings, not detail records.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from wise_investor.config import settings

logger = logging.getLogger(__name__)


BASE_URL = "https://finnhub.io/api/v1"

# Finnhub returns dollar values in /profile2 and /metric as millions of USD.
DOLLAR_MILLIONS = 1_000_000.0


# Ticker-to-financials-reporting alias map. Surfaces the cases where a
# ticker we hold in the manifest is NOT the ticker Finnhub indexes the
# SEC filings under. Calibration finding (#4, 2026-04):
#
#   GOOG → GOOGL: post-2015 Alphabet reorg, the 10-K is filed under
#                 GOOGL (Class A); GOOG (Class C non-voting) carries
#                 only pre-reorg filings (2011-2015). Querying GOOG's
#                 financials-reported gives 4 stale entries; GOOGL
#                 gives 11 current ones.
#
# This map only affects the financials-reported endpoint. Quote,
# profile, peers, and metric still resolve under the original ticker
# (those are price/identity metadata, not filing data).
_FINANCIALS_TICKER_ALIAS: dict[str, str] = {
    "GOOG": "GOOGL",
}


def _financials_symbol(symbol: str) -> str:
    """Return the ticker Finnhub indexes 10-K filings under.

    Most tickers map to themselves. The alias map handles the cases
    where a ticker holds the price symbol but the financials live
    under a sibling (most often a dual-class-share reorg).
    """
    return _FINANCIALS_TICKER_ALIAS.get(symbol.upper(), symbol.upper())


class FinnhubError(RuntimeError):
    """Raised when Finnhub returns an error payload or non-retryable HTTP failure."""


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class Quote(_Model):
    """`/quote?symbol=...` response. `c` is current price."""

    c: float  # current price
    d: float | None = None  # change
    dp: float | None = None  # change percent
    h: float | None = None  # high
    l: float | None = None  # low  # noqa: E741 — mirrors Finnhub /quote API field name
    o: float | None = None  # open
    pc: float | None = None  # previous close
    t: int | None = None  # timestamp

    @property
    def price(self) -> float:
        return self.c


class Profile(_Model):
    """`/stock/profile2` — marketCapitalization and shareOutstanding are in MILLIONS."""

    country: str | None = None
    currency: str | None = None
    estimate_currency: str | None = None
    exchange: str | None = None
    finnhub_industry: str | None = None
    ipo: str | None = None
    logo: str | None = None
    market_capitalization: float | None = None  # USD millions
    name: str | None = None
    phone: str | None = None
    share_outstanding: float | None = None  # millions of shares
    ticker: str | None = None
    weburl: str | None = None

    @property
    def market_cap_usd(self) -> float | None:
        """Market cap in dollars (not millions)."""
        if self.market_capitalization is None:
            return None
        return self.market_capitalization * DOLLAR_MILLIONS

    @property
    def shares_outstanding_abs(self) -> float | None:
        if self.share_outstanding is None:
            return None
        return self.share_outstanding * DOLLAR_MILLIONS  # shares are also in millions


class Metric(_Model):
    """`/stock/metric?metric=all` — the big flat dict of pre-computed metrics.

    Only fields we actually consume are enumerated; the rest land in extras
    (ignored by pydantic). Dollar values marked with `_usd_millions` are
    stored as returned (millions). Use the helper properties for dollars.
    """

    # Valuation ratios (pre-computed by Finnhub)
    pe_annual: float | None = Field(default=None, alias="peAnnual")
    pe_ttm: float | None = Field(default=None, alias="peTTM")
    pe_excl_extra_annual: float | None = Field(default=None, alias="peExclExtraAnnual")
    pe_excl_extra_ttm: float | None = Field(default=None, alias="peExclExtraTTM")
    pe_normalized_annual: float | None = Field(default=None, alias="peNormalizedAnnual")
    forward_pe: float | None = Field(default=None, alias="forwardPE")
    ev_ebitda_ttm: float | None = Field(default=None, alias="evEbitdaTTM")
    ev_revenue_ttm: float | None = Field(default=None, alias="evRevenueTTM")
    peg_ttm: float | None = Field(default=None, alias="pegTTM")

    # Absolute values (millions)
    enterprise_value: float | None = None  # USD millions

    # Growth (percent)
    revenue_growth_ttm_yoy: float | None = Field(default=None, alias="revenueGrowthTTMYoy")
    revenue_growth_3y: float | None = Field(default=None, alias="revenueGrowth3Y")
    revenue_growth_5y: float | None = Field(default=None, alias="revenueGrowth5Y")
    ebitda_cagr_5y: float | None = Field(default=None, alias="ebitdaCagr5Y")

    # Margins (percent)
    operating_margin_ttm: float | None = Field(default=None, alias="operatingMarginTTM")
    operating_margin_annual: float | None = Field(default=None, alias="operatingMarginAnnual")

    # Leverage
    total_debt_to_total_equity_annual: float | None = Field(
        default=None, alias="totalDebt/totalEquityAnnual"
    )
    long_term_debt_to_equity_annual: float | None = Field(
        default=None, alias="longTermDebt/equityAnnual"
    )

    @property
    def enterprise_value_usd(self) -> float | None:
        if self.enterprise_value is None:
            return None
        return self.enterprise_value * DOLLAR_MILLIONS


class MetricResponse(_Model):
    """Top-level `/stock/metric` response: wraps the nested metric dict."""

    metric: Metric
    metric_type: str | None = None
    symbol: str
    series: dict[str, Any] | None = None


class FinancialLineItem(_Model):
    """One XBRL concept entry inside report.ic/bs/cf."""

    concept: str | None = None
    label: str | None = None
    unit: str | None = None
    value: float | None = None

    # Finnhub occasionally returns 'N/A' (or '-', '') in `value` for
    # filings where a particular XBRL concept wasn't reported. The plain
    # `float | None` annotation rejects these as a parse error and the
    # whole FinancialsResponse fails validation — which used to take
    # down our quarterly fetch entirely. Coerce common sentinels to None
    # so a single bad concept doesn't poison the whole response.
    @field_validator("value", mode="before")
    @classmethod
    def _coerce_missing_to_none(cls, v: object) -> object:
        if isinstance(v, str):
            stripped = v.strip().upper()
            if stripped in {"", "-", "—", "N/A", "NA", "NULL", "NONE"}:
                return None
        return v


class FinancialReport(_Model):
    """A `report` sub-object inside a financials-reported entry."""

    bs: list[FinancialLineItem] = Field(default_factory=list)
    ic: list[FinancialLineItem] = Field(default_factory=list)
    cf: list[FinancialLineItem] = Field(default_factory=list)


class FinancialsEntry(_Model):
    """One filing entry in /stock/financials-reported data[]."""

    access_number: str | None = None
    cik: str | None = None
    end_date: str | None = None
    filed_date: str | None = None
    form: str | None = None
    start_date: str | None = None
    symbol: str
    year: int | None = None
    quarter: int | None = None
    report: FinancialReport = Field(default_factory=FinancialReport)


class FinancialsResponse(_Model):
    cik: str | None = None
    symbol: str
    data: list[FinancialsEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# XBRL concept extraction helpers
# ---------------------------------------------------------------------------


# Map our logical field names to the ordered list of XBRL concepts to try.
# Finnhub (and SEC) expose multiple concept variants; we search in order.
CONCEPT_CANDIDATES: dict[str, list[str]] = {
    "revenue": [
        "us-gaap_Revenues",
        "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap_SalesRevenueNet",
        "us-gaap_RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "gross_profit": ["us-gaap_GrossProfit"],
    "operating_income": [
        "us-gaap_OperatingIncomeLoss",
    ],
    "net_income": [
        "us-gaap_NetIncomeLoss",
        "us-gaap_ProfitLoss",
    ],
    "eps_diluted": [
        "us-gaap_EarningsPerShareDiluted",
    ],
    "eps_basic": [
        "us-gaap_EarningsPerShareBasic",
    ],
    "depreciation_and_amortization": [
        "us-gaap_DepreciationDepletionAndAmortization",
        "us-gaap_DepreciationAndAmortization",
        "us-gaap_Depreciation",
    ],
    "total_assets": ["us-gaap_Assets"],
    "total_stockholders_equity": [
        "us-gaap_StockholdersEquity",
        "us-gaap_StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "cash_and_cash_equivalents": [
        "us-gaap_CashAndCashEquivalentsAtCarryingValue",
        "us-gaap_Cash",
    ],
    "long_term_debt": [
        "us-gaap_LongTermDebtNoncurrent",
        "us-gaap_LongTermDebt",
    ],
    "short_term_debt": [
        "us-gaap_LongTermDebtCurrent",
        "us-gaap_CommercialPaper",
        "us-gaap_ShortTermBorrowings",
    ],
    "operating_cash_flow": [
        "us-gaap_NetCashProvidedByUsedInOperatingActivities",
    ],
    "capital_expenditure": [
        # Order matters: most specific / common first.
        "us-gaap_PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap_PaymentsToAcquireProductiveAssets",  # used by NVDA
        "us-gaap_PaymentsToAcquireMachineryAndEquipment",
        "us-gaap_PaymentsForCapitalImprovements",
    ],
}


def extract_concept(items: list[FinancialLineItem], concepts: list[str]) -> float | None:
    """Find the first line item matching any of the candidate concepts.

    Uses case-insensitive match on the `concept` field. Returns None if none
    match or if the matched item has no value.
    """
    concepts_lower = [c.lower() for c in concepts]
    for item in items:
        if not item.concept:
            continue
        if item.concept.lower() in concepts_lower:
            return item.value
    return None


def extract_field(entry: FinancialsEntry, field: str) -> float | None:
    """Extract one of our logical fields from a financials-reported entry.

    Looks in ic for income-related, bs for balance-sheet, cf for cash-flow.
    """
    concepts = CONCEPT_CANDIDATES.get(field)
    if concepts is None:
        raise ValueError(f"Unknown logical field '{field}'")

    # Routing by field name: income statement fields live in `ic`, balance
    # sheet in `bs`, cash flow in `cf`. A few overlap; try the most likely
    # bucket first.
    if field in {
        "revenue", "gross_profit", "operating_income", "net_income",
        "eps_diluted", "eps_basic",
    }:
        return extract_concept(entry.report.ic, concepts)
    if field in {
        "total_assets", "total_stockholders_equity", "cash_and_cash_equivalents",
        "long_term_debt", "short_term_debt",
    }:
        return extract_concept(entry.report.bs, concepts)
    if field in {"operating_cash_flow", "capital_expenditure", "depreciation_and_amortization"}:
        val = extract_concept(entry.report.cf, concepts)
        # D&A sometimes appears in income statement instead
        if val is None and field == "depreciation_and_amortization":
            val = extract_concept(entry.report.ic, concepts)
        return val
    return None


def total_debt(entry: FinancialsEntry) -> float | None:
    """Sum short-term + long-term debt; None if neither is present."""
    lt = extract_field(entry, "long_term_debt")
    st = extract_field(entry, "short_term_debt")
    if lt is None and st is None:
        return None
    return (lt or 0.0) + (st or 0.0)


def derive_free_cash_flow(entry: FinancialsEntry) -> float | None:
    """Free Cash Flow = Operating CF − |CapEx|."""
    ocf = extract_field(entry, "operating_cash_flow")
    capex = extract_field(entry, "capital_expenditure")
    if ocf is None or capex is None:
        return None
    # Finnhub/SEC report capex as a positive number; subtract magnitude.
    return ocf - abs(capex)


def derive_ebitda(entry: FinancialsEntry) -> float | None:
    """EBITDA ≈ Operating income + D&A (approximation)."""
    oi = extract_field(entry, "operating_income")
    da = extract_field(entry, "depreciation_and_amortization")
    if oi is None:
        return None
    return oi + (da or 0.0)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FinnhubClient:
    """Thin sync client over the Finnhub /api/v1 endpoints we depend on."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.finnhub_api_key
        if not self.api_key or self.api_key == "your_finnhub_api_key_here":
            raise FinnhubError("Finnhub API key missing. Set FINNHUB_API_KEY in .env.")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> FinnhubClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, **params: Any) -> Any:
        full_params = {k: v for k, v in params.items() if v is not None}
        full_params["token"] = self.api_key
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.get(f"{self.base_url}{path}", params=full_params)
            except httpx.TransportError as e:
                last_exc = e
                sleep_for = min(2 ** attempt, 4)
                logger.warning("Finnhub transport error on %s: %s; retry in %ds", path, e, sleep_for)
                time.sleep(sleep_for)
                continue

            if r.status_code == 429:
                # 60 calls/min: back off a second or two and retry.
                sleep_for = min(2 ** attempt, 4)
                logger.warning("Finnhub 429 on %s; retry in %ds", path, sleep_for)
                time.sleep(sleep_for)
                continue

            if r.status_code >= 400:
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:200]
                raise FinnhubError(f"HTTP {r.status_code} on {path}: {body}")

            try:
                return r.json()
            except Exception as e:
                raise FinnhubError(f"JSON parse failed on {path}: {e}") from e

        raise FinnhubError(f"{path} failed after {self.max_retries} retries: {last_exc}")

    # ----- raw endpoints -----

    def quote(self, symbol: str) -> Quote:
        data = self._get("/quote", symbol=symbol.upper())
        if not data or data.get("c") in (0, None):
            raise FinnhubError(f"quote: no current price for {symbol}")
        return Quote.model_validate(data)

    def profile(self, symbol: str) -> Profile:
        data = self._get("/stock/profile2", symbol=symbol.upper())
        if not data:
            raise FinnhubError(f"profile: empty for {symbol}")
        return Profile.model_validate(data)

    def metric(self, symbol: str) -> MetricResponse:
        data = self._get("/stock/metric", symbol=symbol.upper(), metric="all")
        if not data or "metric" not in data:
            raise FinnhubError(f"metric: empty for {symbol}")
        return MetricResponse.model_validate(data)

    def financials(self, symbol: str, freq: str = "annual") -> FinancialsResponse:
        # Apply alias normalization here, not at call sites — the alias
        # is specific to the financials-reported endpoint, not to other
        # Finnhub APIs (price/profile/peers all still resolve under the
        # original ticker).
        data = self._get(
            "/stock/financials-reported",
            symbol=_financials_symbol(symbol),
            freq=freq,
        )
        return FinancialsResponse.model_validate(data)

    def peers(self, symbol: str) -> list[str]:
        data = self._get("/stock/peers", symbol=symbol.upper())
        if not isinstance(data, list):
            return []
        # Filter to strings and exclude self-listing duplicates for cleanliness.
        return [s for s in data if isinstance(s, str)]

    # ----- convenience: latest annual financials entry -----

    def latest_annual_financials(self, symbol: str) -> FinancialsEntry | None:
        resp = self.financials(symbol, freq="annual")
        for entry in resp.data:
            # Most recent annual 10-K (quarter == 0 in Finnhub convention) at top
            if entry.form and entry.form.upper().startswith("10-K"):
                return entry
        # Fallback: first entry regardless of form
        return resp.data[0] if resp.data else None


def get_client() -> FinnhubClient:
    return FinnhubClient()
