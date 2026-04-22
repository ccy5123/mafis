"""FMP (Financial Modeling Prep) /stable/ API client.

Typed wrapper over the endpoints confirmed on free-tier access via
scripts/probe_fmp.py. All numeric outputs flow through this module into
the calculation tools in src/wise_investor/tools/; LLMs never compute ratios
themselves (design-v2.2 §7).

Design notes:
- pydantic v2 with alias_generator=to_camel so Python uses snake_case while the
  wire uses camelCase (FMP convention). extra='ignore' means FMP can add new
  fields without breaking our parsing.
- Retries on 429 / 5xx / transport errors with exponential backoff.
- No caching yet (Phase 1A decision). FMP free tier is 250 calls/day — run
  deliberately during dev.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from wise_investor.config import settings


logger = logging.getLogger(__name__)


class FMPError(RuntimeError):
    """Raised when FMP returns an error payload or non-retryable HTTP failure."""


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        alias_generator=to_camel,
        populate_by_name=True,
    )


class SymbolMatch(_Model):
    symbol: str
    name: str | None = None
    currency: str | None = None
    exchange_full_name: str | None = None
    exchange: str | None = None


class Quote(_Model):
    """Response from /stable/quote.

    Note: FMP's /stable/quote returns only 17 fields — no pe, eps, or
    sharesOutstanding on the free tier. For PE use calculate_per() from tools;
    for EPS read income_statement. Those fields are retained here as None-by-
    default so the cross_validate module can safely reference them.
    """

    symbol: str
    name: str | None = None
    price: float
    change_percentage: float | None = None
    change: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    # Not populated by /stable/quote; kept for interface compatibility only.
    pe: float | None = None
    eps: float | None = None
    shares_outstanding: float | None = None


class Profile(_Model):
    symbol: str
    price: float | None = None
    market_cap: float | None = None
    beta: float | None = None
    sector: str | None = None
    industry: str | None = None
    description: str | None = None
    ceo: str | None = None
    country: str | None = None
    full_time_employees: str | None = None
    currency: str | None = None
    is_etf: bool | None = None


class IncomeStatement(_Model):
    date: str
    symbol: str
    reported_currency: str | None = None
    period: str | None = None
    fiscal_year: str | None = None
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps: float | None = None
    eps_diluted: float | None = None
    ebitda: float | None = None
    depreciation_and_amortization: float | None = None
    weighted_average_shs_out: float | None = None
    weighted_average_shs_out_dil: float | None = None


class BalanceSheet(_Model):
    date: str
    symbol: str
    reported_currency: str | None = None
    period: str | None = None
    fiscal_year: str | None = None
    cash_and_cash_equivalents: float | None = None
    short_term_investments: float | None = None
    total_current_assets: float | None = None
    total_assets: float | None = None
    short_term_debt: float | None = None
    long_term_debt: float | None = None
    total_debt: float | None = None
    total_current_liabilities: float | None = None
    total_liabilities: float | None = None
    total_stockholders_equity: float | None = None


class CashFlowStatement(_Model):
    date: str
    symbol: str
    reported_currency: str | None = None
    period: str | None = None
    fiscal_year: str | None = None
    net_cash_provided_by_operating_activities: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    dividends_paid: float | None = None
    common_stock_repurchased: float | None = None


class Ratios(_Model):
    symbol: str
    date: str
    fiscal_year: str | None = None
    period: str | None = None
    reported_currency: str | None = None
    gross_profit_margin: float | None = None
    operating_profit_margin: float | None = None
    net_profit_margin: float | None = None
    price_to_earnings_ratio: float | None = None
    price_to_sales_ratio: float | None = None
    price_to_book_ratio: float | None = None
    enterprise_value_multiple: float | None = None
    debt_to_equity_ratio: float | None = None
    return_on_equity: float | None = None
    return_on_assets: float | None = None


class KeyMetrics(_Model):
    symbol: str
    date: str
    fiscal_year: str | None = None
    period: str | None = None
    reported_currency: str | None = None
    market_cap: float | None = None
    enterprise_value: float | None = None
    ev_to_sales: float | None = None
    # FMP emits the all-caps acronym `evToEBITDA`, not the auto-generated
    # `evToEbitda`. Explicit alias overrides the alias_generator.
    ev_to_ebitda: float | None = Field(default=None, alias="evToEBITDA")
    ev_to_operating_cash_flow: float | None = None
    ev_to_free_cash_flow: float | None = None
    free_cash_flow_per_share: float | None = None
    working_capital: float | None = None


class EnterpriseValue(_Model):
    symbol: str
    date: str
    stock_price: float | None = None
    number_of_shares: float | None = None
    market_capitalization: float | None = None
    minus_cash_and_cash_equivalents: float | None = None
    add_total_debt: float | None = None
    enterprise_value: float | None = None


class HistoricalPrice(_Model):
    symbol: str | None = None
    date: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None


class StockPeer(_Model):
    symbol: str
    company_name: str | None = None
    price: float | None = None
    mkt_cap: float | None = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FMPClient:
    """Thin sync client over FMP /stable/ endpoints.

    Usage:
        with FMPClient() as c:
            q = c.quote("AAPL")
            ...
    """

    BASE_URL = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.fmp_api_key
        if not self.api_key or self.api_key == "your_fmp_api_key_here":
            raise FMPError("FMP API key missing. Set FMP_API_KEY in .env.")
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FMPClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- low-level -------------------------------------------------------

    def _get(self, path: str, **params: Any) -> Any:
        full_params = {k: v for k, v in params.items() if v is not None}
        full_params["apikey"] = self.api_key
        logged = {k: v for k, v in full_params.items() if k != "apikey"}
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("FMP GET %s params=%s (attempt %d)", path, logged, attempt)
                r = self._client.get(f"{self.base_url}{path}", params=full_params)
            except httpx.TransportError as e:
                last_exc = e
                sleep_for = min(2 ** attempt, 8)
                logger.warning("FMP transport error on %s: %s; retry in %ds", path, e, sleep_for)
                time.sleep(sleep_for)
                continue

            if r.status_code == 429:
                # Free-tier 250/day quota uses a daily window — retrying within
                # seconds is pointless. Fail fast so callers can handle it.
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:200]
                raise FMPError(
                    f"HTTP 429 on {path} — daily API quota likely exhausted. "
                    f"Body: {body}"
                )

            if 500 <= r.status_code < 600:
                sleep_for = min(2 ** attempt, 8)
                logger.warning("FMP %d on %s; retry in %ds", r.status_code, path, sleep_for)
                time.sleep(sleep_for)
                continue

            if r.status_code >= 400:
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:200]
                raise FMPError(f"HTTP {r.status_code} on {path}: {body}")

            try:
                body = r.json()
            except Exception as e:
                raise FMPError(f"JSON parse failed on {path}: {e}") from e

            if isinstance(body, dict) and "Error Message" in body:
                raise FMPError(f"FMP error on {path}: {body['Error Message']}")
            return body

        raise FMPError(f"{path} failed after {self.max_retries} retries: {last_exc}")

    # -- endpoint wrappers ----------------------------------------------

    def search_symbol(self, query: str) -> list[SymbolMatch]:
        data = self._get("/search-symbol", query=query)
        return [SymbolMatch.model_validate(d) for d in data]

    def quote(self, symbol: str) -> Quote:
        data = self._get("/quote", symbol=symbol)
        if not data:
            raise FMPError(f"quote: no data for {symbol}")
        return Quote.model_validate(data[0])

    def profile(self, symbol: str) -> Profile:
        data = self._get("/profile", symbol=symbol)
        if not data:
            raise FMPError(f"profile: no data for {symbol}")
        return Profile.model_validate(data[0])

    def income_statement(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[IncomeStatement]:
        data = self._get("/income-statement", symbol=symbol, period=period, limit=limit)
        return [IncomeStatement.model_validate(d) for d in data]

    def balance_sheet(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[BalanceSheet]:
        data = self._get("/balance-sheet-statement", symbol=symbol, period=period, limit=limit)
        return [BalanceSheet.model_validate(d) for d in data]

    def cash_flow(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[CashFlowStatement]:
        data = self._get("/cash-flow-statement", symbol=symbol, period=period, limit=limit)
        return [CashFlowStatement.model_validate(d) for d in data]

    def ratios(self, symbol: str, period: str = "annual", limit: int = 5) -> list[Ratios]:
        data = self._get("/ratios", symbol=symbol, period=period, limit=limit)
        return [Ratios.model_validate(d) for d in data]

    def key_metrics(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[KeyMetrics]:
        data = self._get("/key-metrics", symbol=symbol, period=period, limit=limit)
        return [KeyMetrics.model_validate(d) for d in data]

    def enterprise_values(
        self, symbol: str, period: str = "annual", limit: int = 5
    ) -> list[EnterpriseValue]:
        data = self._get("/enterprise-values", symbol=symbol, period=period, limit=limit)
        return [EnterpriseValue.model_validate(d) for d in data]

    def historical_prices(self, symbol: str) -> list[HistoricalPrice]:
        data = self._get("/historical-price-eod/full", symbol=symbol)
        # Some endpoints wrap payload in {"symbol":..., "historical": [...]}; /stable
        # returns a flat list, but handle both for safety.
        if isinstance(data, dict) and "historical" in data:
            data = data["historical"]
        return [HistoricalPrice.model_validate(d) for d in data]

    def stock_peers(self, symbol: str) -> list[StockPeer]:
        data = self._get("/stock-peers", symbol=symbol)
        return [StockPeer.model_validate(d) for d in data]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def get_client() -> FMPClient:
    """Return a new FMPClient using the configured API key."""
    return FMPClient()
