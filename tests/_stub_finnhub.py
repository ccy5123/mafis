"""Shared FinnhubClient-shaped stub for offline tests.

The valuation/DCF/verify modules all depend on the same seven client
methods: quote, profile, metric, financials, latest_annual_financials,
peers, close. This module defines one `StubFinnhub` that mirrors that
surface, plus a handful of factory helpers that translate test-friendly
inputs (revenue in dollars, EPS as a plain float) into the Finnhub
wire shapes (XBRL line items tagged with us-gaap concepts, values in
millions for profile/metric).

Usage:

    from tests._stub_finnhub import StubFinnhub, make_financials_entry, make_metric

    stub = StubFinnhub(
        quote_price=180.0,
        financials=[make_financials_entry("AAPL", end_date="2024-09-28",
                                          ic={"revenue": 391e9, "eps_diluted": 6.0})],
        metric=make_metric(pe_annual=30.0),
    )
    r = calculate_per("AAPL", client=stub)

Values are always passed in their natural unit (dollars, shares, etc.).
The helpers convert to Finnhub's "millions" internal convention so the
downstream `.market_cap_usd` / `.enterprise_value_usd` properties return
the same dollar values you put in.
"""

from __future__ import annotations

from typing import Any

from wise_investor.data.finnhub import (
    CONCEPT_CANDIDATES,
    DOLLAR_MILLIONS,
    FinancialLineItem,
    FinancialReport,
    FinancialsEntry,
    FinancialsResponse,
    Metric,
    MetricResponse,
    Profile,
    Quote,
)


# Logical-field -> which bucket it lives in inside FinancialReport.
_IC_FIELDS = {
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "eps_diluted",
    "eps_basic",
}
_BS_FIELDS = {
    "total_assets",
    "total_stockholders_equity",
    "cash_and_cash_equivalents",
    "long_term_debt",
    "short_term_debt",
}
_CF_FIELDS = {
    "operating_cash_flow",
    "capital_expenditure",
    "depreciation_and_amortization",  # may also live in ic; we place it in cf by default
}


def make_line_item(logical_field: str, value: float) -> FinancialLineItem:
    """Create one FinancialLineItem tagged with the primary XBRL concept."""
    concepts = CONCEPT_CANDIDATES.get(logical_field)
    if not concepts:
        raise KeyError(f"No XBRL concept candidates for logical field {logical_field!r}")
    return FinancialLineItem(concept=concepts[0], value=value)


def make_financials_entry(
    symbol: str,
    end_date: str = "2024-12-31",
    form: str = "10-K",
    ic: dict[str, float] | None = None,
    bs: dict[str, float] | None = None,
    cf: dict[str, float] | None = None,
) -> FinancialsEntry:
    """Build a FinancialsEntry from simple {logical_field: value} dicts.

    Unknown logical fields raise — this keeps typos from silently producing
    empty line items.
    """
    ic_items = [make_line_item(k, v) for k, v in (ic or {}).items()]
    bs_items = [make_line_item(k, v) for k, v in (bs or {}).items()]
    cf_items = [make_line_item(k, v) for k, v in (cf or {}).items()]
    return FinancialsEntry(
        symbol=symbol.upper(),
        end_date=end_date,
        form=form,
        report=FinancialReport(ic=ic_items, bs=bs_items, cf=cf_items),
    )


def make_profile(
    *,
    market_cap: float | None = None,
    name: str | None = None,
    ticker: str | None = None,
    industry: str | None = None,
) -> Profile:
    """Create a Profile from a market cap in DOLLARS.

    Internally stored as millions to match Finnhub's wire shape so
    `.market_cap_usd` returns the dollar value you supplied.
    """
    mc_millions = None if market_cap is None else market_cap / DOLLAR_MILLIONS
    return Profile(
        market_capitalization=mc_millions,
        name=name,
        ticker=ticker,
        finnhub_industry=industry,
    )


def make_metric(
    *,
    enterprise_value: float | None = None,  # DOLLARS
    pe_annual: float | None = None,
    pe_ttm: float | None = None,
    ev_ebitda_ttm: float | None = None,
    ev_revenue_ttm: float | None = None,
) -> Metric:
    """Create a Metric from an enterprise value in DOLLARS.

    Internally stored as millions so `.enterprise_value_usd` returns the
    dollar value you supplied.
    """
    ev_millions = None if enterprise_value is None else enterprise_value / DOLLAR_MILLIONS
    return Metric(
        enterprise_value=ev_millions,
        pe_annual=pe_annual,
        pe_ttm=pe_ttm,
        ev_ebitda_ttm=ev_ebitda_ttm,
        ev_revenue_ttm=ev_revenue_ttm,
    )


class StubFinnhub:
    """FinnhubClient-shaped stand-in for offline tests.

    Construction takes default payloads (applied to any symbol) plus an
    optional `per_symbol` dict for scenarios involving peers where each
    symbol needs its own financials/profile/metric.
    """

    def __init__(
        self,
        *,
        quote_price: float | None = None,
        profile: Profile | None = None,
        metric: Metric | None = None,
        financials: list[FinancialsEntry] | None = None,
        peers: list[str] | None = None,
        per_symbol: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._default: dict[str, Any] = {
            "quote_price": quote_price,
            "profile": profile,
            "metric": metric,
            "financials": financials or [],
            "peers": peers or [],
        }
        self._per_symbol = per_symbol or {}

    def _pick(self, symbol: str, key: str) -> Any:
        s = symbol.upper()
        if s in self._per_symbol and key in self._per_symbol[s]:
            return self._per_symbol[s][key]
        return self._default[key]

    # ---- client API ---------------------------------------------------

    def quote(self, symbol: str) -> Quote:
        price = self._pick(symbol, "quote_price")
        if price is None:
            raise RuntimeError(f"StubFinnhub: no quote configured for {symbol}")
        return Quote(c=price)

    def profile(self, symbol: str) -> Profile:
        p = self._pick(symbol, "profile")
        if p is None:
            # Return an "empty" Profile so callers get None market_cap rather
            # than a RuntimeError — many tests don't supply a profile.
            return Profile()
        return p

    def metric(self, symbol: str) -> MetricResponse:
        m = self._pick(symbol, "metric")
        if m is None:
            m = Metric()
        return MetricResponse(metric=m, symbol=symbol.upper())

    def financials(self, symbol: str, freq: str = "annual") -> FinancialsResponse:
        entries = self._pick(symbol, "financials")
        return FinancialsResponse(symbol=symbol.upper(), data=list(entries))

    def latest_annual_financials(self, symbol: str) -> FinancialsEntry | None:
        entries = self._pick(symbol, "financials")
        for e in entries:
            if e.form and e.form.upper().startswith("10-K"):
                return e
        return entries[0] if entries else None

    def peers(self, symbol: str) -> list[str]:
        return list(self._pick(symbol, "peers"))

    def close(self) -> None:
        pass


__all__ = [
    "StubFinnhub",
    "make_financials_entry",
    "make_line_item",
    "make_metric",
    "make_profile",
]
