"""Reverse DCF — solve for the implied growth rate the market is pricing in.

Post Phase 1B migration: backed by Finnhub. Market cap from /stock/profile2
(USD millions → dollars), FCF derived from latest 10-K's operating cash flow
and capex via /stock/financials-reported XBRL extraction.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from wise_investor.data.finnhub import (
    FinnhubClient,
    derive_free_cash_flow,
)


DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_TERMINAL_GROWTH = 0.025
DEFAULT_HIGH_GROWTH_YEARS = 10
SEARCH_LO = -0.30
SEARCH_HI = 1.00


class ReverseDCFResult(BaseModel):
    symbol: str
    metric: str = "reverse_dcf_implied_growth"
    implied_growth_rate: float | None
    current_market_cap: float | None
    inputs: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    as_of: str | None = None


def dcf_fair_value(
    fcf_0: float,
    growth_rate: float,
    discount_rate: float,
    terminal_growth: float,
    high_growth_years: int,
) -> float:
    r = discount_rate
    g = growth_rate
    g_t = terminal_growth
    n = high_growth_years

    pv_stage1 = 0.0
    for t in range(1, n + 1):
        fcf_t = fcf_0 * (1.0 + g) ** t
        pv_stage1 += fcf_t / (1.0 + r) ** t

    fcf_n_plus_1 = fcf_0 * (1.0 + g) ** n * (1.0 + g_t)
    terminal_value = fcf_n_plus_1 / (r - g_t)
    pv_terminal = terminal_value / (1.0 + r) ** n

    return pv_stage1 + pv_terminal


def _bisect(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> float | None:
    f_lo = f(lo)
    f_hi = f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if f_lo * f_hi > 0:
        return None

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) / 2.0 < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


def solve_implied_growth(
    market_cap: float,
    fcf_0: float,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    high_growth_years: int = DEFAULT_HIGH_GROWTH_YEARS,
) -> float | None:
    if discount_rate <= terminal_growth:
        raise ValueError(
            f"discount_rate ({discount_rate}) must exceed terminal_growth ({terminal_growth})"
        )
    if fcf_0 <= 0:
        raise ValueError(f"fcf_0 must be positive, got {fcf_0}")
    if market_cap <= 0:
        raise ValueError(f"market_cap must be positive, got {market_cap}")

    def f(g: float) -> float:
        return dcf_fair_value(fcf_0, g, discount_rate, terminal_growth, high_growth_years) - market_cap

    return _bisect(f, SEARCH_LO, SEARCH_HI)


def reverse_dcf(
    symbol: str,
    client: FinnhubClient | None = None,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    terminal_growth: float = DEFAULT_TERMINAL_GROWTH,
    high_growth_years: int = DEFAULT_HIGH_GROWTH_YEARS,
) -> ReverseDCFResult:
    """Compute the implied annual FCF growth rate for a symbol at its current price."""
    owned = False
    if client is None:
        client = FinnhubClient()
        owned = True

    warnings: list[str] = []
    inputs: dict[str, Any] = {
        "discount_rate": discount_rate,
        "terminal_growth": terminal_growth,
        "high_growth_years": high_growth_years,
    }
    try:
        profile = client.profile(symbol)
        quote = client.quote(symbol)
        latest = client.latest_annual_financials(symbol)
    finally:
        if owned:
            client.close()

    market_cap = profile.market_cap_usd
    inputs["market_cap"] = market_cap
    inputs["price"] = quote.price

    if market_cap is None or market_cap <= 0:
        warnings.append("market cap unavailable or non-positive")
        return ReverseDCFResult(
            symbol=symbol.upper(),
            implied_growth_rate=None,
            current_market_cap=market_cap,
            inputs=inputs,
            warnings=warnings,
        )

    if latest is None:
        warnings.append("no annual financials available")
        return ReverseDCFResult(
            symbol=symbol.upper(),
            implied_growth_rate=None,
            current_market_cap=market_cap,
            inputs=inputs,
            warnings=warnings,
        )

    fcf = derive_free_cash_flow(latest)
    inputs["fcf_source"] = "derived (operating_cash_flow − |capital_expenditure|)"
    inputs["fcf_latest_annual"] = fcf
    inputs["fiscal_date"] = str(latest.end_date) if latest.end_date else None

    if fcf is None:
        warnings.append("operating_cash_flow or capital_expenditure unavailable — cannot derive FCF")
        return ReverseDCFResult(
            symbol=symbol.upper(),
            implied_growth_rate=None,
            current_market_cap=market_cap,
            inputs=inputs,
            warnings=warnings,
            as_of=str(latest.end_date) if latest.end_date else None,
        )

    if fcf <= 0:
        warnings.append(
            f"FCF <= 0 ({fcf}); reverse DCF is not meaningful on negative cash flow"
        )
        return ReverseDCFResult(
            symbol=symbol.upper(),
            implied_growth_rate=None,
            current_market_cap=market_cap,
            inputs=inputs,
            warnings=warnings,
            as_of=str(latest.end_date) if latest.end_date else None,
        )

    implied_g = solve_implied_growth(
        market_cap=market_cap,
        fcf_0=fcf,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        high_growth_years=high_growth_years,
    )

    if implied_g is None:
        low_val = dcf_fair_value(fcf, SEARCH_LO, discount_rate, terminal_growth, high_growth_years)
        if market_cap < low_val:
            warnings.append(
                f"market cap below DCF at {SEARCH_LO:.0%} growth — market implies "
                f"extreme FCF decline beyond search range"
            )
        else:
            warnings.append(
                f"market cap above DCF at {SEARCH_HI:.0%} growth — market implies "
                f"growth above search range (>100%/yr for {high_growth_years}y)"
            )
    elif implied_g > 0.25:
        warnings.append(
            f"implied growth {implied_g:.1%} is unusually high — stress-test the assumption"
        )
    elif implied_g < 0:
        warnings.append(
            f"implied growth {implied_g:.1%} is negative — market pricing in FCF decline"
        )

    return ReverseDCFResult(
        symbol=symbol.upper(),
        implied_growth_rate=round(implied_g, 5) if implied_g is not None else None,
        current_market_cap=market_cap,
        inputs=inputs,
        warnings=warnings,
        as_of=str(latest.end_date) if latest.end_date else None,
    )
