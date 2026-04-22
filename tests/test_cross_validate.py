"""Tests for the yfinance wrapper and FMP↔yfinance cross-validation.

Offline tests exercise the comparison math with injected values.
Network tests hit both FMP and yfinance against AAPL.
"""

from __future__ import annotations

import pytest

from wise_investor.config import settings
from wise_investor.data.cross_validate import (
    DEFAULT_THRESHOLD_PCT,
    compare_value,
    cross_validate_quote,
)
from wise_investor.data.finnhub import FinnhubClient, Profile, Quote
from wise_investor.data.yf import YFQuote


# ---------------------------------------------------------------------------
# Offline: compare_value math
# ---------------------------------------------------------------------------


def test_compare_value_within_threshold() -> None:
    c = compare_value("price", 100.0, 102.0, 5.0)
    assert c.within_threshold is True
    assert c.diff_pct == 2.0


def test_compare_value_exceeds_threshold() -> None:
    c = compare_value("price", 100.0, 107.0, 5.0)
    assert c.within_threshold is False
    assert c.diff_pct == 7.0


def test_compare_value_handles_both_missing() -> None:
    c = compare_value("pe", None, None, 5.0)
    assert c.within_threshold is None
    assert c.note and "missing" in c.note


def test_compare_value_handles_one_missing() -> None:
    c = compare_value("pe", 30.0, None, 5.0)
    assert c.within_threshold is None
    assert c.diff_pct is None
    assert c.fmp_value == 30.0


def test_compare_value_fmp_zero_uses_absolute_equality() -> None:
    same = compare_value("pe", 0.0, 0.0, 5.0)
    assert same.within_threshold is True
    diff = compare_value("pe", 0.0, 1.0, 5.0)
    assert diff.within_threshold is False


def test_compare_value_exact_threshold_boundary() -> None:
    # Exactly at threshold counts as within.
    c = compare_value("price", 100.0, 105.0, 5.0)
    assert c.within_threshold is True
    assert c.diff_pct == 5.0


def test_compare_value_negative_direction_uses_absolute() -> None:
    # yfinance lower than FMP should still diff symmetrically.
    c = compare_value("price", 100.0, 93.0, 5.0)
    assert c.within_threshold is False
    assert c.diff_pct == 7.0


# ---------------------------------------------------------------------------
# Offline: cross_validate_quote with injected values
# ---------------------------------------------------------------------------


class _StubFinnhub:
    """Minimal FinnhubClient-shaped stand-in with canned quote + profile."""

    def __init__(self, price: float, market_cap_usd: float) -> None:
        self._quote = Quote(c=price)
        # profile stores market cap in MILLIONS; market_cap_usd helper multiplies by 1M.
        self._profile = Profile(marketCapitalization=market_cap_usd / 1_000_000.0)

    def quote(self, symbol: str) -> Quote:  # noqa: ARG002
        return self._quote

    def profile(self, symbol: str) -> Profile:  # noqa: ARG002
        return self._profile

    def close(self) -> None:
        pass


def test_cross_validate_all_fields_within() -> None:
    yf_quote = YFQuote(symbol="AAPL", price=181.0, market_cap=2.81e12)
    result = cross_validate_quote(
        "AAPL",
        fmp=_StubFinnhub(price=180.0, market_cap_usd=2.8e12),  # type: ignore[arg-type]
        yf_quote=yf_quote,
    )
    assert result.symbol == "AAPL"
    assert result.threshold_pct == DEFAULT_THRESHOLD_PCT
    assert not result.any_flagged
    assert {c.field for c in result.comparisons} == {"price", "market_cap"}


def test_cross_validate_flags_divergence() -> None:
    # 10% price difference should be flagged at 5% threshold.
    yf_quote = YFQuote(symbol="AAPL", price=198.0, market_cap=2.81e12)
    result = cross_validate_quote(
        "AAPL",
        fmp=_StubFinnhub(price=180.0, market_cap_usd=2.8e12),  # type: ignore[arg-type]
        yf_quote=yf_quote,
    )
    assert result.any_flagged
    price_cmp = next(c for c in result.comparisons if c.field == "price")
    assert price_cmp.within_threshold is False


# ---------------------------------------------------------------------------
# Network: real FMP + yfinance for AAPL
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_network_cross_validate_aapl_within_reasonable_bounds() -> None:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        pytest.skip("FINNHUB_API_KEY not set")

    with FinnhubClient() as fmp:
        result = cross_validate_quote("AAPL", fmp=fmp, threshold_pct=10.0)

    # Both sources present for price — market is the same underlying reality,
    # so we expect agreement within 10% even accounting for timing/stale data.
    price = next(c for c in result.comparisons if c.field == "price")
    assert price.fmp_value is not None and price.fmp_value > 0
    if price.yf_value is not None:
        assert price.within_threshold is True, (
            f"Price diverged: FMP={price.fmp_value} vs YF={price.yf_value} ({price.diff_pct}%)"
        )
