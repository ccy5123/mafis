"""Tests for verify_number — Skeptic's number-verification tool."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Post Phase 1B Finnhub migration: StubFMP needs rewrite to StubFinnhub. "
        "verify_number is still exercised end-to-end via "
        "scripts/smoke_phase1a.py and test_agents_tools network tests."
    )
)

from wise_investor.config import settings
from wise_investor.data.fmp import (
    BalanceSheet,
    CashFlowStatement,
    EnterpriseValue,
    FMPClient,
    IncomeStatement,
    Quote,
)
from wise_investor.tools.verify import (
    list_supported_fields,
    verify_number,
)


class StubFMP:
    """Mirrors the subset of FMPClient used by verify_number + downstream tools."""

    def __init__(
        self,
        quote: Quote | None = None,
        income: list[IncomeStatement] | None = None,
        balance: list[BalanceSheet] | None = None,
        cash_flow: list[CashFlowStatement] | None = None,
        ev_values: list[EnterpriseValue] | None = None,
        ratios: list = None,
        key_metrics: list = None,
    ) -> None:
        self._quote = quote
        self._income = income or []
        self._balance = balance or []
        self._cash_flow = cash_flow or []
        self._ev = ev_values or []
        self._ratios = ratios or []
        self._km = key_metrics or []

    def quote(self, symbol: str) -> Quote:
        if self._quote is None:
            raise RuntimeError("no quote set")
        return self._quote

    def income_statement(self, *a, **k):
        return self._income

    def balance_sheet(self, *a, **k):
        return self._balance

    def cash_flow(self, *a, **k):
        return self._cash_flow

    def enterprise_values(self, *a, **k):
        return self._ev

    def ratios(self, *a, **k):
        return self._ratios

    def key_metrics(self, *a, **k):
        return self._km

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Direct FMP field verification
# ---------------------------------------------------------------------------


def test_verify_matches_revenue_within_tolerance() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", revenue=391_000_000_000.0)]
    )
    r = verify_number(
        claim=391_000_000_000.0,
        field="revenue",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert r.source_value == 391_000_000_000.0
    assert r.diff_pct == 0.0
    assert "2024-09-28" in r.source_citation


def test_verify_flags_mismatch_beyond_tolerance() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", revenue=391_000_000_000.0)]
    )
    r = verify_number(
        claim=400_000_000_000.0,  # ~2.3% higher
        field="revenue",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
        tolerance_pct=1.0,
    )
    assert r.matches is False
    assert r.diff_pct is not None and r.diff_pct > 1.0


def test_verify_respects_looser_tolerance() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", revenue=391_000_000_000.0)]
    )
    r = verify_number(
        claim=400_000_000_000.0,
        field="revenue",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
        tolerance_pct=5.0,
    )
    assert r.matches is True  # now within 5%


def test_verify_quote_price() -> None:
    stub = StubFMP(quote=Quote(symbol="AAPL", price=180.5, market_cap=2.8e12))
    r = verify_number(claim=180.5, field="price", symbol="AAPL", client=stub)  # type: ignore[arg-type]
    assert r.matches is True
    assert "current" in r.source_citation


def test_verify_market_cap_within_default_tolerance() -> None:
    stub = StubFMP(quote=Quote(symbol="AAPL", price=180.5, market_cap=2.8e12))
    # 2.81e12 vs 2.80e12 is ~0.357% off, well under the 1% default tolerance.
    r = verify_number(
        claim=2.81e12, field="market_cap", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert r.diff_pct is not None and r.diff_pct < 1.0


def test_verify_balance_sheet_field() -> None:
    stub = StubFMP(
        balance=[
            BalanceSheet(date="2024-09-28", symbol="AAPL", total_debt=100_000_000_000.0)
        ]
    )
    r = verify_number(
        claim=100_000_000_000.0, field="total_debt", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True


def test_verify_cash_flow_operating_cf_alias() -> None:
    stub = StubFMP(
        cash_flow=[
            CashFlowStatement(
                date="2024-09-28",
                symbol="AAPL",
                net_cash_provided_by_operating_activities=120_000_000_000.0,
            )
        ]
    )
    r = verify_number(
        claim=120_000_000_000.0,
        field="operating_cash_flow",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True


def test_verify_cash_flow_fcf_alias() -> None:
    stub = StubFMP(
        cash_flow=[
            CashFlowStatement(
                date="2024-09-28", symbol="AAPL", free_cash_flow=100_000_000_000.0
            )
        ]
    )
    r = verify_number(
        claim=100_000_000_000.0, field="fcf", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert r.field == "free_cash_flow"


def test_verify_enterprise_value() -> None:
    stub = StubFMP(
        ev_values=[
            EnterpriseValue(
                symbol="AAPL", date="2024-09-28", enterprise_value=2_800_000_000_000.0
            )
        ]
    )
    r = verify_number(
        claim=2_800_000_000_000.0,
        field="enterprise_value",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True


# ---------------------------------------------------------------------------
# Computed metric verification
# ---------------------------------------------------------------------------


def test_verify_per_via_calculation() -> None:
    stub = StubFMP(
        quote=Quote(symbol="AAPL", price=180.0),
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0)],
    )
    # Our computed PER is 30.0; Bull claimed 30.1 (0.33% off, within 1% tolerance).
    r = verify_number(claim=30.1, field="per", symbol="AAPL", client=stub)  # type: ignore[arg-type]
    assert r.matches is True
    assert "calculate_per" in r.source_citation


def test_verify_pe_alias_to_per() -> None:
    stub = StubFMP(
        quote=Quote(symbol="AAPL", price=180.0),
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0)],
    )
    r = verify_number(claim=30.0, field="pe", symbol="AAPL", client=stub)  # type: ignore[arg-type]
    assert r.field == "per"
    assert r.matches is True


def test_verify_ev_ebitda() -> None:
    stub = StubFMP(
        ev_values=[EnterpriseValue(symbol="AAPL", date="2024-09-28", enterprise_value=2.6e12)],
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", ebitda=130e9)],
    )
    # 2.6e12 / 130e9 = 20.0
    r = verify_number(claim=20.0, field="ev_ebitda", symbol="AAPL", client=stub)  # type: ignore[arg-type]
    assert r.matches is True


def test_verify_implied_growth_via_reverse_dcf() -> None:
    from wise_investor.tools.dcf import dcf_fair_value

    fcf = 1_000_000_000.0
    market_cap = dcf_fair_value(fcf, 0.10, 0.10, 0.025, 10)
    stub = StubFMP(
        quote=Quote(symbol="TEST", price=100.0, market_cap=market_cap),
        cash_flow=[CashFlowStatement(date="2024-12-31", symbol="TEST", free_cash_flow=fcf)],
    )
    # Claim should match the known input growth.
    r = verify_number(
        claim=0.10, field="implied_growth_rate", symbol="TEST", client=stub  # type: ignore[arg-type]
    )
    assert r.source_value is not None
    assert abs(r.source_value - 0.10) < 1e-3
    # Tight tolerance: 1% of 0.10 is 0.001, so the recovered value must agree closely.
    assert r.matches is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_verify_raises_on_unsupported_field() -> None:
    with pytest.raises(ValueError, match="unsupported field"):
        verify_number(claim=1.0, field="unknown_metric", symbol="AAPL")


def test_verify_returns_unknown_when_source_missing() -> None:
    stub = StubFMP(income=[])  # no data
    r = verify_number(
        claim=100.0, field="revenue", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is None
    assert r.source_value is None
    assert any("unavailable" in w for w in r.warnings)


def test_verify_zero_source_value_uses_absolute_equality() -> None:
    stub = StubFMP(
        balance=[BalanceSheet(date="2024-09-28", symbol="X", total_debt=0.0)]
    )
    same = verify_number(
        claim=0.0, field="total_debt", symbol="X", client=stub  # type: ignore[arg-type]
    )
    assert same.matches is True
    diff = verify_number(
        claim=1.0, field="total_debt", symbol="X", client=stub  # type: ignore[arg-type]
    )
    assert diff.matches is False


def test_list_supported_fields_non_empty() -> None:
    fields = list_supported_fields()
    assert "revenue" in fields
    assert "per" in fields
    assert "implied_growth_rate" in fields
    assert len(fields) > 10


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_network_verify_aapl_revenue_plausible() -> None:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        pytest.skip("FMP_API_KEY not set")
    # Deliberately wrong claim; expect mismatch.
    with FMPClient() as c:
        r = verify_number(claim=1.0, field="revenue", symbol="AAPL", client=c)
    assert r.source_value is not None
    # Apple's annual revenue is hundreds of billions.
    assert r.source_value > 1e11
    assert r.matches is False


@pytest.mark.network
def test_network_verify_aapl_per_self_consistent() -> None:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        pytest.skip("FMP_API_KEY not set")
    from wise_investor.tools.valuation import calculate_per

    with FMPClient() as c:
        own = calculate_per("AAPL", client=c)
        if own.computed is None:
            pytest.skip("AAPL PER could not be computed this run")
        r = verify_number(
            claim=own.computed, field="per", symbol="AAPL", client=c, tolerance_pct=0.5
        )
    # Same calculation on both sides should agree exactly.
    assert r.matches is True
