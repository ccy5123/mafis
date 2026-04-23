"""Tests for verify_number — Skeptic's number-verification tool (Finnhub-backed).

Port of the original StubFMP tests onto StubFinnhub. Covers the read
paths (quote, profile, metric, financials-reported) and the computed
paths (per, ev_ebitda, implied_growth_rate).
"""

from __future__ import annotations

import pytest

from tests._stub_finnhub import (
    StubFinnhub,
    make_financials_entry,
    make_metric,
    make_profile,
)
from wise_investor.config import settings
from wise_investor.data.finnhub import FinnhubClient
from wise_investor.tools.dcf import dcf_fair_value
from wise_investor.tools.verify import (
    list_supported_fields,
    verify_number,
)


# ---------------------------------------------------------------------------
# Direct read-path verification
# ---------------------------------------------------------------------------


def test_verify_matches_revenue_within_tolerance() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                ic={"revenue": 391_000_000_000.0},
            )
        ]
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
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                ic={"revenue": 391_000_000_000.0},
            )
        ]
    )
    r = verify_number(
        claim=400_000_000_000.0,
        field="revenue",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
        tolerance_pct=1.0,
    )
    assert r.matches is False
    assert r.diff_pct is not None and r.diff_pct > 1.0


def test_verify_respects_looser_tolerance() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                ic={"revenue": 391_000_000_000.0},
            )
        ]
    )
    r = verify_number(
        claim=400_000_000_000.0,
        field="revenue",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
        tolerance_pct=5.0,
    )
    assert r.matches is True


def test_verify_quote_price() -> None:
    stub = StubFinnhub(quote_price=180.5)
    r = verify_number(
        claim=180.5, field="price", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert "current" in r.source_citation


def test_verify_market_cap_within_default_tolerance() -> None:
    # Profile market cap in dollars; the helper converts to Finnhub's millions.
    stub = StubFinnhub(profile=make_profile(market_cap=2.8e12))
    r = verify_number(
        claim=2.81e12, field="market_cap", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert r.diff_pct is not None and r.diff_pct < 1.0


def test_verify_total_debt_sums_long_and_short() -> None:
    # Finnhub total_debt() sums long_term_debt + short_term_debt.
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                bs={
                    "long_term_debt": 80_000_000_000.0,
                    "short_term_debt": 20_000_000_000.0,
                },
            )
        ]
    )
    r = verify_number(
        claim=100_000_000_000.0,
        field="total_debt",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True


def test_verify_operating_cash_flow_field() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                cf={"operating_cash_flow": 120_000_000_000.0},
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


def test_verify_free_cash_flow_alias_derives_from_ocf_minus_capex() -> None:
    # Finnhub has no explicit FCF; derive_free_cash_flow returns ocf - |capex|.
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                cf={
                    "operating_cash_flow": 120_000_000_000.0,
                    "capital_expenditure": 20_000_000_000.0,
                },
            )
        ]
    )
    r = verify_number(
        claim=100_000_000_000.0,
        field="fcf",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert r.field == "free_cash_flow"


def test_verify_enterprise_value() -> None:
    stub = StubFinnhub(metric=make_metric(enterprise_value=2_800_000_000_000.0))
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
    stub = StubFinnhub(
        quote_price=180.0,
        financials=[
            make_financials_entry(
                "AAPL", end_date="2024-09-28", ic={"eps_diluted": 6.0}
            )
        ],
    )
    # Computed PER is 30.0; Bull claimed 30.1 (0.33% off, within 1% tolerance).
    r = verify_number(
        claim=30.1, field="per", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is True
    assert "calculate_per" in r.source_citation


def test_verify_pe_alias_to_per() -> None:
    stub = StubFinnhub(
        quote_price=180.0,
        financials=[
            make_financials_entry(
                "AAPL", end_date="2024-09-28", ic={"eps_diluted": 6.0}
            )
        ],
    )
    r = verify_number(
        claim=30.0, field="pe", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.field == "per"
    assert r.matches is True


def test_verify_ev_ebitda() -> None:
    # EV=2.6e12, EBITDA = OI 130B + D&A 0 ⇒ ratio = 20.0
    stub = StubFinnhub(
        metric=make_metric(enterprise_value=2.6e12),
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                ic={"operating_income": 130e9},
                cf={"depreciation_and_amortization": 0.0},
            )
        ],
    )
    r = verify_number(
        claim=20.0,
        field="ev_ebitda",
        symbol="AAPL",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.matches is True


def test_verify_implied_growth_via_reverse_dcf() -> None:
    # Round-trip: price-in g=10%, verify recovery of that g.
    fcf = 1_000_000_000.0
    market_cap = dcf_fair_value(fcf, 0.10, 0.10, 0.025, 10)
    stub = StubFinnhub(
        quote_price=100.0,
        profile=make_profile(market_cap=market_cap),
        financials=[
            make_financials_entry(
                "TEST",
                end_date="2024-12-31",
                cf={"operating_cash_flow": fcf, "capital_expenditure": 0.0},
            )
        ],
    )
    r = verify_number(
        claim=0.10,
        field="implied_growth_rate",
        symbol="TEST",
        client=stub,  # type: ignore[arg-type]
    )
    assert r.source_value is not None
    assert abs(r.source_value - 0.10) < 1e-3
    # 1% of 0.10 is 0.001, so the recovered value must agree very closely.
    assert r.matches is True


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_verify_raises_on_unsupported_field() -> None:
    with pytest.raises(ValueError, match="unsupported field"):
        verify_number(claim=1.0, field="unknown_metric", symbol="AAPL")


def test_verify_returns_unknown_when_source_missing() -> None:
    # No financials at all — revenue read returns None.
    stub = StubFinnhub(financials=[])
    r = verify_number(
        claim=100.0, field="revenue", symbol="AAPL", client=stub  # type: ignore[arg-type]
    )
    assert r.matches is None
    assert r.source_value is None
    assert any("unavailable" in w for w in r.warnings)


def test_verify_zero_source_value_uses_absolute_equality() -> None:
    # total_debt sums long+short. If both are zero, source_value == 0.0.
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "X",
                end_date="2024-09-28",
                bs={"long_term_debt": 0.0, "short_term_debt": 0.0},
            )
        ]
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
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        pytest.skip("FINNHUB_API_KEY not set")
    with FinnhubClient() as c:
        r = verify_number(claim=1.0, field="revenue", symbol="AAPL", client=c)
    assert r.source_value is not None
    # Apple's annual revenue is hundreds of billions.
    assert r.source_value > 1e11
    assert r.matches is False


@pytest.mark.network
def test_network_verify_aapl_per_self_consistent() -> None:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        pytest.skip("FINNHUB_API_KEY not set")
    from wise_investor.tools.valuation import calculate_per

    with FinnhubClient() as c:
        own = calculate_per("AAPL", client=c)
        if own.computed is None:
            pytest.skip("AAPL PER could not be computed this run")
        r = verify_number(
            claim=own.computed,
            field="per",
            symbol="AAPL",
            client=c,
            tolerance_pct=0.5,
        )
    assert r.matches is True
