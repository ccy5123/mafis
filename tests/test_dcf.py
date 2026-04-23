"""Tests for reverse_dcf — pure math + Finnhub-backed orchestration.

Port of the original StubFMP tests onto StubFinnhub. The pure-math half
(dcf_fair_value, solve_implied_growth, _bisect) is provider-agnostic and
unchanged; the orchestration half now exercises the Finnhub path:
market cap from /stock/profile2, FCF derived from OCF − |capex| in
/stock/financials-reported.
"""

from __future__ import annotations

import pytest

from tests._stub_finnhub import (
    StubFinnhub,
    make_financials_entry,
    make_profile,
)
from wise_investor.config import settings
from wise_investor.data.finnhub import FinnhubClient
from wise_investor.tools.dcf import (
    DEFAULT_DISCOUNT_RATE,
    DEFAULT_HIGH_GROWTH_YEARS,
    DEFAULT_TERMINAL_GROWTH,
    dcf_fair_value,
    reverse_dcf,
    solve_implied_growth,
)


# ---------------------------------------------------------------------------
# Pure math tests
# ---------------------------------------------------------------------------


def test_dcf_fair_value_monotonic_in_growth() -> None:
    fv_low = dcf_fair_value(100.0, 0.05, 0.10, 0.025, 10)
    fv_mid = dcf_fair_value(100.0, 0.10, 0.10, 0.025, 10)
    fv_high = dcf_fair_value(100.0, 0.20, 0.10, 0.025, 10)
    assert fv_low < fv_mid < fv_high


def test_dcf_fair_value_zero_growth_sanity() -> None:
    # With g=0, g_t=0, r=10%, long n: perpetuity PV ≈ FCF/r = 100/0.10 = 1000.
    fv = dcf_fair_value(100.0, 0.0, 0.10, 0.0, 50)
    assert 900 < fv < 1100


def test_solve_implied_growth_recovers_known_growth() -> None:
    """Round-trip: pick g*, compute fair value, solve reverse DCF → get g* back."""
    g_star = 0.08
    fcf = 5_000_000_000.0
    market_cap = dcf_fair_value(
        fcf,
        g_star,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_HIGH_GROWTH_YEARS,
    )
    g_hat = solve_implied_growth(market_cap=market_cap, fcf_0=fcf)
    assert g_hat is not None
    assert abs(g_hat - g_star) < 1e-3


def test_solve_implied_growth_zero_growth_case() -> None:
    g_star = 0.0
    fcf = 1_000_000_000.0
    market_cap = dcf_fair_value(
        fcf,
        g_star,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_HIGH_GROWTH_YEARS,
    )
    g_hat = solve_implied_growth(market_cap=market_cap, fcf_0=fcf)
    assert g_hat is not None
    assert abs(g_hat) < 1e-3


def test_solve_implied_growth_rejects_discount_le_terminal() -> None:
    with pytest.raises(ValueError, match="discount_rate"):
        solve_implied_growth(
            market_cap=1e9, fcf_0=1e8, discount_rate=0.02, terminal_growth=0.025
        )


def test_solve_implied_growth_rejects_negative_fcf() -> None:
    with pytest.raises(ValueError, match="fcf_0"):
        solve_implied_growth(market_cap=1e9, fcf_0=-1e8)


def test_solve_implied_growth_returns_none_outside_bracket() -> None:
    # Market cap so high even 100%/yr for 10y can't justify it.
    g_hat = solve_implied_growth(market_cap=1e20, fcf_0=1.0)
    assert g_hat is None


# ---------------------------------------------------------------------------
# reverse_dcf orchestration (Finnhub-backed)
# ---------------------------------------------------------------------------


def _build_stub_for_dcf(
    *, symbol: str, market_cap: float, ocf: float, capex: float, end_date: str
) -> StubFinnhub:
    """Helper: compose a stub where FCF derives to `ocf - |capex|`."""
    return StubFinnhub(
        quote_price=100.0,
        profile=make_profile(market_cap=market_cap),
        financials=[
            make_financials_entry(
                symbol,
                end_date=end_date,
                cf={"operating_cash_flow": ocf, "capital_expenditure": capex},
            )
        ],
    )


def test_reverse_dcf_happy_path() -> None:
    # Build a scenario where implied growth should be exactly 10%.
    fcf = 10_000_000_000.0
    target_g = 0.10
    market_cap = dcf_fair_value(
        fcf,
        target_g,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_HIGH_GROWTH_YEARS,
    )
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=market_cap,
        ocf=fcf,  # capex=0 ⇒ FCF == ocf
        capex=0.0,
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is not None
    assert abs(r.implied_growth_rate - target_g) < 1e-3
    assert r.current_market_cap == market_cap
    assert r.inputs["fcf_latest_annual"] == fcf
    assert "derived" in r.inputs["fcf_source"]
    assert r.as_of == "2024-12-31"


def test_reverse_dcf_derives_fcf_from_ocf_minus_capex() -> None:
    # Same target growth as happy path but split across ocf + capex.
    fcf = 5_000_000_000.0
    target_g = 0.05
    market_cap = dcf_fair_value(
        fcf,
        target_g,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_HIGH_GROWTH_YEARS,
    )
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=market_cap,
        ocf=7_000_000_000.0,
        capex=2_000_000_000.0,  # Finnhub XBRL: positive magnitude
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is not None
    assert r.inputs["fcf_latest_annual"] == 5_000_000_000.0
    assert "derived" in r.inputs["fcf_source"]


def test_reverse_dcf_warns_on_high_implied_growth() -> None:
    fcf = 1_000_000_000.0
    target_g = 0.40  # extreme
    market_cap = dcf_fair_value(
        fcf,
        target_g,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_HIGH_GROWTH_YEARS,
    )
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=market_cap,
        ocf=fcf,
        capex=0.0,
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is not None
    assert r.implied_growth_rate > 0.25
    assert any("unusually high" in w for w in r.warnings)


def test_reverse_dcf_warns_on_negative_implied_growth() -> None:
    fcf = 1_000_000_000.0
    market_cap = dcf_fair_value(
        fcf, -0.10, DEFAULT_DISCOUNT_RATE, DEFAULT_TERMINAL_GROWTH, DEFAULT_HIGH_GROWTH_YEARS
    )
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=market_cap,
        ocf=fcf,
        capex=0.0,
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is not None
    assert r.implied_growth_rate < 0
    assert any("negative" in w for w in r.warnings)


def test_reverse_dcf_returns_none_on_negative_fcf() -> None:
    # OCF < capex ⇒ derived FCF negative.
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=1e9,
        ocf=1e8,
        capex=6e8,  # abs(capex) > ocf ⇒ fcf = -5e8
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is None
    assert any("FCF <= 0" in w for w in r.warnings)


def test_reverse_dcf_returns_none_when_fcf_components_missing() -> None:
    # Filing exists but ic/bs/cf all empty → derive_free_cash_flow returns None.
    stub = StubFinnhub(
        quote_price=100.0,
        profile=make_profile(market_cap=1e9),
        financials=[make_financials_entry("TEST", end_date="2024-12-31")],
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is None
    assert any("cannot derive" in w.lower() for w in r.warnings)


def test_reverse_dcf_returns_none_on_missing_market_cap() -> None:
    # Profile exists but has no market cap.
    stub = StubFinnhub(
        quote_price=100.0,
        profile=make_profile(market_cap=None),
        financials=[
            make_financials_entry(
                "TEST",
                end_date="2024-12-31",
                cf={"operating_cash_flow": 1e9, "capital_expenditure": 0.0},
            )
        ],
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is None
    assert any("market cap" in w.lower() for w in r.warnings)


def test_reverse_dcf_returns_none_when_no_annual_financials() -> None:
    stub = StubFinnhub(
        quote_price=100.0,
        profile=make_profile(market_cap=1e9),
        financials=[],
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.implied_growth_rate is None
    assert any("no annual financials" in w for w in r.warnings)


def test_reverse_dcf_records_inputs_for_report_citation() -> None:
    fcf = 1_000_000_000.0
    market_cap = dcf_fair_value(
        fcf, 0.08, DEFAULT_DISCOUNT_RATE, DEFAULT_TERMINAL_GROWTH, DEFAULT_HIGH_GROWTH_YEARS
    )
    stub = _build_stub_for_dcf(
        symbol="TEST",
        market_cap=market_cap,
        ocf=fcf,
        capex=0.0,
        end_date="2024-12-31",
    )
    r = reverse_dcf("TEST", client=stub)  # type: ignore[arg-type]
    assert r.inputs["discount_rate"] == DEFAULT_DISCOUNT_RATE
    assert r.inputs["terminal_growth"] == DEFAULT_TERMINAL_GROWTH
    assert r.inputs["high_growth_years"] == DEFAULT_HIGH_GROWTH_YEARS
    assert r.inputs["fcf_latest_annual"] == fcf
    assert r.inputs["market_cap"] == market_cap


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_network_reverse_dcf_aapl_runs_end_to_end() -> None:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        pytest.skip("FINNHUB_API_KEY not set")
    with FinnhubClient() as c:
        r = reverse_dcf("AAPL", client=c)
    if r.implied_growth_rate is None:
        assert r.warnings, "null result must carry a warning"
    else:
        assert -0.10 <= r.implied_growth_rate <= 0.40, (
            f"implied growth {r.implied_growth_rate} outside sanity band"
        )
    assert r.inputs["discount_rate"] == DEFAULT_DISCOUNT_RATE
