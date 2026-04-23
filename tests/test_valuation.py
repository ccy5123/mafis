"""Tests for the Phase 1A valuation calculation tools (Finnhub-backed).

Port of the original StubFMP tests onto StubFinnhub after the Phase 1B
migration. Covers calculate_per, calculate_ev_ebitda, and
get_peer_multiples against deterministic in-memory stubs. Live network
coverage still runs via `pytest -m network`.
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
from wise_investor.tools.valuation import (
    calculate_ev_ebitda,
    calculate_per,
    get_peer_multiples,
)


# ---------------------------------------------------------------------------
# calculate_per
# ---------------------------------------------------------------------------


def test_per_happy_path_matches_finnhub_reported() -> None:
    stub = StubFinnhub(
        quote_price=180.0,
        financials=[
            make_financials_entry(
                "AAPL", end_date="2024-09-28", ic={"eps_diluted": 6.0}
            )
        ],
        metric=make_metric(pe_annual=30.0),
    )
    r = calculate_per("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == 30.0
    assert r.fmp_reported == 30.0
    assert r.diff_pct_vs_fmp == 0.0
    assert r.inputs["price"] == 180.0
    assert r.inputs["eps_diluted_latest_annual"] == 6.0
    assert r.as_of == "2024-09-28"
    assert r.warnings == []


def test_per_returns_none_on_negative_eps_with_warning() -> None:
    stub = StubFinnhub(
        quote_price=50.0,
        financials=[
            make_financials_entry(
                "XYZ", end_date="2024-12-31", ic={"eps_diluted": -2.0}
            )
        ],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EPS <= 0" in w for w in r.warnings)


def test_per_returns_none_on_zero_eps() -> None:
    stub = StubFinnhub(
        quote_price=50.0,
        financials=[
            make_financials_entry(
                "XYZ", end_date="2024-12-31", ic={"eps_diluted": 0.0}
            )
        ],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None


def test_per_returns_none_when_eps_concept_missing() -> None:
    # Filing entry exists but contains no EPS concept — extract_field returns None.
    stub = StubFinnhub(
        quote_price=50.0,
        financials=[make_financials_entry("XYZ", end_date="2024-12-31", ic={})],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EPS diluted not found" in w for w in r.warnings)


def test_per_returns_none_when_no_annual_financials() -> None:
    # No filings at all — latest_annual_financials() returns None.
    stub = StubFinnhub(quote_price=50.0, financials=[])
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("no annual financials" in w for w in r.warnings)


def test_per_reports_divergence_when_finnhub_disagrees() -> None:
    # Our computation uses current price / latest annual EPS; Finnhub's
    # peAnnual uses fiscal-year-end price, so divergence is realistic.
    stub = StubFinnhub(
        quote_price=180.0,
        financials=[
            make_financials_entry(
                "AAPL", end_date="2024-09-28", ic={"eps_diluted": 6.0}
            )
        ],
        metric=make_metric(pe_annual=25.0),
    )
    r = calculate_per("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == 30.0
    assert r.fmp_reported == 25.0
    assert r.diff_pct_vs_fmp == 20.0


# ---------------------------------------------------------------------------
# calculate_ev_ebitda
# ---------------------------------------------------------------------------


def test_ev_ebitda_happy_path() -> None:
    # EBITDA is derived as operating_income + D&A. Split 110B + 20B = 130B.
    ev = 2_800_000_000_000.0
    ebitda = 130_000_000_000.0
    expected = round(ev / ebitda, 3)
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "AAPL",
                end_date="2024-09-28",
                ic={"operating_income": 110_000_000_000.0},
                cf={"depreciation_and_amortization": 20_000_000_000.0},
            )
        ],
        metric=make_metric(enterprise_value=ev, ev_ebitda_ttm=expected),
    )
    r = calculate_ev_ebitda("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == expected
    assert r.fmp_reported == expected
    assert r.diff_pct_vs_fmp == 0.0
    assert r.as_of == "2024-09-28"


def test_ev_ebitda_returns_none_on_negative_ebitda() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "XYZ",
                end_date="2024-09-28",
                ic={"operating_income": -1e9},
                cf={"depreciation_and_amortization": 0.0},
            )
        ],
        metric=make_metric(enterprise_value=5e9),
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EBITDA <= 0" in w for w in r.warnings)


def test_ev_ebitda_returns_none_on_zero_ebitda() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "XYZ",
                end_date="2024-09-28",
                ic={"operating_income": 0.0},
                cf={"depreciation_and_amortization": 0.0},
            )
        ],
        metric=make_metric(enterprise_value=5e9),
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None


def test_ev_ebitda_handles_missing_enterprise_value() -> None:
    stub = StubFinnhub(
        financials=[
            make_financials_entry(
                "XYZ",
                end_date="2024-09-28",
                ic={"operating_income": 1e9},
                cf={"depreciation_and_amortization": 0.0},
            )
        ],
        metric=make_metric(enterprise_value=None),
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("enterprise_value" in w for w in r.warnings)


def test_ev_ebitda_handles_missing_ebitda_components() -> None:
    # operating_income concept not in the filing → derive_ebitda returns None.
    stub = StubFinnhub(
        financials=[make_financials_entry("XYZ", end_date="2024-09-28", ic={}, cf={})],
        metric=make_metric(enterprise_value=5e9),
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("ebitda" in w.lower() for w in r.warnings)


# ---------------------------------------------------------------------------
# get_peer_multiples
# ---------------------------------------------------------------------------


def test_peer_multiples_includes_target_and_peers() -> None:
    stub = StubFinnhub(
        peers=["MSFT"],
        per_symbol={
            "AAPL": {
                "profile": make_profile(market_cap=2.8e12, name="Apple", ticker="AAPL"),
                "metric": make_metric(pe_ttm=30.0, ev_ebitda_ttm=21.5),
                "financials": [
                    make_financials_entry("AAPL", end_date="2024-09-28")
                ],
                "peers": ["MSFT"],
            },
            "MSFT": {
                "profile": make_profile(
                    market_cap=3.1e12, name="Microsoft", ticker="MSFT"
                ),
                "metric": make_metric(pe_ttm=40.0, ev_ebitda_ttm=22.2),
                "financials": [],
            },
        },
    )
    table = get_peer_multiples("AAPL", client=stub)  # type: ignore[arg-type]
    assert table.target_symbol == "AAPL"
    assert [r.symbol for r in table.rows] == ["AAPL", "MSFT"]

    aapl_row = table.rows[0]
    assert aapl_row.name == "Apple"
    assert aapl_row.per == 30.0
    assert aapl_row.ev_ebitda == 21.5

    msft_row = table.rows[1]
    assert msft_row.name == "Microsoft"
    assert msft_row.per == 40.0
    assert msft_row.ev_ebitda == 22.2


def test_peer_multiples_target_as_of_reflects_latest_fiscal_date() -> None:
    stub = StubFinnhub(
        per_symbol={
            "AAPL": {
                "profile": make_profile(market_cap=2.8e12),
                "metric": make_metric(pe_ttm=30.0, ev_ebitda_ttm=21.0),
                "financials": [
                    make_financials_entry("AAPL", end_date="2024-09-28")
                ],
                "peers": [],
            },
        },
    )
    table = get_peer_multiples("AAPL", client=stub)  # type: ignore[arg-type]
    assert table.as_of == "2024-09-28"


def test_peer_multiples_max_peers_caps_size() -> None:
    peer_symbols = [f"PEER{i}" for i in range(15)]
    per_sym: dict[str, dict] = {
        "TARGET": {
            "profile": make_profile(market_cap=1e11),
            "metric": make_metric(pe_ttm=20.0, ev_ebitda_ttm=12.0),
            "financials": [make_financials_entry("TARGET", end_date="2024-12-31")],
            "peers": peer_symbols,
        }
    }
    for i, p in enumerate(peer_symbols):
        per_sym[p] = {
            "profile": make_profile(market_cap=5e10),
            "metric": make_metric(pe_ttm=20.0 + i * 0.1, ev_ebitda_ttm=12.0),
            "financials": [],
        }
    stub = StubFinnhub(per_symbol=per_sym)
    table = get_peer_multiples("TARGET", client=stub, max_peers=3)  # type: ignore[arg-type]
    # 1 target + 3 peers
    assert len(table.rows) == 4


def test_peer_multiples_sanity_filter_excludes_garbage_multiples() -> None:
    # One peer has PER=1000 (exceeds SANITY_MAX_PER=300) — must land in excluded.
    stub = StubFinnhub(
        per_symbol={
            "NVDA": {
                "profile": make_profile(market_cap=3e12),
                "metric": make_metric(pe_ttm=40.0, ev_ebitda_ttm=30.0),
                "financials": [make_financials_entry("NVDA", end_date="2024-01-28")],
                "peers": ["JUNK"],
            },
            "JUNK": {
                "profile": make_profile(market_cap=1e10),
                "metric": make_metric(pe_ttm=1000.0, ev_ebitda_ttm=50.0),
                "financials": [],
            },
        }
    )
    table = get_peer_multiples("NVDA", client=stub)  # type: ignore[arg-type]
    kept_symbols = [r.symbol for r in table.rows]
    excluded_symbols = [r.symbol for r in table.excluded]
    assert "NVDA" in kept_symbols
    assert "JUNK" in excluded_symbols
    assert "JUNK" not in kept_symbols


def test_peer_multiples_additional_peers_override_is_merged() -> None:
    stub = StubFinnhub(
        per_symbol={
            "GEV": {
                "profile": make_profile(market_cap=1e11),
                "metric": make_metric(pe_ttm=45.0, ev_ebitda_ttm=25.0),
                "financials": [make_financials_entry("GEV", end_date="2024-12-31")],
                "peers": [],  # Finnhub returns nothing useful for a spin-off
            },
            "SMNEY": {
                "profile": make_profile(market_cap=1.2e11),
                "metric": make_metric(pe_ttm=22.0, ev_ebitda_ttm=14.0),
                "financials": [],
            },
        }
    )
    table = get_peer_multiples(
        "GEV", client=stub, additional_peers=["SMNEY"]  # type: ignore[arg-type]
    )
    symbols = [r.symbol for r in table.rows]
    assert "SMNEY" in symbols
    assert "SMNEY" in table.override_sources


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        pytest.skip("FINNHUB_API_KEY not set")
    with FinnhubClient() as c:
        yield c


@pytest.mark.network
def test_network_per_aapl_reasonable(client: FinnhubClient) -> None:
    r = calculate_per("AAPL", client=client)
    assert r.computed is not None
    # Apple's PER has been in the 20-45 range in recent years.
    assert 5 < r.computed < 100, f"AAPL PER {r.computed} outside sanity band"


@pytest.mark.network
def test_network_ev_ebitda_aapl_reasonable(client: FinnhubClient) -> None:
    r = calculate_ev_ebitda("AAPL", client=client)
    if r.computed is not None:
        assert 5 < r.computed < 80
    else:
        # If we couldn't compute, we must have explained why.
        assert r.warnings, "empty computed value must carry a warning"


@pytest.mark.network
def test_network_peer_multiples_aapl(client: FinnhubClient) -> None:
    table = get_peer_multiples("AAPL", client=client, max_peers=5)
    assert table.target_symbol == "AAPL"
    assert len(table.rows) >= 2  # target + at least one peer
    assert table.rows[0].symbol == "AAPL"
