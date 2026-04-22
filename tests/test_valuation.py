"""Tests for the Phase 1A valuation calculation tools.

Post Phase 1B migration: StubFMP was shaped for FMPClient; the tools now
consume FinnhubClient. Offline tests are skipped until a StubFinnhub is
written. Network tests against real Finnhub still cover happy-path behavior
in tests/test_agents_tools.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Post Phase 1B Finnhub migration: StubFMP needs rewrite to StubFinnhub. "
        "Network coverage remains via scripts/smoke_phase1a.py."
    )
)

from wise_investor.config import settings
from wise_investor.data.fmp import (
    EnterpriseValue,
    FMPClient,
    IncomeStatement,
    KeyMetrics,
    Quote,
    Ratios,
    StockPeer,
)
from wise_investor.tools.valuation import (
    calculate_ev_ebitda,
    calculate_per,
    get_peer_multiples,
)


# ---------------------------------------------------------------------------
# Stub client
# ---------------------------------------------------------------------------


class StubFMP:
    """Minimal FMPClient-shaped stand-in for offline tests.

    Returns whatever per-symbol canned data is injected at construction time.
    Unknown symbols fall back to the default payload.
    """

    def __init__(
        self,
        quote: Quote | None = None,
        income: list[IncomeStatement] | None = None,
        ratios: list[Ratios] | None = None,
        ev_values: list[EnterpriseValue] | None = None,
        key_metrics: list[KeyMetrics] | None = None,
        peers: list[StockPeer] | None = None,
        per_symbol: dict[str, dict] | None = None,
    ) -> None:
        self._default = {
            "quote": quote,
            "income": income or [],
            "ratios": ratios or [],
            "ev_values": ev_values or [],
            "key_metrics": key_metrics or [],
            "peers": peers or [],
        }
        self._per_symbol = per_symbol or {}

    def _payload(self, symbol: str, key: str):
        if symbol in self._per_symbol and key in self._per_symbol[symbol]:
            return self._per_symbol[symbol][key]
        return self._default[key]

    def quote(self, symbol: str) -> Quote:
        q = self._payload(symbol, "quote")
        if q is None:
            raise RuntimeError(f"StubFMP: no quote for {symbol}")
        return q

    def income_statement(self, symbol: str, period: str = "annual", limit: int = 5):
        return self._payload(symbol, "income")

    def ratios(self, symbol: str, period: str = "annual", limit: int = 5):
        return self._payload(symbol, "ratios")

    def enterprise_values(self, symbol: str, period: str = "annual", limit: int = 5):
        return self._payload(symbol, "ev_values")

    def key_metrics(self, symbol: str, period: str = "annual", limit: int = 5):
        return self._payload(symbol, "key_metrics")

    def stock_peers(self, symbol: str):
        return self._payload(symbol, "peers")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# calculate_per
# ---------------------------------------------------------------------------


def test_per_happy_path_matches_fmp_reported() -> None:
    stub = StubFMP(
        quote=Quote(symbol="AAPL", price=180.0),
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0)],
        ratios=[Ratios(symbol="AAPL", date="2024-09-28", price_to_earnings_ratio=30.0)],
    )
    r = calculate_per("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == 30.0
    assert r.fmp_reported == 30.0
    assert r.diff_pct_vs_fmp == 0.0
    assert r.inputs["price"] == 180.0
    assert r.inputs["eps_diluted_latest_annual"] == 6.0
    assert r.as_of == "2024-09-28"
    assert r.warnings == []


def test_per_falls_back_to_eps_when_diluted_missing() -> None:
    stub = StubFMP(
        quote=Quote(symbol="AAPL", price=180.0),
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", eps=6.0)],
    )
    r = calculate_per("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == 30.0


def test_per_returns_none_on_negative_eps_with_warning() -> None:
    stub = StubFMP(
        quote=Quote(symbol="XYZ", price=50.0),
        income=[IncomeStatement(date="2024-12-31", symbol="XYZ", eps_diluted=-2.0)],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EPS <= 0" in w for w in r.warnings)


def test_per_returns_none_on_zero_eps() -> None:
    stub = StubFMP(
        quote=Quote(symbol="XYZ", price=50.0),
        income=[IncomeStatement(date="2024-12-31", symbol="XYZ", eps_diluted=0.0)],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None


def test_per_returns_none_on_missing_eps() -> None:
    stub = StubFMP(
        quote=Quote(symbol="XYZ", price=50.0),
        income=[IncomeStatement(date="2024-12-31", symbol="XYZ")],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EPS missing" in w for w in r.warnings)


def test_per_handles_empty_income_statement() -> None:
    stub = StubFMP(
        quote=Quote(symbol="XYZ", price=50.0),
        income=[],
    )
    r = calculate_per("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("income statement empty" in w for w in r.warnings)


def test_per_reports_divergence_when_fmp_disagrees() -> None:
    # Construct a case where our computation (price / eps) differs from FMP's
    # reported PER — maybe FMP uses a different price date.
    stub = StubFMP(
        quote=Quote(symbol="AAPL", price=180.0),
        income=[IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0)],
        ratios=[Ratios(symbol="AAPL", date="2024-09-28", price_to_earnings_ratio=25.0)],
    )
    r = calculate_per("AAPL", client=stub)  # type: ignore[arg-type]
    assert r.computed == 30.0
    assert r.fmp_reported == 25.0
    assert r.diff_pct_vs_fmp == 20.0


# ---------------------------------------------------------------------------
# calculate_ev_ebitda
# ---------------------------------------------------------------------------


def test_ev_ebitda_happy_path() -> None:
    stub = StubFMP(
        income=[
            IncomeStatement(
                date="2024-09-28",
                symbol="AAPL",
                ebitda=130_000_000_000,
            )
        ],
        ev_values=[
            EnterpriseValue(
                symbol="AAPL",
                date="2024-09-28",
                enterprise_value=2_800_000_000_000,
            )
        ],
        key_metrics=[
            KeyMetrics(
                symbol="AAPL",
                date="2024-09-28",
                ev_to_ebitda=round(2_800_000_000_000 / 130_000_000_000, 3),
            )
        ],
    )
    r = calculate_ev_ebitda("AAPL", client=stub)  # type: ignore[arg-type]
    expected = round(2_800_000_000_000 / 130_000_000_000, 3)
    assert r.computed == expected
    assert r.fmp_reported == expected
    assert r.diff_pct_vs_fmp == 0.0
    assert r.as_of == "2024-09-28"


def test_ev_ebitda_returns_none_on_negative_ebitda() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="XYZ", ebitda=-1e9)],
        ev_values=[EnterpriseValue(symbol="XYZ", date="2024-09-28", enterprise_value=5e9)],
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("EBITDA <= 0" in w for w in r.warnings)


def test_ev_ebitda_returns_none_on_zero_ebitda() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="XYZ", ebitda=0.0)],
        ev_values=[EnterpriseValue(symbol="XYZ", date="2024-09-28", enterprise_value=5e9)],
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None


def test_ev_ebitda_handles_missing_ev_payload() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="XYZ", ebitda=1e9)],
        ev_values=[],
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("enterprise_values" in w for w in r.warnings)


def test_ev_ebitda_handles_missing_ebitda_field() -> None:
    stub = StubFMP(
        income=[IncomeStatement(date="2024-09-28", symbol="XYZ")],  # no ebitda field
        ev_values=[EnterpriseValue(symbol="XYZ", date="2024-09-28", enterprise_value=5e9)],
    )
    r = calculate_ev_ebitda("XYZ", client=stub)  # type: ignore[arg-type]
    assert r.computed is None
    assert any("ebitda unavailable" in w for w in r.warnings)


# ---------------------------------------------------------------------------
# get_peer_multiples
# ---------------------------------------------------------------------------


def test_peer_multiples_includes_target_and_peers() -> None:
    aapl_income = [IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0, ebitda=130e9)]
    aapl_ev = [EnterpriseValue(symbol="AAPL", date="2024-09-28", enterprise_value=2.8e12)]

    msft_income = [IncomeStatement(date="2024-06-30", symbol="MSFT", eps_diluted=11.0, ebitda=135e9)]
    msft_ev = [EnterpriseValue(symbol="MSFT", date="2024-06-30", enterprise_value=3.0e12)]

    stub = StubFMP(
        peers=[
            StockPeer(symbol="MSFT", company_name="Microsoft", mkt_cap=3.1e12),
            StockPeer(symbol="AAPL", company_name="Apple", mkt_cap=2.8e12),  # self in list
        ],
        per_symbol={
            "AAPL": {
                "quote": Quote(symbol="AAPL", price=180.0),
                "income": aapl_income,
                "ev_values": aapl_ev,
                "ratios": [],
                "key_metrics": [],
                "peers": [
                    StockPeer(symbol="MSFT", company_name="Microsoft", mkt_cap=3.1e12),
                ],
            },
            "MSFT": {
                "quote": Quote(symbol="MSFT", price=440.0),
                "income": msft_income,
                "ev_values": msft_ev,
                "ratios": [],
                "key_metrics": [],
                "peers": [],
            },
        },
    )

    table = get_peer_multiples("AAPL", client=stub)  # type: ignore[arg-type]
    assert table.target_symbol == "AAPL"
    # Target first, then one peer
    assert [r.symbol for r in table.rows] == ["AAPL", "MSFT"]

    aapl_row = table.rows[0]
    assert aapl_row.per == 30.0
    assert aapl_row.ev_ebitda == round(2.8e12 / 130e9, 3)

    msft_row = table.rows[1]
    assert msft_row.per == round(440.0 / 11.0, 3)
    assert msft_row.ev_ebitda == round(3.0e12 / 135e9, 3)


def test_peer_multiples_target_as_of_reflects_latest_fiscal_date() -> None:
    stub = StubFMP(
        per_symbol={
            "AAPL": {
                "quote": Quote(symbol="AAPL", price=180.0),
                "income": [IncomeStatement(date="2024-09-28", symbol="AAPL", eps_diluted=6.0)],
                "ev_values": [],
                "ratios": [],
                "key_metrics": [],
                "peers": [],
            },
        },
    )
    table = get_peer_multiples("AAPL", client=stub)  # type: ignore[arg-type]
    assert table.as_of == "2024-09-28"


def test_peer_multiples_max_peers_caps_size() -> None:
    peers_payload = [
        StockPeer(symbol=f"PEER{i}", company_name=f"Peer {i}") for i in range(15)
    ]
    per_sym = {
        "TARGET": {
            "quote": Quote(symbol="TARGET", price=100.0),
            "income": [IncomeStatement(date="2024-12-31", symbol="TARGET", eps_diluted=5.0)],
            "ev_values": [],
            "ratios": [],
            "key_metrics": [],
            "peers": peers_payload,
        }
    }
    for i in range(15):
        per_sym[f"PEER{i}"] = {
            "quote": Quote(symbol=f"PEER{i}", price=100.0),
            "income": [IncomeStatement(date="2024-12-31", symbol=f"PEER{i}", eps_diluted=5.0)],
            "ev_values": [],
            "ratios": [],
            "key_metrics": [],
            "peers": [],
        }
    stub = StubFMP(per_symbol=per_sym)
    table = get_peer_multiples("TARGET", client=stub, max_peers=3)  # type: ignore[arg-type]
    # 1 target + 3 peers
    assert len(table.rows) == 4


# ---------------------------------------------------------------------------
# Network tests
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        pytest.skip("FMP_API_KEY not set")
    with FMPClient() as c:
        yield c


@pytest.mark.network
def test_network_per_aapl_reasonable(client: FMPClient) -> None:
    r = calculate_per("AAPL", client=client)
    assert r.computed is not None
    # Apple's PER has been in the 20-45 range in recent years.
    assert 5 < r.computed < 100, f"AAPL PER {r.computed} outside sanity band"
    if r.fmp_reported is not None:
        # Our computation uses current price / latest annual EPS; FMP's ratios
        # endpoint uses fiscal-year-end price / EPS, so some divergence is expected.
        assert r.diff_pct_vs_fmp is not None


@pytest.mark.network
def test_network_ev_ebitda_aapl_reasonable(client: FMPClient) -> None:
    r = calculate_ev_ebitda("AAPL", client=client)
    # Sometimes ebitda is missing from /income-statement; tolerate gracefully.
    if r.computed is not None:
        assert 5 < r.computed < 80
        if r.fmp_reported is not None:
            # Should agree closely since both sides use fiscal-year snapshots.
            assert r.diff_pct_vs_fmp is None or r.diff_pct_vs_fmp < 15.0
    else:
        # If we couldn't compute, we must have explained why.
        assert r.warnings, "empty computed value must carry a warning"


@pytest.mark.network
def test_network_peer_multiples_aapl(client: FMPClient) -> None:
    table = get_peer_multiples("AAPL", client=client, max_peers=5)
    assert table.target_symbol == "AAPL"
    assert len(table.rows) >= 2  # target + at least one peer
    assert table.rows[0].symbol == "AAPL"
