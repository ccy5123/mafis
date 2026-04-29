"""Peer aggregator tests — Finnhub client fully stubbed.

Verifies:
  - Peer list trimming (self-exclude, dedup, limit cap)
  - Per-peer ROIC computation (NOPAT/IC, recent 3y avg)
  - Median across peers, std across pooled GM observations
  - Cache write + read round-trip with TTL handling
  - Graceful degradation on empty / failing peer responses
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from wise_investor.screening.peer_aggregator import (
    DEFAULT_PEER_LIMIT,
    PeerAggregateResult,
    compute_industry_aggregates,
)

# ---------------------------------------------------------------------------
# Stubs (reuse the same Finnhub-shape seen in test_live_adapter.py)
# ---------------------------------------------------------------------------


class _Item:
    def __init__(self, concept: str, value: float | None) -> None:
        self.concept = concept
        self.value = value


class _Report:
    def __init__(self, ic=(), bs=(), cf=()) -> None:
        self.ic = list(ic)
        self.bs = list(bs)
        self.cf = list(cf)


class _Entry:
    def __init__(self, year, *, quarter=None, form="10-K", report=None) -> None:
        self.year = year
        self.quarter = quarter
        self.form = form
        self.report = report or _Report()


class _Resp:
    def __init__(self, data) -> None:
        self.data = data


class _StubClient:
    """Configurable peer-capable client.

    `peer_list` controls the response of `peers()`.
    `financials_by_symbol` keys are the symbol passed to financials().
    """

    def __init__(
        self,
        *,
        peer_list: list[str] | None = None,
        financials_by_symbol: dict[str, list[_Entry]] | None = None,
        default_financials: list[_Entry] | None = None,
    ) -> None:
        self._peer_list = list(peer_list or [])
        self._financials_by_symbol = financials_by_symbol or {}
        self._default = default_financials or []
        self.peers_calls: list[str] = []
        self.financials_calls: list[tuple[str, str]] = []

    def peers(self, symbol: str) -> list[str]:
        self.peers_calls.append(symbol)
        return list(self._peer_list)

    def financials(self, symbol: str, freq: str = "annual"):
        self.financials_calls.append((symbol, freq))
        return _Resp(
            self._financials_by_symbol.get(symbol, self._default)
        )


def _annual(
    year, *, revenue, gross, operating, debt, equity, cash,
    total_assets: float | None = None,
) -> _Entry:
    """Build a Finnhub-shape annual entry with the IC/BS line items.

    `total_assets` defaults to debt + equity (clean BS identity for
    fixtures that don't track operating liabilities). Tests that need
    a specific total-assets shape override.
    """
    if total_assets is None:
        total_assets = (debt or 0.0) + (equity or 0.0)
    ic = [
        _Item("us-gaap_Revenues", revenue),
        _Item("us-gaap_GrossProfit", gross),
        _Item("us-gaap_OperatingIncomeLoss", operating),
    ]
    bs = [
        _Item("us-gaap_LongTermDebt", debt),
        _Item("us-gaap_StockholdersEquity", equity),
        _Item("us-gaap_CashAndCashEquivalentsAtCarryingValue", cash),
        _Item("us-gaap_Assets", total_assets),
    ]
    return _Entry(year=year, form="10-K", report=_Report(ic=ic, bs=bs))


def _profitable_peer(years: list[int]) -> list[_Entry]:
    """A peer with stable ~10% ROIC and ~50% gross margin."""
    return [
        _annual(y, revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50)
        for y in years
    ]


def _high_roic_peer(years: list[int]) -> list[_Entry]:
    """A peer with ~20% ROIC and ~60% GM."""
    return [
        _annual(y, revenue=1000, gross=600, operating=260, debt=50, equity=900, cash=50)
        for y in years
    ]


# ---------------------------------------------------------------------------
# Peer list handling
# ---------------------------------------------------------------------------


def test_self_reference_excluded(tmp_path: Path) -> None:
    """Finnhub sometimes self-includes the query symbol in the peers list."""
    client = _StubClient(
        peer_list=["NVDA", "AMD", "INTC"],  # NVDA is self when querying NVDA
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    fetched = [c[0] for c in client.financials_calls]
    assert "NVDA" not in fetched
    assert "AMD" in fetched
    assert "INTC" in fetched


def test_peer_limit_caps_api_calls(tmp_path: Path) -> None:
    client = _StubClient(
        peer_list=["AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "ON"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, peer_limit=3, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_attempted == 3
    assert len(client.financials_calls) == 3


def test_default_peer_limit() -> None:
    assert DEFAULT_PEER_LIMIT == 5


def test_dedupe_preserves_first_occurrence(tmp_path: Path) -> None:
    client = _StubClient(
        peer_list=["AMD", "INTC", "AMD", "INTC"],  # duplicates
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    fetched = [c[0] for c in client.financials_calls]
    assert fetched == ["AMD", "INTC"]


def test_empty_peer_list_yields_empty_aggregates(tmp_path: Path) -> None:
    client = _StubClient(peer_list=[])
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_attempted == 0
    assert result.industry_aggregates.industry_roic_3y_median is None
    assert result.industry_aggregates.industry_gross_margin_3y_std is None


def test_peers_call_failure_yields_empty_aggregates(tmp_path: Path) -> None:
    class _PeersErrorClient(_StubClient):
        def peers(self, symbol):
            raise RuntimeError("Finnhub 500")

    client = _PeersErrorClient()
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_attempted == 0
    assert result.industry_aggregates.industry_roic_3y_median is None


# ---------------------------------------------------------------------------
# Per-peer aggregation math
# ---------------------------------------------------------------------------


def test_median_roic_across_peers(tmp_path: Path) -> None:
    """Three peers with ROICs ~10%, ~10%, ~20% → median = 10%."""
    client = _StubClient(
        peer_list=["A", "B", "C"],
        financials_by_symbol={
            "A": _profitable_peer([2022, 2023, 2024]),  # ~10% ROIC
            "B": _profitable_peer([2022, 2023, 2024]),
            "C": _high_roic_peer([2022, 2023, 2024]),   # ~20% ROIC
        },
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    median = result.industry_aggregates.industry_roic_3y_median
    assert median is not None
    # Profitable peer ROIC: NOPAT = 130*0.79 = 102.7; IC = 50+900-50 = 900;
    # ROIC ≈ 0.114
    assert 0.10 < median < 0.13


def test_pooled_gm_std_uses_all_peer_years(tmp_path: Path) -> None:
    """5 GM observations of identical 0.50 → std = 0."""
    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={
            "A": _profitable_peer([2020, 2021, 2022]),  # GM constant 0.50
        },
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    gm_std = result.industry_aggregates.industry_gross_margin_3y_std
    assert gm_std == pytest.approx(0.0, abs=1e-9)


def test_gm_std_with_variation(tmp_path: Path) -> None:
    def variable_gm_peer(years):
        # Each year has slightly different GM
        return [
            _annual(2020, revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50),  # 0.50
            _annual(2021, revenue=1000, gross=520, operating=130, debt=50, equity=900, cash=50),  # 0.52
            _annual(2022, revenue=1000, gross=540, operating=130, debt=50, equity=900, cash=50),  # 0.54
        ]

    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={"A": variable_gm_peer([2020, 2021, 2022])},
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    gm_std = result.industry_aggregates.industry_gross_margin_3y_std
    assert gm_std is not None
    assert gm_std > 0  # non-zero variance


def test_peer_with_zero_invested_capital_skipped(tmp_path: Path) -> None:
    """A peer with debt+equity-cash <= 0 (e.g. negative equity) shouldn't
    contribute to the median — division by zero or negative IC yields
    a meaningless ROIC."""
    bad = _annual(2024, revenue=1000, gross=500, operating=100, debt=0, equity=0, cash=0)
    good = _profitable_peer([2022, 2023, 2024])
    client = _StubClient(
        peer_list=["BAD", "GOOD"],
        financials_by_symbol={
            "BAD": [bad],
            "GOOD": good,
        },
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    # BAD contributes nothing to ROIC; GOOD's ~11% is the median (only one).
    median = result.industry_aggregates.industry_roic_3y_median
    assert median is not None
    assert 0.10 < median < 0.13


def test_per_peer_failure_logged_not_raised(tmp_path: Path) -> None:
    class _PartialFailClient(_StubClient):
        def financials(self, symbol, freq="annual"):
            if symbol == "BAD":
                raise RuntimeError("data outage")
            return super().financials(symbol, freq)

    client = _PartialFailClient(
        peer_list=["GOOD", "BAD", "ALSOGOOD"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_attempted == 3
    # BAD threw; GOOD and ALSOGOOD landed details.
    assert result.n_peers_with_data == 2
    assert {d.symbol for d in result.peers_used} == {"GOOD", "ALSOGOOD"}


def test_insufficient_observations_for_std_yields_none(tmp_path: Path) -> None:
    """Only 1 GM observation total → std requires ≥2, returns None."""
    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={
            "A": [_annual(2024, revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50)],
        },
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.industry_aggregates.industry_gross_margin_3y_std is None


# ---------------------------------------------------------------------------
# Per-peer detail
# ---------------------------------------------------------------------------


def test_per_peer_detail_records_n_years(tmp_path: Path) -> None:
    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={"A": _profitable_peer([2020, 2021, 2022, 2023, 2024])},
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert len(result.peers_used) == 1
    detail = result.peers_used[0]
    assert detail.symbol == "A"
    assert detail.n_years == 5
    # Recent 3y is what we average for ROIC
    assert detail.avg_roic_3y is not None


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path: Path) -> None:
    """First call hits the network; second call same day reads cache."""
    client = _StubClient(
        peer_list=["A", "B"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    today = dt.date(2026, 4, 27)

    first = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path, today=today,
    )
    n_calls_after_first = len(client.financials_calls)
    second = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path, today=today,
    )
    assert len(client.financials_calls) == n_calls_after_first  # no extra calls
    assert (
        first.industry_aggregates.industry_roic_3y_median
        == second.industry_aggregates.industry_roic_3y_median
    )


def test_cache_disabled_always_fetches(tmp_path: Path) -> None:
    client = _StubClient(
        peer_list=["A", "B"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    compute_industry_aggregates("NVDA", client=client, cache=False, cache_dir=tmp_path)
    n1 = len(client.financials_calls)
    compute_industry_aggregates("NVDA", client=client, cache=False, cache_dir=tmp_path)
    n2 = len(client.financials_calls)
    assert n2 == 2 * n1


def test_cache_writes_durable_json(tmp_path: Path) -> None:
    client = _StubClient(
        peer_list=["A"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    today = dt.date(2026, 4, 27)
    compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path, today=today,
    )
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert "industry_aggregates" in payload
    assert "peers_used" in payload


def test_cache_returns_peer_aggregate_result(tmp_path: Path) -> None:
    """Round-tripped object must be a PeerAggregateResult, not a raw dict."""
    client = _StubClient(
        peer_list=["A"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    today = dt.date(2026, 4, 27)
    compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path, today=today,
    )
    second = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path, today=today,
    )
    assert isinstance(second, PeerAggregateResult)
    assert second.n_peers_attempted == 1


# ---------------------------------------------------------------------------
# as_of_date filtering — lookahead-bias guard for back-validation (#2)
# ---------------------------------------------------------------------------


def _public_entry(
    year: int, *, filed: str, **kwargs
) -> _Entry:
    """Build a stub entry with an explicit filed_date so the
    historical_adapter_finnhub._is_public_by filter can match."""
    e = _annual(year, **kwargs)
    e.filed_date = filed  # type: ignore[attr-defined]
    e.end_date = f"{year}-12-31"  # type: ignore[attr-defined]
    return e


def test_as_of_date_filters_out_future_filings(tmp_path: Path) -> None:
    """Calibration finding (#2, 2026-04): without as_of_date filtering,
    a 2018-06-30 calibration would pull peer financials filed in 2024
    and use them as the "industry baseline" — straight lookahead bias.

    With as_of_date set, peer entries filed AFTER that date must be
    excluded. The peer's avg_roic_3y should reflect only filings
    public on the calibration date.
    """
    # Peer 'A' has 5 annual entries: 2014, 2015, 2016, 2017, 2018.
    # Each filed in March of the following year.
    entries = [
        _public_entry(
            2014, filed="2015-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        ),
        _public_entry(
            2015, filed="2016-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        ),
        _public_entry(
            2016, filed="2017-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        ),
        _public_entry(
            2017, filed="2018-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        ),
        _public_entry(
            2018, filed="2019-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        ),
    ]
    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={"A": entries},
    )

    # Calibration on 2018-06-30: only 2014/2015/2016/2017 should be
    # public (filed by Mar 2018). The 2018 fiscal year was filed in
    # Mar 2019 — must be excluded.
    result = compute_industry_aggregates(
        "NVDA", client=client,
        cache=False, cache_dir=tmp_path,
        as_of_date=dt.date(2018, 6, 30),
    )
    detail = result.peers_used[0]
    # n_years counts ALL entries received from the API (pre-filter).
    # The post-filter entries are what matter for ROIC; we expose
    # their count via the gm_observations / avg_roic_3y fields.
    assert len(detail.gm_observations) <= 4  # at most 4 public-by-2018-06-30
    assert detail.avg_roic_3y is not None  # 4 valid years → avg computed


def test_as_of_date_none_uses_all_entries(tmp_path: Path) -> None:
    """Live-mode default: as_of_date=None → no filtering, all entries
    used. Confirms the new parameter is opt-in for back-validation
    and doesn't change live-mode behavior."""
    entries = [
        _public_entry(
            year, filed=f"{year + 1}-03-15",
            revenue=1000, gross=500, operating=130, debt=50, equity=900, cash=50,
        )
        for year in (2020, 2021, 2022, 2023, 2024)
    ]
    client = _StubClient(
        peer_list=["A"],
        financials_by_symbol={"A": entries},
    )
    result = compute_industry_aggregates(
        "NVDA", client=client,
        cache=False, cache_dir=tmp_path,
        as_of_date=None,
    )
    detail = result.peers_used[0]
    # All 5 entries used (no filter applied)
    assert detail.n_years == 5


def test_cache_separates_by_as_of_date(tmp_path: Path) -> None:
    """Two back-validation runs at different calibration dates must
    NOT collide in cache — that would let stale 2018 medians leak
    into a 2020 calibration run.
    """
    client = _StubClient(
        peer_list=["A"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
    )
    compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path,
        as_of_date=dt.date(2018, 6, 30),
    )
    compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path,
        as_of_date=dt.date(2020, 6, 30),
    )
    # Two separate cache files should exist
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 2


# ---------------------------------------------------------------------------
# P1d (2026-04): explicit industry filter
# ---------------------------------------------------------------------------


class _Profile:
    """Minimal Finnhub Profile shape for explicit industry filter tests."""

    def __init__(self, industry: str | None) -> None:
        self.finnhub_industry = industry


class _ProfiledClient(_StubClient):
    """`_StubClient` extended with a `profile()` method.

    `industry_by_symbol` maps a symbol to its `finnhub_industry`. A
    missing entry → profile() raises (simulates Finnhub 404), which
    `_safe_industry` catches and treats as None.
    """

    def __init__(
        self,
        *,
        peer_list=None,
        financials_by_symbol=None,
        default_financials=None,
        industry_by_symbol: dict[str, str | None] | None = None,
        focal_industry: str | None = None,
    ) -> None:
        super().__init__(
            peer_list=peer_list,
            financials_by_symbol=financials_by_symbol,
            default_financials=default_financials,
        )
        self._industries = dict(industry_by_symbol or {})
        # The focal symbol's industry is provided separately (it isn't
        # in peer_list but profile() is still called for it).
        if focal_industry is not None:
            self._focal_industry = focal_industry
        else:
            self._focal_industry = None
        self.profile_calls: list[str] = []

    def profile(self, symbol: str):
        self.profile_calls.append(symbol)
        u = symbol.upper()
        if u in self._industries:
            return _Profile(self._industries[u])
        if self._focal_industry is not None and u == "NVDA":
            return _Profile(self._focal_industry)
        # Simulate Finnhub returning a profile but with no industry
        # field populated (common for foreign-listing peers like .TW).
        return _Profile(None)


def test_industry_filter_keeps_matched_peers(tmp_path: Path) -> None:
    """When focal and peers share the industry, all peers contribute."""
    client = _ProfiledClient(
        peer_list=["AMD", "INTC"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
        industry_by_symbol={"AMD": "Semiconductors", "INTC": "Semiconductors"},
        focal_industry="Semiconductors",
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.focal_industry == "Semiconductors"
    assert result.n_peers_industry_mismatch == 0
    assert result.n_peers_with_data == 2
    assert result.industry_aggregates.industry_roic_3y_median is not None


def test_industry_filter_excludes_mismatched_peer(tmp_path: Path) -> None:
    """A peer with a different industry is rejected from ROIC median."""
    client = _ProfiledClient(
        peer_list=["AMD", "COIN.TO"],
        financials_by_symbol={
            "AMD": _profitable_peer([2022, 2023, 2024]),
            # COIN.TO has financials, but its industry differs — must
            # NOT contribute even though data is available.
            "COIN.TO": _high_roic_peer([2022, 2023, 2024]),
        },
        industry_by_symbol={
            "AMD": "Semiconductors",
            "COIN.TO": "Financial Services",
        },
        focal_industry="Semiconductors",
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_industry_mismatch == 1
    # Only AMD contributes; the median is just AMD's avg_roic_3y.
    assert result.n_peers_with_data == 1
    matched = [d for d in result.peers_used if not d.industry_mismatch]
    rejected = [d for d in result.peers_used if d.industry_mismatch]
    assert {d.symbol for d in matched} == {"AMD"}
    assert {d.symbol for d in rejected} == {"COIN.TO"}
    # Mismatched peer should not have triggered a financials() fetch.
    fetched = [c[0] for c in client.financials_calls]
    assert "COIN.TO" not in fetched
    assert "AMD" in fetched


def test_industry_filter_treats_unknown_peer_industry_as_mismatch(
    tmp_path: Path,
) -> None:
    """Foreign-listing peers (like .TW) often return industry=None.

    With an explicit focal industry known, an unknown peer industry
    is treated as mismatch (defensive — could be a cross-listing).
    """
    client = _ProfiledClient(
        peer_list=["2330.TW"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
        industry_by_symbol={"2330.TW": None},  # explicitly returns None
        focal_industry="Semiconductors",
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.n_peers_industry_mismatch == 1
    assert result.industry_aggregates.industry_roic_3y_median is None


def test_industry_filter_disabled_when_focal_industry_unknown(
    tmp_path: Path,
) -> None:
    """If even the focal's industry is None, the filter disables.

    Backward compat: clients without a working profile() method, or
    cases where Finnhub doesn't index the focal's industry, must fall
    back to natural filtering (empty financials → excluded). All peers
    pass the explicit gate; only data availability gates them.
    """
    client = _ProfiledClient(
        peer_list=["AMD", "INTC"],
        default_financials=_profitable_peer([2022, 2023, 2024]),
        industry_by_symbol={"AMD": "Semiconductors", "INTC": "Software"},
        focal_industry=None,
    )
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=False, cache_dir=tmp_path,
    )
    assert result.focal_industry is None
    assert result.n_peers_industry_mismatch == 0
    # Both peers contribute even though INTC has a different industry,
    # because the filter is disabled when focal is unknown.
    assert result.n_peers_with_data == 2


def test_industry_filter_cache_roundtrip(tmp_path: Path) -> None:
    """The new fields survive a cache write + read cycle."""
    client = _ProfiledClient(
        peer_list=["AMD", "COIN.TO"],
        financials_by_symbol={
            "AMD": _profitable_peer([2022, 2023, 2024]),
            "COIN.TO": _profitable_peer([2022, 2023, 2024]),
        },
        industry_by_symbol={"AMD": "Semiconductors", "COIN.TO": "Financial Services"},
        focal_industry="Semiconductors",
    )
    fresh = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path,
    )
    cached = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path,
    )
    assert cached.focal_industry == fresh.focal_industry == "Semiconductors"
    assert cached.n_peers_industry_mismatch == fresh.n_peers_industry_mismatch == 1
    cached_rejected = [d for d in cached.peers_used if d.industry_mismatch]
    assert len(cached_rejected) == 1
    assert cached_rejected[0].industry == "Financial Services"


def test_industry_filter_legacy_cache_backward_compat(tmp_path: Path) -> None:
    """A pre-P1d cache file (no industry/mismatch fields) deserializes cleanly."""
    legacy_payload = {
        "industry_aggregates": {
            "industry_roic_3y_median": 0.10,
            "industry_gross_margin_3y_std": 0.02,
        },
        "peers_used": [
            {
                "symbol": "AMD",
                "n_years": 3,
                "avg_roic_3y": 0.10,
                "gm_observations": [0.50, 0.51, 0.52],
            }
        ],
        "n_peers_attempted": 1,
        "n_peers_with_data": 1,
    }
    today = dt.date.today()
    cache_path = tmp_path / f"NVDA_{today.isoformat()}.json"
    cache_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    client = _StubClient()  # would error if hit; cache must short-circuit
    result = compute_industry_aggregates(
        "NVDA", client=client, cache=True, cache_dir=tmp_path,
    )
    assert result.focal_industry is None
    assert result.n_peers_industry_mismatch == 0
    assert result.peers_used[0].industry is None
    assert result.peers_used[0].industry_mismatch is False
