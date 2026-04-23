"""Tests for the portfolio SQLite store (design-v2.2 §5.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.portfolio.store import PortfolioStore, Position


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_upsert_and_get(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "portfolio.sqlite")
    p = store.upsert_position(
        symbol="NVDA", shares=10.0, cost_basis_usd=5000.0, tier=1
    )
    assert p.symbol == "NVDA"
    assert p.shares == 10.0
    assert p.tier == 1
    assert p.first_bought  # non-empty ISO date
    # Round-trip
    assert store.get_position("NVDA") == p
    assert store.get_position("nvda") == p  # symbol normalized


def test_upsert_existing_preserves_first_bought(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    p1 = store.upsert_position("NVDA", 10.0, 5000.0, 1, first_bought="2026-01-15")
    p2 = store.upsert_position("NVDA", 20.0, 11000.0, 1)  # no first_bought arg
    assert p2.first_bought == "2026-01-15"
    assert p2.shares == 20.0
    assert p2.cost_basis_usd == 11000.0


def test_upsert_rejects_negative_shares(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    with pytest.raises(ValueError, match="shares"):
        store.upsert_position("NVDA", -5.0, 1000.0, 1)


def test_upsert_rejects_invalid_tier(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    with pytest.raises(ValueError, match="tier"):
        store.upsert_position("NVDA", 10.0, 5000.0, 4)


def test_delete_returns_true_when_row_existed(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    assert store.delete_position("NVDA") is True
    assert store.get_position("NVDA") is None


def test_delete_returns_false_when_missing(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    assert store.delete_position("ZZZZ") is False


def test_list_sorted_by_tier_then_symbol(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("TSM", 5.0, 500.0, 2)
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    store.upsert_position("GEV", 3.0, 1200.0, 1)
    symbols = [p.symbol for p in store.list_positions()]
    # tier 1 alphabetical, then tier 2
    assert symbols == ["GEV", "NVDA", "TSM"]


# ---------------------------------------------------------------------------
# Derived views
# ---------------------------------------------------------------------------


def test_avg_cost_per_share(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    p = store.upsert_position("NVDA", 10.0, 5000.0, 1)
    assert p.avg_cost_per_share == 500.0


def test_avg_cost_per_share_zero_shares(tmp_path: Path) -> None:
    p = Position(
        symbol="X", shares=0.0, cost_basis_usd=0.0,
        first_bought="2026-01-01", last_updated="2026-01-01T00:00:00",
        tier=1,
    )
    assert p.avg_cost_per_share is None


def test_snapshot_weights_happy_path(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    store.upsert_position("GEV", 5.0, 1000.0, 1)
    snaps = store.snapshot_weights({"NVDA": 600.0, "GEV": 400.0})
    # Total MV = 10*600 + 5*400 = 6000 + 2000 = 8000
    by_sym = {s.symbol: s for s in snaps}
    assert by_sym["NVDA"].market_value_usd == 6000.0
    assert by_sym["NVDA"].weight_pct == 75.0
    assert by_sym["GEV"].weight_pct == 25.0
    # P/L: NVDA cost 5000, MV 6000 → +1000
    assert by_sym["NVDA"].unrealized_pnl_usd == 1000.0


def test_snapshot_weights_missing_price(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    store.upsert_position("GEV", 5.0, 1000.0, 1)
    snaps = store.snapshot_weights({"NVDA": 600.0, "GEV": None})
    by_sym = {s.symbol: s for s in snaps}
    # NVDA is the only priced name so owns 100% of the known-MV denominator.
    assert by_sym["NVDA"].weight_pct == 100.0
    assert by_sym["GEV"].market_value_usd is None
    assert by_sym["GEV"].weight_pct is None


# ---------------------------------------------------------------------------
# sizing_gap
# ---------------------------------------------------------------------------


def test_sizing_gap_within_band(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    msg = store.sizing_gap(
        "NVDA", suggested_low_pct=3.0, suggested_high_pct=5.0,
        prices={"NVDA": 400.0},
    )
    # Single holding → weight is 100%, clearly above band.
    assert "trim" in msg


def test_sizing_gap_below_band(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 1.0, 500.0, 1)
    store.upsert_position("OTHER", 100.0, 50000.0, 2)
    # NVDA is tiny — weight ~1%.
    msg = store.sizing_gap(
        "NVDA", suggested_low_pct=3.0, suggested_high_pct=5.0,
        prices={"NVDA": 500.0, "OTHER": 500.0},
    )
    assert "add" in msg
    assert "suggestion 3.0-5.0%" in msg


def test_sizing_gap_above_band(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    store.upsert_position("NVDA", 10.0, 5000.0, 1)
    store.upsert_position("OTHER", 1.0, 50.0, 2)
    # NVDA dominates — weight ~99%.
    msg = store.sizing_gap(
        "NVDA", suggested_low_pct=3.0, suggested_high_pct=5.0,
        prices={"NVDA": 500.0, "OTHER": 50.0},
    )
    assert "trim" in msg


def test_sizing_gap_no_position(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    msg = store.sizing_gap(
        "NVDA", suggested_low_pct=2.0, suggested_high_pct=4.0,
        prices={"NVDA": 500.0},
    )
    assert "No position" in msg
    assert "new entry opportunity" in msg


def test_sizing_gap_within_band_precise(tmp_path: Path) -> None:
    # Construct a scenario where NVDA weight is exactly inside the band.
    store = PortfolioStore(tmp_path / "p.sqlite")
    # Total MV: 400 (NVDA) + 9600 (other) = 10000. NVDA weight = 4%.
    store.upsert_position("NVDA", 1.0, 350.0, 1)
    store.upsert_position("OTHER", 96.0, 9000.0, 2)
    msg = store.sizing_gap(
        "NVDA", suggested_low_pct=3.0, suggested_high_pct=5.0,
        prices={"NVDA": 400.0, "OTHER": 100.0},
    )
    assert "within band" in msg
    assert "no action" in msg


def test_sizing_gap_rejects_inverted_band(tmp_path: Path) -> None:
    store = PortfolioStore(tmp_path / "p.sqlite")
    with pytest.raises(ValueError, match="low pct"):
        store.sizing_gap(
            "NVDA", suggested_low_pct=5.0, suggested_high_pct=3.0,
            prices={"NVDA": 500.0},
        )
