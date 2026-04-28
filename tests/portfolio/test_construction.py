"""Stage 6 portfolio construction tests.

Verifies the orchestrator wiring:
  - Returns fetcher injection
  - HRP → bounds → trade list pipeline
  - Cluster collision adjustment trims non-leaders
  - 1%/30% bounds enforced with redistribution
  - Trade list math under various existing-portfolio scenarios
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from wise_investor.portfolio.construction import (
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    _apply_bounds,
    _apply_cluster_adjustment,
    _compute_trades,
    construct_portfolio,
)

# ---------------------------------------------------------------------------
# Synthetic returns + injectable fetcher
# ---------------------------------------------------------------------------


def _synth_returns(symbols: list[str], n_obs: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {s: rng.standard_normal(n_obs) * 0.02 for s in symbols}
    return pd.DataFrame(cols)


def _make_fetcher(returns: pd.DataFrame):
    def _fetch(symbols, start, end):
        return returns[[s for s in symbols if s in returns.columns]].copy()
    return _fetch


# ---------------------------------------------------------------------------
# Stage 5 positioning report stub (matches the public shape we use)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PositionStub:
    ticker: str
    cluster_id: int | None


@dataclass(frozen=True)
class _ReportStub:
    survivor_positions: tuple[_PositionStub, ...]
    clusters: tuple = ()  # unused by the orchestrator's adjustment logic


# ---------------------------------------------------------------------------
# Construct end-to-end
# ---------------------------------------------------------------------------


def test_construct_returns_target_weights_summing_to_one() -> None:
    syms = ["A", "B", "C", "D"]
    returns = _synth_returns(syms, n_obs=500)
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
    )
    assert set(result.target_weights.keys()) == set(syms)
    total = sum(result.target_weights.values())
    assert abs(total - 1.0) < 1e-6


def test_construct_records_excluded_tickers_when_no_data() -> None:
    """Tickers absent from the returns DataFrame should be flagged as
    excluded, not silently ignored."""
    syms = ["A", "B", "GHOST"]
    returns = _synth_returns(["A", "B"], n_obs=200)
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
    )
    assert result.n_excluded_no_data == 1
    assert "GHOST" in result.excluded_tickers
    assert "GHOST" not in result.target_weights


def test_construct_no_returns_yields_empty_result() -> None:
    syms = ["A", "B"]
    result = construct_portfolio(
        syms,
        return_fetcher=lambda s, st, en: pd.DataFrame(),
    )
    assert result.target_weights == {}


def test_construct_empty_survivor_list() -> None:
    result = construct_portfolio([], return_fetcher=_make_fetcher(pd.DataFrame()))
    assert result.target_weights == {}


def test_construct_fetcher_failure_handled_gracefully() -> None:
    def _fail(syms, st, en):
        raise RuntimeError("yfinance offline")

    result = construct_portfolio(
        ["A", "B"], return_fetcher=_fail,
    )
    assert result.target_weights == {}
    assert result.n_excluded_no_data == 2


# ---------------------------------------------------------------------------
# Cluster adjustment
# ---------------------------------------------------------------------------


def test_cluster_adjustment_trims_followers_and_redistributes_to_leader() -> None:
    """Two survivors A, B in same cluster (A heavier). After cluster
    adjustment, A's weight grows by trim taken from B."""
    weights = pd.Series({"A": 0.6, "B": 0.4})
    report = _ReportStub(
        survivor_positions=(
            _PositionStub("A", 0),
            _PositionStub("B", 0),
        ),
    )
    adjusted, trims = _apply_cluster_adjustment(
        weights, report, trim_factor=0.5
    )
    # A is leader, B is follower → B trimmed by 50% of original (0.4 → 0.2)
    # Trim of 0.2 redistributed to A (0.6 → 0.8). After renormalize, weights sum to 1.
    assert abs(adjusted.sum() - 1.0) < 1e-9
    assert adjusted["A"] > weights["A"]
    assert adjusted["B"] < weights["B"]
    assert "B" in trims
    assert "A" not in trims


def test_cluster_adjustment_skips_singletons() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    report = _ReportStub(
        survivor_positions=(
            _PositionStub("A", 0),
            _PositionStub("B", 1),
            _PositionStub("C", 2),
        ),
    )
    adjusted, trims = _apply_cluster_adjustment(weights, report, trim_factor=0.5)
    # Each in its own cluster → no trim.
    assert trims == {}
    assert adjusted.equals(weights)


def test_cluster_adjustment_handles_three_in_cluster() -> None:
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    report = _ReportStub(
        survivor_positions=(
            _PositionStub(t, 0) for t in ("A", "B", "C")
        ),
    )
    adjusted, trims = _apply_cluster_adjustment(weights, report, trim_factor=0.5)
    # A is leader; B and C are followers. Both should be trimmed.
    assert "B" in trims
    assert "C" in trims
    assert "A" not in trims
    assert adjusted["A"] > weights["A"]


def test_cluster_adjustment_no_op_when_no_report() -> None:
    syms = ["A", "B", "C"]
    returns = _synth_returns(syms, n_obs=300)
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
    )
    assert result.cluster_adjustments == {}


# ---------------------------------------------------------------------------
# Bounds enforcement
# ---------------------------------------------------------------------------


def test_bounds_caps_weights_above_max() -> None:
    # 4 positions → feasible under max=0.30 (sum-at-max = 1.20).
    weights = pd.Series({"A": 0.50, "B": 0.20, "C": 0.20, "D": 0.10})
    out = _apply_bounds(weights, min_weight=0.01, max_weight=0.30)
    assert out["A"] <= 0.30 + 1e-9
    assert abs(out.sum() - 1.0) < 1e-9


def test_bounds_floors_weights_below_min() -> None:
    weights = pd.Series({"A": 0.95, "B": 0.04, "C": 0.01})
    out = _apply_bounds(weights, min_weight=0.05, max_weight=0.95)
    assert out["B"] >= 0.05 - 1e-9
    assert out["C"] >= 0.05 - 1e-9
    assert abs(out.sum() - 1.0) < 1e-9


def test_bounds_redistributes_excess_proportionally() -> None:
    """If A is capped, the excess goes to other unpinned positions in
    proportion to their current weights (not equally)."""
    weights = pd.Series({"A": 0.50, "B": 0.30, "C": 0.10, "D": 0.10})
    out = _apply_bounds(weights, min_weight=0.01, max_weight=0.30)
    # A capped at 0.30; B already at boundary (also pinned).
    # Excess from A's clip goes to C and D, but C and D started equal,
    # so B (originally 0.30, pinned at max) should remain at 0.30 and
    # C, D should grow equally — both above their original 0.10.
    assert out["C"] > 0.10
    assert out["D"] > 0.10
    assert abs(out.sum() - 1.0) < 1e-9


def test_bounds_uniform_fallback_when_too_many_positions() -> None:
    """Min weight 0.01 × 200 positions = 2.0 > 1.0 → infeasible.
    Should fall back to uniform allocation gracefully."""
    weights = pd.Series({f"S{i}": 1.0 / 200 for i in range(200)})
    out = _apply_bounds(weights, min_weight=0.01, max_weight=0.30)
    # 200 × 0.01 = 2.0; infeasible. Output should be uniform 1/200.
    assert all(abs(v - 1.0 / 200) < 1e-9 for v in out)


def test_bounds_already_satisfied_pass_through() -> None:
    weights = pd.Series({"A": 0.20, "B": 0.30, "C": 0.50})
    out = _apply_bounds(weights, min_weight=0.01, max_weight=0.50)
    pd.testing.assert_series_equal(out, weights / weights.sum(), check_exact=False)


def test_default_bounds_are_constitutional() -> None:
    assert DEFAULT_MIN_WEIGHT == 0.01
    assert DEFAULT_MAX_WEIGHT == 0.30


# ---------------------------------------------------------------------------
# Trade computation
# ---------------------------------------------------------------------------


def test_trades_buy_when_target_above_current() -> None:
    weights = pd.Series({"NVDA": 0.40, "AAPL": 0.60})
    existing = {"NVDA": 10000.0, "AAPL": 60000.0}
    trades = _compute_trades(weights, existing_positions=existing, total_capital=100000)
    by_sym = {t.symbol: t for t in trades}
    # NVDA: target 40%, current 10% → buy 30k
    assert by_sym["NVDA"].trade_value_usd == pytest.approx(40000 - 10000)
    # AAPL: target 60%, current 60% → no trade
    assert by_sym["AAPL"].trade_value_usd == pytest.approx(60000 - 60000)


def test_trades_sell_when_target_below_current() -> None:
    weights = pd.Series({"NVDA": 0.30})
    existing = {"NVDA": 50000.0}
    trades = _compute_trades(weights, existing_positions=existing, total_capital=100000)
    nvda = next(t for t in trades if t.symbol == "NVDA")
    # target 30k, current 50k → sell 20k (negative)
    assert nvda.trade_value_usd == pytest.approx(30000 - 50000)


def test_trades_full_sell_for_dropped_holdings() -> None:
    """A current holding not in target weights gets a full sell."""
    weights = pd.Series({"AAPL": 1.0})
    existing = {"NVDA": 50000.0, "AAPL": 50000.0}
    trades = _compute_trades(weights, existing_positions=existing, total_capital=100000)
    nvda = next(t for t in trades if t.symbol == "NVDA")
    assert nvda.target_weight == 0.0
    assert nvda.trade_value_usd == pytest.approx(-50000)


def test_trades_case_insensitive_symbol_match() -> None:
    weights = pd.Series({"nvda": 0.50, "aapl": 0.50})
    existing = {"NVDA": 30000.0, "AAPL": 30000.0}
    trades = _compute_trades(weights, existing_positions=existing, total_capital=100000)
    by_sym = {t.symbol: t for t in trades}
    assert "NVDA" in by_sym  # uppercase normalized


def test_construct_with_existing_positions_emits_trades() -> None:
    syms = ["A", "B", "C"]
    returns = _synth_returns(syms, n_obs=300, seed=1)
    existing = {"A": 50000.0}
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
        existing_positions=existing,
    )
    assert result.total_capital_usd == pytest.approx(50000.0)
    assert len(result.trades) >= len(syms)


def test_construct_total_capital_overrides_existing_sum() -> None:
    syms = ["A", "B"]
    returns = _synth_returns(syms, n_obs=300)
    existing = {"A": 10000.0}
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
        existing_positions=existing,
        total_capital_usd=100000.0,
    )
    assert result.total_capital_usd == pytest.approx(100000.0)


def test_construct_no_trades_when_no_existing_positions() -> None:
    syms = ["A", "B"]
    returns = _synth_returns(syms, n_obs=300)
    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
    )
    assert result.trades == ()


# ---------------------------------------------------------------------------
# End-to-end with all features
# ---------------------------------------------------------------------------


def test_end_to_end_with_clusters_and_bounds_and_trades() -> None:
    syms = ["NVDA", "AMD", "INTC", "KO", "PEP"]
    returns = _synth_returns(syms, n_obs=400, seed=7)

    # Stage-5-like stub: NVDA/AMD/INTC in cluster 0, KO/PEP in cluster 1
    report = _ReportStub(
        survivor_positions=(
            _PositionStub("NVDA", 0),
            _PositionStub("AMD", 0),
            _PositionStub("INTC", 0),
            _PositionStub("KO", 1),
            _PositionStub("PEP", 1),
        ),
    )
    existing = {"NVDA": 20000.0, "KO": 5000.0}

    result = construct_portfolio(
        syms,
        return_fetcher=_make_fetcher(returns),
        positioning_report=report,
        existing_positions=existing,
        total_capital_usd=100000.0,
    )

    # Weights sum to 1, all in [min, max]
    assert abs(sum(result.target_weights.values()) - 1.0) < 1e-9
    for w in result.target_weights.values():
        assert w >= DEFAULT_MIN_WEIGHT - 1e-9
        assert w <= DEFAULT_MAX_WEIGHT + 1e-9

    # Trades cover all 5 + 0 dropped = 5 entries
    assert len(result.trades) == 5
    # Total target value matches total capital within float tolerance
    target_total = sum(t.target_value_usd for t in result.trades)
    assert abs(target_total - 100000.0) < 1e-3
