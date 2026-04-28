"""HRP algorithm tests on synthetic returns.

Verifies the López de Prado 2016 properties:
  - Weights sum to 1.0
  - Lower-volatility assets get larger weights (inverse-variance bias)
  - Equal-correlation, equal-volatility universe → equal weights
  - Single-asset universe → 100% weight
  - Empty / degenerate inputs handled gracefully
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wise_investor.portfolio.hrp import compute_hrp_weights


def _make_returns(
    n_obs: int,
    sigma_per_asset: dict[str, float],
    *,
    correlation: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic returns DataFrame.

    Each asset has its own per-day std dev `sigma`. When `correlation`
    is non-zero, all assets share the same correlation via a single
    common factor model:

        r_i = sigma_i * (sqrt(rho) * c + sqrt(1-rho) * eps_i)

    where c is the shared latent and eps_i is iid.
    """
    rng = np.random.default_rng(seed)
    tickers = list(sigma_per_asset.keys())
    n_assets = len(tickers)

    common = rng.standard_normal(n_obs)
    independent = rng.standard_normal((n_obs, n_assets))

    rho = correlation
    sqrt_rho = np.sqrt(max(rho, 0.0))
    sqrt_one_minus = np.sqrt(max(1.0 - rho, 0.0))

    cols: dict[str, np.ndarray] = {}
    for j, t in enumerate(tickers):
        sigma = sigma_per_asset[t]
        cols[t] = sigma * (sqrt_rho * common + sqrt_one_minus * independent[:, j])
    return pd.DataFrame(cols)


# ---------------------------------------------------------------------------
# Basic shape contracts
# ---------------------------------------------------------------------------


def test_weights_sum_to_one() -> None:
    returns = _make_returns(
        n_obs=300, sigma_per_asset={"A": 0.01, "B": 0.02, "C": 0.03}
    )
    w = compute_hrp_weights(returns)
    assert abs(w.sum() - 1.0) < 1e-9


def test_weights_indexed_by_ticker() -> None:
    returns = _make_returns(
        n_obs=300, sigma_per_asset={"NVDA": 0.02, "AMD": 0.025, "INTC": 0.018}
    )
    w = compute_hrp_weights(returns)
    assert set(w.index) == {"NVDA", "AMD", "INTC"}


def test_all_weights_positive() -> None:
    returns = _make_returns(
        n_obs=300, sigma_per_asset={"A": 0.01, "B": 0.02, "C": 0.03}
    )
    w = compute_hrp_weights(returns)
    assert (w >= 0).all()


# ---------------------------------------------------------------------------
# Inverse-variance bias
# ---------------------------------------------------------------------------


def test_lower_variance_asset_gets_higher_weight() -> None:
    """One quiet asset (sigma=0.005) plus three loud assets (sigma=0.05)
    → the quiet one should land with a much larger weight."""
    returns = _make_returns(
        n_obs=400,
        sigma_per_asset={"QUIET": 0.005, "LOUD1": 0.05, "LOUD2": 0.05, "LOUD3": 0.05},
        correlation=0.0,
    )
    w = compute_hrp_weights(returns)
    # The quiet asset should dominate; loose lower bound to absorb noise.
    assert w["QUIET"] > 0.40
    for noisy in ("LOUD1", "LOUD2", "LOUD3"):
        assert w[noisy] < w["QUIET"]


def test_equal_volatility_yields_roughly_equal_weights() -> None:
    """Independent assets with identical volatility → HRP should split
    weight roughly evenly (within a margin)."""
    returns = _make_returns(
        n_obs=500,
        sigma_per_asset={"A": 0.02, "B": 0.02, "C": 0.02, "D": 0.02},
        correlation=0.0,
    )
    w = compute_hrp_weights(returns)
    # Expected: 0.25 each. Allow ±0.07 to absorb sampling noise.
    for t in ("A", "B", "C", "D"):
        assert abs(w[t] - 0.25) < 0.07


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_asset_yields_full_weight() -> None:
    returns = _make_returns(n_obs=100, sigma_per_asset={"NVDA": 0.02})
    w = compute_hrp_weights(returns)
    assert len(w) == 1
    assert w["NVDA"] == pytest.approx(1.0)


def test_empty_returns_yields_empty_weights() -> None:
    w = compute_hrp_weights(pd.DataFrame())
    assert w.empty


def test_zero_variance_column_dropped() -> None:
    """A column that's all zeros has zero variance → undefined
    correlation. The cleaner should drop it before HRP runs."""
    returns = _make_returns(
        n_obs=200, sigma_per_asset={"A": 0.01, "B": 0.02}
    )
    returns["DEAD"] = 0.0  # zero variance column
    w = compute_hrp_weights(returns)
    assert "DEAD" not in w.index
    assert {"A", "B"} == set(w.index)


def test_all_nan_column_dropped() -> None:
    returns = _make_returns(
        n_obs=200, sigma_per_asset={"A": 0.01, "B": 0.02}
    )
    returns["MISSING"] = float("nan")
    w = compute_hrp_weights(returns)
    assert "MISSING" not in w.index


def test_perfect_correlation_handled_without_nan() -> None:
    """Perfectly correlated columns produce zeros in the distance
    matrix, which can confuse the linkage step. The implementation
    must handle this without producing NaN weights."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(300)
    df = pd.DataFrame({"A": base * 0.01, "B": base * 0.01, "C": base * 0.02})
    w = compute_hrp_weights(df)
    assert not w.isna().any()
    assert abs(w.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_calls_same_result() -> None:
    returns = _make_returns(
        n_obs=300,
        sigma_per_asset={"A": 0.01, "B": 0.02, "C": 0.015, "D": 0.03},
    )
    w1 = compute_hrp_weights(returns)
    w2 = compute_hrp_weights(returns)
    pd.testing.assert_series_equal(w1, w2)
