"""Hierarchical Risk Parity (López de Prado, JPM 2016).

A direct NumPy + SciPy implementation. We avoid riskfolio-lib because
the algorithm itself is short enough to write and audit, and
riskfolio-lib brings cvxpy / scikit-learn — heavyweight dependencies
that we don't need for any other purpose.

The three steps from the paper:

  1. **Tree clustering.** Convert the correlation matrix to a distance
     matrix d_ij = sqrt(0.5 * (1 - corr_ij)) — ensures triangle
     inequality. Run single-linkage hierarchical clustering on it.

  2. **Quasi-diagonalization.** Walk the linkage tree from the last
     merger backwards, expanding each merged-node id into its two
     children until only original-leaf positions remain. The result
     is a permutation that places similar tickers next to each other.

  3. **Recursive bisection.** Split the quasi-diagonal order in half;
     within each pair of halves, allocate inverse-cluster-variance
     weights (alpha for the left, 1-alpha for the right). Recurse on
     each half until each cluster is a single ticker.

The output is a `pd.Series` of weights indexed by ticker, summing to 1.

Numerical robustness: degenerate inputs (single-asset universe, zero
variance, perfect correlation) are handled explicitly so the function
never raises on real-world data quirks. Bounds enforcement is in
`construction.py` as a separate concern — the constitution's 1%/30%
caps live there because they're not part of HRP itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute HRP portfolio weights from a returns DataFrame.

    Args:
        returns: rows = trading days, columns = tickers, values = daily
            simple or log returns. NaN columns and rows are dropped
            before computation.

    Returns:
        pd.Series of weights indexed by ticker. Sums to 1.0 (or 0.0
        when the input is empty after cleaning).
    """
    cleaned = _clean_returns(returns)
    if cleaned.empty or len(cleaned.columns) == 0:
        return pd.Series(dtype=float)

    if len(cleaned.columns) == 1:
        return pd.Series([1.0], index=cleaned.columns)

    cov = cleaned.cov()
    corr = cleaned.corr().values

    # Distance: triangle-inequality preserving from López de Prado.
    # Clip to non-negative to absorb floating-point rounding that can
    # produce values like -1e-16 on perfectly correlated columns.
    dist = np.sqrt(0.5 * (1.0 - corr).clip(min=0.0))
    np.fill_diagonal(dist, 0.0)

    # Quasi-diagonal ordering via single-linkage clustering.
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_positions = _quasi_diag(link, n_items=len(cleaned.columns))
    sort_ix = [cleaned.columns[i] for i in sort_positions]

    # Recursive bisection.
    weights = pd.Series(1.0, index=sort_ix)
    groups: list[list[str]] = [sort_ix]
    while groups:
        next_groups: list[list[str]] = []
        for group in groups:
            if len(group) <= 1:
                continue
            mid = len(group) // 2
            left = group[:mid]
            right = group[mid:]
            v_left = _cluster_variance(cov, left)
            v_right = _cluster_variance(cov, right)
            denom = v_left + v_right
            # Both sides have zero variance → split evenly.
            alpha = 1.0 - v_left / denom if denom > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            next_groups.extend([left, right])
        groups = next_groups

    total = weights.sum()
    if total <= 0:
        # Degenerate: return uniform weights as a fallback.
        return pd.Series(
            [1.0 / len(weights)] * len(weights), index=weights.index
        )
    return weights / total


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Drop columns/rows that would poison the correlation matrix.

    - Columns that are entirely NaN have undefined correlation; drop them.
    - Columns with zero standard deviation produce nan correlations; drop.
    - After column drop, drop rows with any remaining NaN.
    """
    if returns.empty:
        return returns
    df = returns.dropna(axis=1, how="all")
    # Drop zero-variance columns
    stds = df.std(axis=0)
    df = df.loc[:, stds > 0]
    df = df.dropna(axis=0, how="any")
    return df


def _quasi_diag(link: np.ndarray, *, n_items: int) -> list[int]:
    """López de Prado's getQuasiDiag (paper Algorithm 1).

    Walks the linkage matrix from the last merger backwards, replacing
    each merged-node id with its two children until only original-leaf
    positions remain. Returns the permutation of leaf positions.
    """
    link_int = link.astype(int)
    sort_ix = pd.Series([link_int[-1, 0], link_int[-1, 1]])
    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        idx = df0.index
        j = df0.values - n_items
        sort_ix[idx] = link_int[j, 0]
        df1 = pd.Series(link_int[j, 1], index=idx + 1)
        sort_ix = pd.concat([sort_ix, df1]).sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    """Inverse-variance weighted cluster variance.

    Allocates within-cluster weight by 1/diag(cov), then computes
    w' Σ w. When any diagonal element is non-positive (zero or
    near-zero variance), falls back to equal weights to avoid
    blow-up. The single-asset case is exact: returns the asset's
    variance.
    """
    sub = cov.loc[items, items].values
    diag = np.diag(sub)
    if (diag <= 0).any():
        ivp = np.ones(len(diag)) / len(diag)
    else:
        ivp = 1.0 / diag
        ivp /= ivp.sum()
    return float(ivp.T @ sub @ ivp)


__all__ = ["compute_hrp_weights"]
