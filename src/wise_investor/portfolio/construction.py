"""Stage 6 portfolio construction orchestrator.

Pipeline (constitution Sec 6):

  1. Pull historical price returns for the survivor pool.
  2. Run HRP on the returns matrix → initial weights.
  3. Apply post-hoc value-chain adjustment: when ≥2 survivors map to
     the same Stage-5 cluster, the smaller-weighted ones get trimmed
     and the trim is redistributed to the cluster leader. This
     enforces "two HRP-favored names on the same node should not
     simultaneously be max-sized."
  4. Apply 1%/30% single-position bounds with iterative redistribution
     until the constraint is satisfied.
  5. If existing positions are supplied, compute the incremental trade
     list (target dollar value - current dollar value).

The 30% upper bound is calibrated against Buffett's largest-holding
historical average (1981-2024). The 1% lower bound prevents the
output from including symbolic, unactionable positions.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from wise_investor.portfolio.hrp import compute_hrp_weights

logger = logging.getLogger(__name__)


# Constitution-mandated single-position bounds.
DEFAULT_MIN_WEIGHT: float = 0.01
DEFAULT_MAX_WEIGHT: float = 0.30

# How aggressively to trim non-leaders when ≥2 survivors land in the
# same Stage-5 cluster. 0.7 = take 30% off the smaller positions and
# redistribute to the cluster leader. Configurable.
DEFAULT_CLUSTER_TRIM_FACTOR: float = 0.7

# Default historical return window for HRP (~2 years of trading days).
DEFAULT_LOOKBACK_DAYS: int = 504


# Type alias for the price-return fetcher injected for tests.
ReturnFetcher = Callable[[list[str], dt.date, dt.date], pd.DataFrame]
"""(symbols, start, end) → DataFrame with index=date, columns=tickers, values=daily returns."""


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionTrade:
    """One row of the rebalance recommendation."""

    symbol: str
    target_weight: float
    target_value_usd: float
    current_value_usd: float
    trade_value_usd: float  # positive = buy, negative = sell


@dataclass(frozen=True)
class PortfolioConstructionResult:
    """Aggregate output of Stage 6."""

    target_weights: dict[str, float]   # ticker -> weight in [0, 1]
    raw_hrp_weights: dict[str, float]  # before bounds + cluster adjust
    cluster_adjustments: dict[str, float]  # ticker -> trim factor applied
    bounds_min: float
    bounds_max: float
    trades: tuple[PositionTrade, ...] = ()
    total_capital_usd: float | None = None
    n_excluded_no_data: int = 0
    excluded_tickers: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def construct_portfolio(
    survivors: list[str],
    *,
    return_fetcher: ReturnFetcher | None = None,
    positioning_report: object | None = None,  # Stage5PositioningReport
    existing_positions: dict[str, float] | None = None,
    total_capital_usd: float | None = None,
    min_weight: float = DEFAULT_MIN_WEIGHT,
    max_weight: float = DEFAULT_MAX_WEIGHT,
    cluster_trim_factor: float = DEFAULT_CLUSTER_TRIM_FACTOR,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    today: dt.date | None = None,
) -> PortfolioConstructionResult:
    """Run the Stage 6 pipeline end-to-end.

    Args:
        survivors: Stage 4 survivor tickers.
        return_fetcher: Anything returning a DataFrame of daily returns.
            None constructs a yfinance-backed default (lazy import).
        positioning_report: Optional `Stage5PositioningReport`. When
            supplied, the cluster collision adjustment runs; without it,
            HRP weights pass through unchanged before bounds enforcement.
        existing_positions: dict {symbol: current market value USD}.
            When present, trades are computed against it. The dollar
            denominator is `total_capital_usd` when supplied, otherwise
            sum(existing_positions.values()).
        total_capital_usd: Capital to size the portfolio against. When
            None and existing_positions is present, it equals
            sum(existing_positions.values()). When both are None, no
            trade list is produced.
        min_weight, max_weight: Per-position bounds. Defaults match
            constitution Sec 6 (1% / 30%).
        cluster_trim_factor: Multiplier applied to non-leader weights
            within an over-represented Stage-5 cluster. 0.7 means
            "leave 70% of the original HRP weight, redistribute the
            remaining 30% to the leader."
        lookback_days: Trading-day window for the return matrix.
            Default ~2 years.
        today: Override for "today"; tests use a fixed date.
    """
    today = today or dt.date.today()
    start = today - dt.timedelta(days=int(lookback_days * 1.5))  # 1.5x to absorb non-trading days

    if return_fetcher is None:
        return_fetcher = _default_return_fetcher

    # 1. Fetch returns
    sym_list = [s for s in survivors if s and s.strip()]
    if not sym_list:
        return PortfolioConstructionResult(
            target_weights={},
            raw_hrp_weights={},
            cluster_adjustments={},
            bounds_min=min_weight,
            bounds_max=max_weight,
        )

    try:
        returns = return_fetcher(sym_list, start, today)
    except Exception as e:
        logger.warning("return_fetcher failed: %s", e)
        returns = pd.DataFrame()

    available_tickers = list(returns.columns) if not returns.empty else []
    excluded = [s for s in sym_list if s not in available_tickers]
    n_excluded = len(excluded)

    if not available_tickers:
        return PortfolioConstructionResult(
            target_weights={},
            raw_hrp_weights={},
            cluster_adjustments={},
            bounds_min=min_weight,
            bounds_max=max_weight,
            n_excluded_no_data=n_excluded,
            excluded_tickers=tuple(excluded),
        )

    # 2. HRP
    raw = compute_hrp_weights(returns)
    if raw.empty:
        return PortfolioConstructionResult(
            target_weights={},
            raw_hrp_weights={},
            cluster_adjustments={},
            bounds_min=min_weight,
            bounds_max=max_weight,
            n_excluded_no_data=n_excluded,
            excluded_tickers=tuple(excluded),
        )

    # 3. Cluster collision adjustment (Stage 5 → Stage 6 hand-off)
    weights = raw.copy()
    cluster_trims: dict[str, float] = {}
    if positioning_report is not None:
        weights, cluster_trims = _apply_cluster_adjustment(
            weights,
            positioning_report,
            trim_factor=cluster_trim_factor,
        )

    # 4. Bounds enforcement
    weights = _apply_bounds(weights, min_weight=min_weight, max_weight=max_weight)

    # 5. Trade list (if applicable)
    trades: tuple[PositionTrade, ...] = ()
    capital: float | None = total_capital_usd
    if existing_positions is not None:
        if capital is None:
            capital = float(sum(existing_positions.values()))
        if capital > 0:
            trades = _compute_trades(
                weights,
                existing_positions=existing_positions,
                total_capital=capital,
            )

    return PortfolioConstructionResult(
        target_weights=weights.to_dict(),
        raw_hrp_weights=raw.to_dict(),
        cluster_adjustments=cluster_trims,
        bounds_min=min_weight,
        bounds_max=max_weight,
        trades=trades,
        total_capital_usd=capital,
        n_excluded_no_data=n_excluded,
        excluded_tickers=tuple(excluded),
    )


# ---------------------------------------------------------------------------
# Cluster collision adjustment
# ---------------------------------------------------------------------------


def _apply_cluster_adjustment(
    weights: pd.Series,
    positioning_report: object,
    *,
    trim_factor: float,
) -> tuple[pd.Series, dict[str, float]]:
    """For each cluster with ≥2 survivors, trim non-leader weights.

    The leader (highest HRP weight in the cluster) absorbs the trimmed
    amount. Result is renormalized so weights still sum to 1.

    Returns:
      - adjusted weights
      - dict {ticker: trim_factor_applied} for audit
    """
    # Group survivors by cluster_id from the positioning report.
    # Survivors not in the weights index are silently ignored (they
    # didn't make HRP for some reason — e.g. excluded due to no data).
    # We don't break here because the report and weights index may not
    # perfectly overlap.
    clusters = getattr(positioning_report, "clusters", ())
    survivor_positions = getattr(positioning_report, "survivor_positions", ())

    # Build {cluster_id: [ticker, ...]} from positions instead of clusters
    # so we know each survivor's actual cluster (clusters list has the
    # member names but those are graph-node names, not tickers).
    by_cluster: dict[int, list[str]] = {}
    for p in survivor_positions:
        cid = getattr(p, "cluster_id", None)
        ticker = getattr(p, "ticker", None)
        if cid is None or ticker is None:
            continue
        if ticker not in weights.index:
            continue
        by_cluster.setdefault(cid, []).append(ticker)

    trims: dict[str, float] = {}
    adjusted = weights.copy()
    for _cid, tickers in by_cluster.items():
        if len(tickers) < 2:
            continue
        # Sort by weight desc; leader is index 0.
        ranked = sorted(tickers, key=lambda t: adjusted[t], reverse=True)
        leader = ranked[0]
        followers = ranked[1:]
        trimmed_total = 0.0
        for f in followers:
            original = float(adjusted[f])
            new = original * trim_factor
            adjusted[f] = new
            trimmed_total += original - new
            trims[f] = trim_factor
        # Redistribute trimmed weight to the leader.
        adjusted[leader] = float(adjusted[leader]) + trimmed_total

    # Renormalize defensively (cluster math should preserve sum, but
    # guard against floating-point drift).
    total = adjusted.sum()
    if total > 0:
        adjusted = adjusted / total

    # Reference clusters arg explicitly to satisfy linters that flag the
    # unused name; also makes the dependency on the public report shape
    # visible to readers.
    _ = clusters

    return adjusted, trims


# ---------------------------------------------------------------------------
# Bounds enforcement
# ---------------------------------------------------------------------------


def _apply_bounds(
    weights: pd.Series,
    *,
    min_weight: float,
    max_weight: float,
    max_iterations: int = 50,
) -> pd.Series:
    """Iteratively clip weights to [min_weight, max_weight] and
    redistribute the slack proportionally to unconstrained positions.

    The iteration converges quickly (usually 2-3 passes) because each
    redistribution can push neighbors into a constraint, which the next
    pass clips.
    """
    if weights.empty:
        return weights
    n = len(weights)
    if n * min_weight > 1.0 + 1e-9:
        # Infeasible: too many positions for the floor. Cap at uniform.
        logger.warning(
            "Cannot satisfy %d positions × %.2f min weight; falling back "
            "to uniform allocation",
            n,
            min_weight,
        )
        return pd.Series([1.0 / n] * n, index=weights.index)

    w = weights.copy().astype(float)
    for _ in range(max_iterations):
        # Cap and floor.
        capped = w.clip(upper=max_weight)
        floored = capped.clip(lower=min_weight)

        # If clipping had no effect, we're done.
        if (capped == w).all() and (floored == capped).all():
            w = floored
            break

        # Redistribute the excess (sum > 1) or shortfall (sum < 1) over
        # positions that aren't pinned at a boundary.
        total = floored.sum()
        if abs(total - 1.0) < 1e-12:
            w = floored
            break

        unpinned = (floored > min_weight + 1e-12) & (floored < max_weight - 1e-12)
        if not unpinned.any():
            # Everyone is pinned. Renormalize by scaling, then re-clip.
            w = floored / floored.sum()
            continue

        slack = total - 1.0
        # Distribute -slack across unpinned positions proportionally to
        # their current weight.
        unpinned_total = floored[unpinned].sum()
        if unpinned_total <= 0:
            w = floored / floored.sum()
            continue
        shares = floored[unpinned] / unpinned_total
        floored[unpinned] = floored[unpinned] - slack * shares

        w = floored

    # Final renormalization to absorb residual drift.
    return w / w.sum()


# ---------------------------------------------------------------------------
# Trade computation
# ---------------------------------------------------------------------------


def _compute_trades(
    weights: pd.Series,
    *,
    existing_positions: dict[str, float],
    total_capital: float,
) -> tuple[PositionTrade, ...]:
    """Compute incremental trades from current to target.

    Both weights index and existing_positions keys are unioned: a
    target ticker not currently held shows up as a buy, and a current
    holding not in the target shows up as a full sell.
    """
    out: list[PositionTrade] = []
    weights_dict = {k.upper(): float(v) for k, v in weights.items()}
    existing_upper = {k.upper(): float(v) for k, v in existing_positions.items()}

    all_symbols = sorted(set(weights_dict) | set(existing_upper))
    for sym in all_symbols:
        w = weights_dict.get(sym, 0.0)
        target_value = w * total_capital
        current_value = existing_upper.get(sym, 0.0)
        out.append(
            PositionTrade(
                symbol=sym,
                target_weight=w,
                target_value_usd=target_value,
                current_value_usd=current_value,
                trade_value_usd=target_value - current_value,
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Default yfinance-backed return fetcher
# ---------------------------------------------------------------------------


def _default_return_fetcher(
    symbols: list[str], start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Pull adjusted-close history from yfinance and convert to daily returns.

    Returns a DataFrame indexed by date, columns by ticker, values by
    simple daily return. Missing tickers are silently dropped.
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "yfinance is required for the default return fetcher."
        ) from e

    if not symbols:
        return pd.DataFrame()

    tickers_str = " ".join(symbols)
    history = yf.download(
        tickers_str,
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if history is None or history.empty:
        return pd.DataFrame()

    # Single-ticker case: yfinance returns flat columns. Normalize to
    # the wide multi-ticker shape so the downstream handles both.
    if len(symbols) == 1:
        if "Close" in history.columns:
            close = history[["Close"]].copy()
            close.columns = symbols
        else:
            return pd.DataFrame()
    else:
        if "Close" not in history.columns.get_level_values(0):
            return pd.DataFrame()
        close = history["Close"]

    returns = close.pct_change().dropna(how="all")
    return returns


__all__ = [
    "DEFAULT_CLUSTER_TRIM_FACTOR",
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MAX_WEIGHT",
    "DEFAULT_MIN_WEIGHT",
    "PortfolioConstructionResult",
    "PositionTrade",
    "ReturnFetcher",
    "construct_portfolio",
]
