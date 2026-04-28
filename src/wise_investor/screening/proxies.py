"""Pure proxy computations on TickerFundamentals.

Implements the quantitative side of constitution §15 — every formula
that can produce a number from already-fetched fundamentals. None of
this code touches the network or an LLM. Inputs are dataclasses,
outputs are dataclasses, behavior is fully deterministic. That makes
the screening pipeline testable end-to-end with hand-curated
fixtures, which is what calibration (Step 4) needs.

Notable choices:

  - Numbers can be `None` when the underlying fundamentals aren't
    available. We don't fabricate values; the prefilter integration
    layer treats `None` as "evaluate qualitatively in Stage 3"
    rather than "fail the axis." This honors §17 on the limits of
    automation.
  - `roic_advantage_trend` uses a simple OLS slope on yearly
    advantages. It's a noisy estimator at 3-5 data points; the
    constitution's `>= -0.005` threshold (less than 0.5pp/year
    erosion) is correspondingly forgiving. We do NOT smooth or use
    EWMA — that would hide the very erosion the rule is meant to
    catch.
  - Standard deviation uses sample std (n-1 divisor) for consistency
    with how analysts cite it. With ≥4 quarters, the bias is
    negligible.
"""

from __future__ import annotations

import statistics
from typing import Iterable

from wise_investor.screening.types import (
    AnnualFinancials,
    BottleneckProxies,
    FrontierProxies,
    MoatProxies,
    QuarterlyMargin,
    SegmentBreakdown,
    TickerFundamentals,
)


# ---------------------------------------------------------------------------
# Moat proxies
# ---------------------------------------------------------------------------


def _yearly_roic(rows: Iterable[AnnualFinancials]) -> list[tuple[int, float]]:
    """Compute (fiscal_year, ROIC) pairs from rows that carry both
    NOPAT and invested capital. Rows missing either are skipped.
    """
    out: list[tuple[int, float]] = []
    for r in rows:
        if (
            r.nopat is None
            or r.invested_capital is None
            or r.invested_capital == 0
        ):
            continue
        out.append((r.fiscal_year, r.nopat / r.invested_capital))
    return out


def _ols_slope(points: list[tuple[float, float]]) -> float | None:
    """Simple OLS slope of y vs x. Returns None when fewer than 2
    points or when x is constant.
    """
    if len(points) < 2:
        return None
    n = len(points)
    sum_x = sum(p[0] for p in points)
    sum_y = sum(p[1] for p in points)
    sum_xx = sum(p[0] * p[0] for p in points)
    sum_xy = sum(p[0] * p[1] for p in points)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    return (n * sum_xy - sum_x * sum_y) / denom


def compute_moat_proxies(funds: TickerFundamentals) -> MoatProxies:
    """Constitution §15 moat block.

    Requires ≥3 years of ROIC history (constitution §10 Auto-PASS 1).
    With fewer than 3 years we return roic_3y_avg=None so the
    prefilter's _evaluate_moat fires the explicit "<3 years" FAIL.
    """
    yearly_roic = _yearly_roic(funds.annual)
    # Constitution §10 Auto-PASS 1 — return None when data spans <3 years.
    if len(yearly_roic) < 3:
        roic_3y_avg = None
    else:
        recent_3 = yearly_roic[-3:]
        roic_3y_avg = sum(r for _, r in recent_3) / len(recent_3)

    if roic_3y_avg is not None and funds.industry_roic_3y_median is not None:
        roic_advantage = roic_3y_avg - funds.industry_roic_3y_median
    else:
        roic_advantage = None

    # Trend = OLS slope of yearly_advantage over yearly_roic, where
    # yearly_advantage_t = roic_t - industry_median (we use the same
    # median for every year because we only have a single industry
    # snapshot; this is a known approximation).
    if (
        funds.industry_roic_3y_median is not None
        and len(yearly_roic) >= 3
    ):
        yearly_adv = [
            (float(year), roic - funds.industry_roic_3y_median)
            for year, roic in yearly_roic
        ]
        roic_advantage_trend = _ols_slope(yearly_adv)
    else:
        roic_advantage_trend = None

    # Quarterly margin std — sample std requires ≥2 points.
    margin_values = [
        q.gross_margin for q in funds.quarterly_margins if q.gross_margin is not None
    ]
    if len(margin_values) >= 2:
        own_std = statistics.stdev(margin_values)
    else:
        own_std = None

    if own_std is not None and funds.industry_gross_margin_3y_std:
        gm_ratio = own_std / funds.industry_gross_margin_3y_std
    else:
        gm_ratio = None

    # Customer-concentration trend: hard to compute from public data
    # without per-year disclosure. We surface the *current* top-5 share
    # only (no trend) and let Stage 3 fill in the trend qualitatively.
    customer_trend = None  # explicit; reserved for future adapter coverage

    return MoatProxies(
        roic_3y_avg=roic_3y_avg,
        roic_advantage=roic_advantage,
        roic_advantage_trend=roic_advantage_trend,
        gross_margin_3y_std=own_std,
        gross_margin_industry_ratio=gm_ratio,
        customer_concentration_trend=customer_trend,
    )


# ---------------------------------------------------------------------------
# Frontier proxies — primarily LLM-driven by design (§17)
# ---------------------------------------------------------------------------


def compute_frontier_proxies(funds: TickerFundamentals) -> FrontierProxies:
    """Constitution §15 new-frontier block.

    We compute only the time-since-introduction guard, plus a count
    of new segments added in the last 5 fiscal years. The substantive
    judgment — imitation evidence, analyst recognition — is deferred
    to Stage 3 LLM per §17.
    """
    if not funds.segments_history:
        return FrontierProxies(
            years_since_first_segment_introduction=None,
            new_segments_added_5y=None,
        )

    # Earliest fiscal year for which we have segment data — reading
    # the history conservatively as "segments existed at this point."
    # Without finer-grained launch dates, this is the best proxy.
    earliest_year = min(s.fiscal_year for s in funds.segments_history)
    latest_year = max(s.fiscal_year for s in funds.segments_history)
    years_since = latest_year - earliest_year

    # Count of segment names that appear in any of the last 5 fiscal
    # years and DON'T appear in earlier years.
    cutoff_year = latest_year - 5
    early_names = {
        seg.name
        for sb in funds.segments_history
        if sb.fiscal_year <= cutoff_year
        for seg in sb.all_segments
    }
    recent_names = {
        seg.name
        for sb in funds.segments_history
        if sb.fiscal_year > cutoff_year
        for seg in sb.all_segments
    }
    new_segments_added = len(recent_names - early_names)

    return FrontierProxies(
        years_since_first_segment_introduction=years_since,
        new_segments_added_5y=new_segments_added,
    )


# ---------------------------------------------------------------------------
# Bottleneck proxies
# ---------------------------------------------------------------------------


def _hhi_from_share_distribution(shares: list[float]) -> int | None:
    """Herfindahl index on a share distribution. Shares should be in
    [0, 1] and need not sum to 1 (the residual is treated as a long
    tail of small competitors). Returns None when the input is empty.

    Output is scaled to 0-10000 (square of percentages, US convention).
    """
    if not shares:
        return None
    total = sum((s * 100.0) ** 2 for s in shares)
    return int(round(total))


def compute_bottleneck_proxies(funds: TickerFundamentals) -> BottleneckProxies:
    """Constitution §15 bottleneck block."""
    hhi: int | None = None
    if funds.top5_customer_share is not None:
        # Without per-customer breakdown we approximate HHI on the
        # top-5 share by assuming uniform distribution within the top
        # five. Crude, but the absolute number isn't what matters —
        # the §15 threshold (>=2500) is a signal of "highly
        # concentrated," and a 40% top-5 share approximated this way
        # already lands at HHI 320, well below 2500. So this is mostly
        # a placeholder; Stage 3 should refine when per-customer data
        # is available.
        per_customer = funds.top5_customer_share / 5.0
        hhi = _hhi_from_share_distribution([per_customer] * 5)

    return BottleneckProxies(
        top5_customer_share=funds.top5_customer_share,
        hhi=hhi,
        diversification_attempt_signals=funds.diversification_attempt_signals,
    )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def compute_all_proxies(
    funds: TickerFundamentals,
) -> tuple[MoatProxies, FrontierProxies, BottleneckProxies]:
    """Compute proxies for every axis in one call. Used by prefilter
    when it doesn't care which axis is being inspected.
    """
    return (
        compute_moat_proxies(funds),
        compute_frontier_proxies(funds),
        compute_bottleneck_proxies(funds),
    )


__all__ = [
    "compute_all_proxies",
    "compute_bottleneck_proxies",
    "compute_frontier_proxies",
    "compute_moat_proxies",
]
