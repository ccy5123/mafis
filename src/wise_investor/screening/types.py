"""Data structures shared across the Stage 2 pre-filter modules.

All structures are immutable dataclasses with explicit field types so
downstream consumers (the prefilter integration logic, the JSON
serializer, future test fixtures) can rely on the shape staying
stable across versions.

Naming convention: a `*_share` field is a fraction in [0, 1]. A
`*_pct` field would be the same number scaled to 0-100; we don't use
that convention here. Stick with shares to avoid arithmetic mistakes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Segment data — feeds the §13 multi-segment 30% rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One reportable business segment for a single fiscal year."""

    name: str
    revenue: float | None  # absolute, in reporting currency; None when undisclosed
    share_of_total: float  # 0.0 to 1.0


@dataclass(frozen=True)
class SegmentBreakdown:
    """Resolved segment picture for a single ticker on a single fiscal year.

    `primary_segment_exists` is True if any segment crosses the §13
    30% threshold. When False, the company is too diversified for
    clean axis evaluation and should be excluded from the universe
    before Stage 3.
    """

    primary_segment_exists: bool
    primary_segment_name: str | None
    primary_segment_revenue_share: float | None
    all_segments: tuple[Segment, ...]
    fiscal_year: int
    source: str  # "finnhub" | "fmp" | "dart" | "single_segment_default" | "stub"


# ---------------------------------------------------------------------------
# Fundamentals input — the uniform shape adapters target
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnnualFinancials:
    """One fiscal year of financial statement essentials."""

    fiscal_year: int
    revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    nopat: float | None  # net operating profit after tax (compute or fetch)
    invested_capital: float | None  # debt + equity - cash, or equivalent
    rd_expense: float | None


@dataclass(frozen=True)
class QuarterlyMargin:
    """Single-quarter gross margin for variability checks."""

    quarter_id: str  # "2024Q3" or "FY24Q3" — caller's convention
    gross_margin: float | None  # 0.0 to 1.0


@dataclass(frozen=True)
class TickerFundamentals:
    """The complete input for proxy computation on a single ticker.

    Adapters populate this from their data source. Proxies module
    consumes it without caring about the source.
    """

    symbol: str
    industry_classification: str  # GICS Level 3 sub-industry name or code
    annual: tuple[AnnualFinancials, ...]  # newest fiscal year last
    quarterly_margins: tuple[QuarterlyMargin, ...]  # last 12 quarters preferred
    segments_history: tuple[SegmentBreakdown, ...]  # one per fiscal year
    top5_customer_share: float | None  # may be None if undisclosed
    diversification_attempt_signals: int  # count of recent events; 0 if none

    # Industry comparison data — populated by the adapter or a separate
    # peer-aggregation step before proxy computation.
    industry_roic_3y_median: float | None
    industry_gross_margin_3y_std: float | None


# ---------------------------------------------------------------------------
# Proxy results — the output of pure proxy computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoatProxies:
    """Quantitative slice for the moat axis (constitution §15)."""

    roic_3y_avg: float | None
    roic_advantage: float | None  # roic_3y_avg - industry_roic_3y_median
    roic_advantage_trend: float | None  # slope of yearly advantage; pp/year
    gross_margin_3y_std: float | None
    gross_margin_industry_ratio: float | None  # ratio of own_std to industry_std
    customer_concentration_trend: float | None  # change in top-N share; None if undisclosed


@dataclass(frozen=True)
class FrontierProxies:
    """Quantitative slice for the new-frontier axis (constitution §15).

    Most of this axis is LLM-driven by design (§17). Quantitative
    inputs only auto-reject on the time threshold and surface the
    other signals as Stage-3 LLM context.
    """

    years_since_first_segment_introduction: int | None
    new_segments_added_5y: int | None


@dataclass(frozen=True)
class BottleneckProxies:
    """Quantitative slice for the bottleneck axis (constitution §15)."""

    top5_customer_share: float | None
    hhi: int | None  # Herfindahl on customer concentration; 0-10000
    diversification_attempt_signals: int


# ---------------------------------------------------------------------------
# Per-axis verdict — fed to the §16 integration logic
# ---------------------------------------------------------------------------


AxisVerdictKind = Literal["PASS", "FAIL", "NEED_LLM"]


@dataclass(frozen=True)
class AxisVerdict:
    """Quantitative verdict on a single axis.

    `reason` is short enough to log; `details` carries full computed
    proxies for downstream stages and audit.
    """

    axis: Literal["moat", "new_frontier", "bottleneck"]
    verdict: AxisVerdictKind
    reason: str
    details: dict  # serializable representation of the underlying proxies


# ---------------------------------------------------------------------------
# Pre-filter result — the public output of the screening package
# ---------------------------------------------------------------------------


HierarchyDecision = Literal["ADVANCE_TO_STAGE_3", "REJECT"]


@dataclass(frozen=True)
class PrefilterResult:
    """Stage 2 output for one ticker.

    `hierarchy_decision` is the gate that either advances the ticker
    to the LLM stage (Stage 3) or rejects outright. `passed_axes`
    lists axes that passed the quantitative cut; `need_llm_axes` lists
    axes the LLM must judge (typically all New Frontier passes go
    here because it's primarily LLM-driven by design).
    """

    symbol: str
    constitution_version: str
    moat: AxisVerdict
    new_frontier: AxisVerdict
    bottleneck: AxisVerdict
    primary_segment: SegmentBreakdown | None  # None when the ticker is excluded
    excluded_reason: str | None  # set when hierarchy_decision == REJECT pre-axis
    hierarchy_decision: HierarchyDecision
    passed_axes: tuple[str, ...]
    need_llm_axes: tuple[str, ...]


__all__ = [
    "AnnualFinancials",
    "AxisVerdict",
    "AxisVerdictKind",
    "BottleneckProxies",
    "FrontierProxies",
    "HierarchyDecision",
    "MoatProxies",
    "PrefilterResult",
    "QuarterlyMargin",
    "Segment",
    "SegmentBreakdown",
    "TickerFundamentals",
]
