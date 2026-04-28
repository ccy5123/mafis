"""Stage 2 integration logic — §16.

Orchestrates: segment resolution → proxy computation → per-axis
verdict (PASS / FAIL / NEED_LLM) → hierarchy gate (§9). Output is a
single `PrefilterResult` per ticker.

The constitution's precision-over-recall commitment shapes most of
the choices here: when proxies are available and clearly fail a
threshold we emit FAIL; when proxies are available and clearly meet
the threshold we emit PASS; when proxies are missing or only
partially conclusive we emit NEED_LLM and let Stage 3 decide. We do
NOT default to PASS when uncertain.

The hierarchy gate (constitution §9): two axes or more must pass, AND
a growth axis (new_frontier or bottleneck) must be among them. We
treat NEED_LLM as "potential pass" at this gate — i.e. a candidate
advances to Stage 3 if the *worst-case* scenario (NEED_LLM resolving
to PASS) would satisfy the hierarchy. Stage 3 then makes the final
call. This keeps the quantitative gate from rejecting candidates that
the LLM would have approved.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.proxies import (
    compute_bottleneck_proxies,
    compute_frontier_proxies,
    compute_moat_proxies,
)
from wise_investor.screening.types import (
    AxisVerdict,
    BottleneckProxies,
    FrontierProxies,
    HierarchyDecision,
    MoatProxies,
    PrefilterResult,
    SegmentBreakdown,
    TickerFundamentals,
)


# ---------------------------------------------------------------------------
# Constitution §15 thresholds — kept as module constants so calibration
# can override them in one place if §24 evidence later recommends change.
# ---------------------------------------------------------------------------


MOAT_ROIC_ADVANTAGE_MIN: float = 0.05  # 5pp
MOAT_ROIC_TREND_MIN: float = -0.005  # less than 0.5pp/year erosion
MOAT_GM_VARIABILITY_RATIO_MAX: float = 1.2  # own_std ≤ 1.2× industry_std

FRONTIER_MIN_YEARS_SINCE_INTRO: int = 3

BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN: float = 0.40  # 40%
BOTTLENECK_HHI_MIN: int = 2500


# ---------------------------------------------------------------------------
# Per-axis evaluation
# ---------------------------------------------------------------------------


def _evaluate_moat(proxies: MoatProxies) -> AxisVerdict:
    details = asdict(proxies)

    # Auto-PASS 1 (constitution §10): insufficient ROIC history.
    if proxies.roic_3y_avg is None:
        return AxisVerdict(
            axis="moat",
            verdict="FAIL",
            reason="auto-PASS 1: ROIC data spans <3 years (insufficient history)",
            details=details,
        )

    # Industry comparison missing → cannot judge advantage; defer to
    # Stage 3 to assess whether structural reason still applies.
    if proxies.roic_advantage is None:
        return AxisVerdict(
            axis="moat",
            verdict="NEED_LLM",
            reason="ROIC computed but industry median unavailable for comparison",
            details=details,
        )

    if proxies.roic_advantage < MOAT_ROIC_ADVANTAGE_MIN:
        return AxisVerdict(
            axis="moat",
            verdict="FAIL",
            reason=(
                f"ROIC advantage {proxies.roic_advantage:.3f} < "
                f"{MOAT_ROIC_ADVANTAGE_MIN:.2f} (5pp persistence threshold)"
            ),
            details=details,
        )

    if (
        proxies.roic_advantage_trend is not None
        and proxies.roic_advantage_trend < MOAT_ROIC_TREND_MIN
    ):
        return AxisVerdict(
            axis="moat",
            verdict="FAIL",
            reason=(
                f"ROIC advantage eroding at {proxies.roic_advantage_trend:.4f}/yr "
                f"(threshold {MOAT_ROIC_TREND_MIN:.4f})"
            ),
            details=details,
        )

    # Margin variability check (gate, not block-PASS) — high variability
    # weakens the moat case but the structural reason check is qualitative.
    if (
        proxies.gross_margin_industry_ratio is not None
        and proxies.gross_margin_industry_ratio > MOAT_GM_VARIABILITY_RATIO_MAX
    ):
        return AxisVerdict(
            axis="moat",
            verdict="NEED_LLM",
            reason=(
                f"ROIC advantage looks structural but margin volatility "
                f"({proxies.gross_margin_industry_ratio:.2f}× industry) flags "
                "possible cyclical artifact — Stage 3 must inspect"
            ),
            details=details,
        )

    # Quantitative side passes; structural-reason classification (§10
    # condition 2) and erosion-threat assessment (§10 condition 3) are
    # qualitative — Stage 3 owns them.
    return AxisVerdict(
        axis="moat",
        verdict="NEED_LLM",
        reason=(
            "Quant proxies consistent with moat candidacy; Stage 3 must "
            "verify structural reason (4-buckets) + erosion threat"
        ),
        details=details,
    )


def _evaluate_frontier(proxies: FrontierProxies) -> AxisVerdict:
    details = asdict(proxies)

    if proxies.years_since_first_segment_introduction is None:
        # No segment history at all — let Stage 3 try to date the
        # paradigm from external evidence (analyst reports, news).
        return AxisVerdict(
            axis="new_frontier",
            verdict="NEED_LLM",
            reason="No segment history available to date the paradigm",
            details=details,
        )

    # Constitution Auto-PASS 4: < 3 years from paradigm intro.
    if (
        proxies.years_since_first_segment_introduction
        < FRONTIER_MIN_YEARS_SINCE_INTRO
    ):
        return AxisVerdict(
            axis="new_frontier",
            verdict="FAIL",
            reason=(
                f"auto-PASS 4: only "
                f"{proxies.years_since_first_segment_introduction} year(s) "
                f"since paradigm introduction (<{FRONTIER_MIN_YEARS_SINCE_INTRO})"
            ),
            details=details,
        )

    # Time threshold satisfied — substantive judgment is qualitative.
    return AxisVerdict(
        axis="new_frontier",
        verdict="NEED_LLM",
        reason=(
            "Time threshold satisfied; Stage 3 must verify imitation "
            "evidence and external analyst recognition"
        ),
        details=details,
    )


def _evaluate_bottleneck(proxies: BottleneckProxies) -> AxisVerdict:
    details = asdict(proxies)

    # Diversification signals → constitution Auto-PASS 4 trigger.
    if proxies.diversification_attempt_signals > 0:
        return AxisVerdict(
            axis="bottleneck",
            verdict="NEED_LLM",
            reason=(
                f"{proxies.diversification_attempt_signals} diversification "
                "attempt signal(s) detected — Stage 3 must judge whether "
                "this represents auto-PASS 4 erosion"
            ),
            details=details,
        )

    if proxies.top5_customer_share is None:
        # No customer concentration data — the bottleneck rule
        # (§12 condition 1) requires either a 5x downstream dependency
        # ratio (LLM-side) or a 40% top-5 share (quant-side). With
        # neither, we can only delegate.
        return AxisVerdict(
            axis="bottleneck",
            verdict="NEED_LLM",
            reason=(
                "Customer concentration undisclosed; Stage 3 must verify "
                "downstream dependency from 10-K Risk Factors or other "
                "qualitative evidence"
            ),
            details=details,
        )

    if proxies.top5_customer_share < BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN:
        return AxisVerdict(
            axis="bottleneck",
            verdict="FAIL",
            reason=(
                f"top-5 customer share {proxies.top5_customer_share:.2f} < "
                f"{BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN:.2f} threshold"
            ),
            details=details,
        )

    # Quant side OK; replacement difficulty (§12 condition 2) and
    # division-of-labor verification are qualitative.
    return AxisVerdict(
        axis="bottleneck",
        verdict="NEED_LLM",
        reason=(
            "Customer concentration consistent with downstream dependency; "
            "Stage 3 must verify replacement difficulty and division-of-"
            "labor character"
        ),
        details=details,
    )


# ---------------------------------------------------------------------------
# Hierarchy gate (constitution §9)
# ---------------------------------------------------------------------------


def _potential_pass(verdict: AxisVerdict) -> bool:
    """At Stage 2, NEED_LLM advances forward — it's the LLM's job in
    Stage 3 to resolve the ambiguity. Only an outright FAIL stops a
    candidate at this gate.
    """
    return verdict.verdict in ("PASS", "NEED_LLM")


def _apply_hierarchy_gate(
    moat: AxisVerdict, frontier: AxisVerdict, bottleneck: AxisVerdict
) -> tuple[HierarchyDecision, list[str], list[str], str | None]:
    """Constitution §9: two axes or more must pass, AND a growth axis
    (new_frontier or bottleneck) must be among them.

    Returns: (decision, passed_axes, need_llm_axes, reject_reason).
    """
    pots: list[tuple[str, AxisVerdict]] = [
        ("moat", moat),
        ("new_frontier", frontier),
        ("bottleneck", bottleneck),
    ]
    potential = [name for name, v in pots if _potential_pass(v)]
    growth_potential = any(name in ("new_frontier", "bottleneck") for name in potential)

    passed = [name for name, v in pots if v.verdict == "PASS"]
    need_llm = [name for name, v in pots if v.verdict == "NEED_LLM"]

    if len(potential) >= 2 and growth_potential:
        return ("ADVANCE_TO_STAGE_3", passed, need_llm, None)

    reasons: list[str] = []
    if len(potential) < 2:
        reasons.append(
            f"only {len(potential)} axis(es) cleared the quant gate "
            "(need 2+)"
        )
    if not growth_potential:
        reasons.append(
            "no growth axis (new_frontier or bottleneck) cleared the gate"
        )
    return ("REJECT", passed, need_llm, "; ".join(reasons))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_ticker(
    funds: TickerFundamentals,
    primary_segment: SegmentBreakdown | None,
) -> PrefilterResult:
    """Run the full Stage 2 evaluation on one ticker.

    `primary_segment` is the result of the §13 multi-segment 30%
    rule (typically from `screening.segments.resolve_primary_segment`).
    When `primary_segment.primary_segment_exists` is False, the
    ticker is excluded from the universe BEFORE axis evaluation —
    we don't even compute proxies, since the conglomerate exclusion
    is a hard gate.
    """
    # Multi-segment 30% gate.
    if primary_segment is None or not primary_segment.primary_segment_exists:
        sentinel = AxisVerdict(
            axis="moat",
            verdict="FAIL",
            reason="excluded: no primary segment ≥30% (constitution §13)",
            details={},
        )
        return PrefilterResult(
            symbol=funds.symbol,
            constitution_version=CONSTITUTION_VERSION,
            moat=sentinel,
            new_frontier=AxisVerdict(
                axis="new_frontier",
                verdict="FAIL",
                reason="excluded: no primary segment ≥30%",
                details={},
            ),
            bottleneck=AxisVerdict(
                axis="bottleneck",
                verdict="FAIL",
                reason="excluded: no primary segment ≥30%",
                details={},
            ),
            primary_segment=primary_segment,
            excluded_reason="constitution §13: no segment ≥30% revenue share",
            hierarchy_decision="REJECT",
            passed_axes=tuple(),
            need_llm_axes=tuple(),
        )

    moat = _evaluate_moat(compute_moat_proxies(funds))
    frontier = _evaluate_frontier(compute_frontier_proxies(funds))
    bottleneck = _evaluate_bottleneck(compute_bottleneck_proxies(funds))

    decision, passed, need_llm, reject_reason = _apply_hierarchy_gate(
        moat, frontier, bottleneck
    )

    return PrefilterResult(
        symbol=funds.symbol,
        constitution_version=CONSTITUTION_VERSION,
        moat=moat,
        new_frontier=frontier,
        bottleneck=bottleneck,
        primary_segment=primary_segment,
        excluded_reason=reject_reason if decision == "REJECT" else None,
        hierarchy_decision=decision,
        passed_axes=tuple(passed),
        need_llm_axes=tuple(need_llm),
    )


def evaluate_universe(
    items: Iterable[tuple[TickerFundamentals, SegmentBreakdown | None]],
) -> list[PrefilterResult]:
    """Apply `evaluate_ticker` to every (fundamentals, segment) pair.
    Convenience for batch screening; ordering of the output matches
    the input.
    """
    return [evaluate_ticker(funds, seg) for funds, seg in items]


__all__ = [
    "BOTTLENECK_HHI_MIN",
    "BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN",
    "FRONTIER_MIN_YEARS_SINCE_INTRO",
    "MOAT_GM_VARIABILITY_RATIO_MAX",
    "MOAT_ROIC_ADVANTAGE_MIN",
    "MOAT_ROIC_TREND_MIN",
    "evaluate_ticker",
    "evaluate_universe",
]
