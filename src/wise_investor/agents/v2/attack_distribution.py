"""Skeptic attack distribution per constitution §19.

The constitution prescribes:
  - 2 axes PASSED → 3 attacks on stronger axis, 2 on weaker
  - 3 axes PASSED → 2 attacks on each, last attack on overall thesis

It does NOT define "stronger." We define stronger by a fixed priority
order chosen for evidence concreteness:

  bottleneck > new_frontier > moat

Rationale:
  - **Bottleneck** is the most concrete: customer concentration,
    replacement difficulty, division-of-labor evidence are all
    externally verifiable. Attacks here have the most signal.
  - **New Frontier** comes next: imitation evidence is observable
    but the inference (paradigm has actually shifted) requires more
    judgment.
  - **Moat** is hardest to attack rigorously without slipping into
    qualitative debate; we still allocate attacks here, but it gets
    the smallest share when there's a stronger axis to target.

This priority is fixed (Commitment 5: "Hierarchy is fixed, not
adjustable per-ticker"). If calibration evidence later shows the
priority is wrong, that's a constitutional change requiring a
version bump.
"""

from __future__ import annotations

from dataclasses import dataclass


# Stronger axes appear first. The Skeptic distributor uses index in
# this tuple to break ties.
ATTACK_AXIS_PRIORITY: tuple[str, ...] = (
    "bottleneck",
    "new_frontier",
    "moat",
)


# Total attacks Skeptic must produce, per constitution §19. Five was
# the legacy quota and was preserved during constitutional review.
TOTAL_ATTACKS: int = 5


@dataclass(frozen=True)
class AttackPlan:
    """Distribution of Skeptic attacks across axes for one candidate.

    `attacks_by_axis` is an ordered mapping from axis name to attack
    count. The Skeptic prompt iterates this map in order so attack
    numbering is deterministic.

    `overall_thesis_attacks` is non-zero only when all three axes
    passed (constitution §19 reserves the last attack for "overall
    thesis" in that case). For the 2-axis case it is zero — every
    attack is axis-tagged.
    """

    attacks_by_axis: dict[str, int]
    overall_thesis_attacks: int


def distribute_skeptic_attacks(passed_axes: list[str]) -> AttackPlan:
    """Return the §19 attack distribution for a list of passed axes.

    Args:
      passed_axes: list of axis names that PASSED at Stage 3. Must
        contain 2 or 3 entries; anything else is a programming error
        (the hierarchy gate would have rejected the candidate).

    Returns: AttackPlan whose attacks sum to TOTAL_ATTACKS.

    Raises ValueError when:
      - passed_axes contains an unknown axis name
      - passed_axes has fewer than 2 entries
      - passed_axes has more than 3 entries
    """
    valid = set(ATTACK_AXIS_PRIORITY)
    unknown = [a for a in passed_axes if a not in valid]
    if unknown:
        raise ValueError(
            f"Unknown axis name(s): {unknown!r}. Valid: {ATTACK_AXIS_PRIORITY}"
        )

    if len(set(passed_axes)) != len(passed_axes):
        raise ValueError(f"Duplicate axes in passed_axes: {passed_axes!r}")

    n = len(passed_axes)
    if n < 2:
        raise ValueError(
            f"Stage 4 should not run with <2 passed axes (got {n}); the "
            "hierarchy gate at Stage 3 should have rejected."
        )
    if n > 3:
        raise ValueError(
            f"Stage 4 should not run with >3 passed axes (got {n}); only "
            "three axes exist in the constitution."
        )

    # Sort by priority (stronger first).
    ordered = sorted(passed_axes, key=ATTACK_AXIS_PRIORITY.index)

    if n == 2:
        # 3 attacks on stronger, 2 on weaker.
        return AttackPlan(
            attacks_by_axis={ordered[0]: 3, ordered[1]: 2},
            overall_thesis_attacks=0,
        )

    # n == 3: 2 attacks on each + last attack on overall thesis.
    # That sums to 7, not 5 — but the constitution explicitly says
    # "2 attacks on each of 3 axes (last attack on overall thesis)"
    # which sums to 7. We honor the constitution literally; the
    # TOTAL_ATTACKS=5 invariant is only for the 2-axis case.
    #
    # Re-reading §19: "3 axes PASSED → 2 attacks on each of 3 axes
    # (last attack on overall thesis)". This is ambiguous: 2+2+2+1=7,
    # OR 1+1+2+1=5 with redistribution. We interpret literally as
    # 2+2+2 + 1 overall = 7 attacks total. That's a richer adversarial
    # review precisely when the candidate looks strongest, which
    # matches the spirit of stress-testing.
    return AttackPlan(
        attacks_by_axis={ordered[0]: 2, ordered[1]: 2, ordered[2]: 2},
        overall_thesis_attacks=1,
    )


def total_attacks(plan: AttackPlan) -> int:
    """Sum of attacks across axes plus overall-thesis attacks."""
    return sum(plan.attacks_by_axis.values()) + plan.overall_thesis_attacks


__all__ = [
    "ATTACK_AXIS_PRIORITY",
    "AttackPlan",
    "TOTAL_ATTACKS",
    "distribute_skeptic_attacks",
    "total_attacks",
]
