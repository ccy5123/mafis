"""Skeptic attack-distribution tests (constitution §19)."""

from __future__ import annotations

import pytest

from wise_investor.agents.v2.attack_distribution import (
    ATTACK_AXIS_PRIORITY,
    distribute_skeptic_attacks,
    total_attacks,
)


# ---------------------------------------------------------------------------
# Priority contract — bottleneck > new_frontier > moat
# ---------------------------------------------------------------------------


def test_priority_order_is_bottleneck_first() -> None:
    """The fixed priority is the only thing that prevents stronger-axis
    designation from drifting per ticker (Commitment 5).
    """
    assert ATTACK_AXIS_PRIORITY == ("bottleneck", "new_frontier", "moat")


# ---------------------------------------------------------------------------
# 2-axis case: 3 attacks on stronger, 2 on weaker
# ---------------------------------------------------------------------------


def test_two_axes_bottleneck_and_moat_distribution() -> None:
    """Bottleneck is stronger → 3 attacks; moat → 2 attacks."""
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    assert plan.attacks_by_axis["bottleneck"] == 3
    assert plan.attacks_by_axis["moat"] == 2
    assert plan.overall_thesis_attacks == 0
    assert total_attacks(plan) == 5


def test_two_axes_input_order_does_not_matter() -> None:
    """Distribution depends on priority, not input order."""
    plan_a = distribute_skeptic_attacks(["moat", "bottleneck"])
    plan_b = distribute_skeptic_attacks(["bottleneck", "moat"])
    assert plan_a.attacks_by_axis == plan_b.attacks_by_axis


def test_two_axes_new_frontier_and_moat() -> None:
    plan = distribute_skeptic_attacks(["moat", "new_frontier"])
    assert plan.attacks_by_axis["new_frontier"] == 3
    assert plan.attacks_by_axis["moat"] == 2


def test_two_axes_bottleneck_and_new_frontier() -> None:
    plan = distribute_skeptic_attacks(["new_frontier", "bottleneck"])
    assert plan.attacks_by_axis["bottleneck"] == 3
    assert plan.attacks_by_axis["new_frontier"] == 2


# ---------------------------------------------------------------------------
# 3-axis case: 2 each + 1 overall thesis
# ---------------------------------------------------------------------------


def test_three_axes_distribution() -> None:
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    assert plan.attacks_by_axis["bottleneck"] == 2
    assert plan.attacks_by_axis["new_frontier"] == 2
    assert plan.attacks_by_axis["moat"] == 2
    assert plan.overall_thesis_attacks == 1
    assert total_attacks(plan) == 7


def test_three_axes_iteration_order_is_priority() -> None:
    """Insertion order in attacks_by_axis must be bottleneck →
    new_frontier → moat so the Skeptic prompt numbering is
    deterministic (#1-2 bottleneck, #3-4 new_frontier, #5-6 moat).
    """
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    keys_in_order = list(plan.attacks_by_axis.keys())
    assert keys_in_order == ["bottleneck", "new_frontier", "moat"]


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_unknown_axis_rejected() -> None:
    with pytest.raises(ValueError) as exc_info:
        distribute_skeptic_attacks(["moat", "vibes"])
    assert "Unknown axis" in str(exc_info.value)


def test_single_axis_rejected() -> None:
    """The hierarchy gate at Stage 3 ensures Stage 4 only runs with
    ≥2 axes; if it didn't, that's a programming error.
    """
    with pytest.raises(ValueError) as exc_info:
        distribute_skeptic_attacks(["moat"])
    assert "<2 passed axes" in str(exc_info.value)


def test_empty_axes_rejected() -> None:
    with pytest.raises(ValueError):
        distribute_skeptic_attacks([])


def test_more_than_three_axes_rejected() -> None:
    with pytest.raises(ValueError) as exc_info:
        distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck", "extra"])
    # The "Unknown axis" check runs first if "extra" is not a known
    # axis name; either error is acceptable for catching this.
    msg = str(exc_info.value)
    assert "Unknown axis" in msg or ">3 passed axes" in msg


def test_duplicate_axes_rejected() -> None:
    """Defense: passing the same axis twice should not happen — Stage
    3 returns deduped axes — but if the caller did, we want a clean
    error rather than silent over-counting.
    """
    with pytest.raises(ValueError) as exc_info:
        distribute_skeptic_attacks(["moat", "moat"])
    assert "Duplicate" in str(exc_info.value)
