"""Constitution v2.0 Skeptic prompt tests (§19)."""

from __future__ import annotations

from wise_investor.agents.v2.attack_distribution import distribute_skeptic_attacks
from wise_investor.agents.v2.skeptic import (
    make_v2_skeptic_system_prompt,
    make_v2_skeptic_user_prompt,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_system_prompt_states_axis_alignment_rule() -> None:
    """The most important behavior change vs legacy — make sure it's
    in the prompt.
    """
    text = make_v2_skeptic_system_prompt()
    assert "AXIS-ALIGNED ATTACKS" in text or "axis-aligned" in text.lower()


def test_system_prompt_forbids_legacy_failure_modes() -> None:
    """Legacy Skeptics drifted into macro hand-waving and hindsight
    reasoning. The v2 prompt forbids both explicitly.
    """
    text = make_v2_skeptic_system_prompt()
    assert "Generic macro" in text or "macro" in text.lower()
    assert "Hindsight" in text or "hindsight" in text.lower()


def test_system_prompt_keeps_universal_citation_rule() -> None:
    text = make_v2_skeptic_system_prompt()
    assert "[Source:" in text or "Universal Citation" in text or "[Source: ...]" in text


def test_system_prompt_keeps_refusal_phrase() -> None:
    """The refusal phrase is the load-bearing anti-fabrication tool."""
    text = make_v2_skeptic_system_prompt()
    assert "Downside not quantifiable from current facts" in text


# ---------------------------------------------------------------------------
# User prompt — 2 axis
# ---------------------------------------------------------------------------


def test_user_prompt_2_axis_lists_5_attacks_with_correct_distribution() -> None:
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA",
        plan=plan,
        value_chain_text="VC content",
        analyst_output="Analyst content",
        valuer_output="Valuer content",
    )
    # Five attack-number lines exist in the distribution block.
    assert out.count("Attack #") == 5
    # The bottleneck axis appears 3 times and moat 2 times in the
    # distribution lines (numbering says "target the **<axis>**").
    assert out.count("target the **bottleneck**") == 3
    assert out.count("target the **moat**") == 2
    # No new_frontier mentions in the distribution.
    assert "target the **new_frontier**" not in out


def test_user_prompt_2_axis_includes_only_relevant_attack_catalogs() -> None:
    """For a moat+bottleneck candidate, the prompt should NOT carry
    the new_frontier or overall_thesis catalogs — they're not
    actionable distractions.
    """
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert "MOAT ATTACK TYPES" in out
    assert "BOTTLENECK ATTACK TYPES" in out
    assert "NEW_FRONTIER ATTACK TYPES" not in out
    assert "OVERALL THESIS ATTACK TYPES" not in out


def test_user_prompt_2_axis_total_attacks_is_5() -> None:
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert "exactly 5 attacks" in out


# ---------------------------------------------------------------------------
# User prompt — 3 axis (7 attacks)
# ---------------------------------------------------------------------------


def test_user_prompt_3_axis_lists_7_attacks() -> None:
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert out.count("Attack #") == 7
    assert out.count("target the **bottleneck**") == 2
    assert out.count("target the **new_frontier**") == 2
    assert out.count("target the **moat**") == 2
    assert out.count("target the **overall_thesis**") == 1


def test_user_prompt_3_axis_includes_overall_thesis_catalog() -> None:
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert "OVERALL THESIS ATTACK TYPES" in out


def test_user_prompt_3_axis_total_attacks_is_7() -> None:
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert "exactly 7 attacks" in out


# ---------------------------------------------------------------------------
# Output format requirements
# ---------------------------------------------------------------------------


def test_user_prompt_includes_attack_format_template() -> None:
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    # The prompt provides the structured rebuttal shape (target / assumption / etc.).
    assert "Target claim" in out
    assert "Assumption under attack" in out
    assert "Counter-evidence" in out
    assert "Downside quantification" in out


def test_user_prompt_demands_axis_tag_in_attack_heading() -> None:
    """The Defender + audit need axis tags to route labels per §20."""
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="VC", analyst_output="A", valuer_output="V",
    )
    assert "[axis: <axis_name>]" in out


def test_user_prompt_includes_value_chain_text() -> None:
    plan = distribute_skeptic_attacks(["bottleneck", "moat"])
    out = make_v2_skeptic_user_prompt(
        symbol="NVDA", plan=plan,
        value_chain_text="UNIQUE_VC_MARKER",
        analyst_output="A", valuer_output="V",
    )
    assert "UNIQUE_VC_MARKER" in out
