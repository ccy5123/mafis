"""Constitution v2.0 Steward prompt tests (§21)."""

from __future__ import annotations

from wise_investor.agents.v2.steward import (
    make_v2_steward_system_prompt,
    make_v2_steward_user_prompt,
)


def test_system_prompt_states_binary_output_commitment() -> None:
    """Commitment 6: BUY or PASS, no conviction levels."""
    text = make_v2_steward_system_prompt()
    assert "BUY or PASS" in text or "binary BUY/PASS" in text or "binary BUY" in text
    # Conviction explicitly absent — neither word should appear in
    # a way that promises gradations.
    assert "Conviction:" not in text  # legacy "Conviction: N" line is gone


def test_system_prompt_states_no_position_sizing() -> None:
    """Position sizing is HRP's job, not Steward's."""
    text = make_v2_steward_system_prompt()
    assert "NO POSITION SIZING" in text or "do not propose" in text.lower()


def test_system_prompt_states_no_independent_judgment() -> None:
    """The Steward applies rules; it does NOT override them on
    narrative grounds (Commitment 5, hierarchy is fixed).
    """
    text = make_v2_steward_system_prompt()
    assert "NO INDEPENDENT JUDGMENT" in text or "don't override" in text.lower()


def test_system_prompt_describes_four_rules() -> None:
    text = make_v2_steward_system_prompt()
    assert "RULE 1" in text and "RULE 2" in text
    assert "RULE 3" in text and "RULE 4" in text


def test_user_prompt_passes_axis_list() -> None:
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "bottleneck" in out
    assert "moat" in out


def test_user_prompt_states_required_defended_ratio() -> None:
    """For 5-attack candidate, defended count must be ≥3 (3/5 ratio)."""
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "3/5" in out
    assert "≥3 attacks" in out or "≥ 3" in out or "3 attacks" in out


def test_user_prompt_states_required_defended_ratio_for_7() -> None:
    """For 7-attack candidate, ≥3/5 of 7 = 4.2 → integer floor is 4 by
    Python int division: int(7*3/5) = 4. So ≥4 defended.
    """
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat", "new_frontier"],
        n_total_attacks=7,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "≥4" in out or "4 attacks" in out or "3/5" in out


def test_user_prompt_describes_score_weighting() -> None:
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "DOWNGRADED" in out and "0.5" in out
    assert "FAILED" in out and "0.0" in out


def test_user_prompt_demands_json_only_output() -> None:
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "JSON" in out
    assert '"verdict"' in out and '"defended_ratio"' in out


def test_user_prompt_specifies_buy_vs_pass_conditions() -> None:
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D", audit_summary="X",
    )
    assert "All four rules satisfied" in out
    assert "rule failed" in out.lower()


def test_user_prompt_includes_audit_summary() -> None:
    out = make_v2_steward_user_prompt(
        symbol="NVDA",
        passed_axes_at_stage_3=["bottleneck", "moat"],
        n_total_attacks=5,
        skeptic_output="S", defender_output="D",
        audit_summary="UNIQUE_AUDIT_MARKER",
    )
    assert "UNIQUE_AUDIT_MARKER" in out
