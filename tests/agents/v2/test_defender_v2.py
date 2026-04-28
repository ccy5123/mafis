"""Constitution v2.0 Defender prompt tests (§20)."""

from __future__ import annotations

from wise_investor.agents.v2.defender import (
    make_v2_defender_system_prompt,
    make_v2_defender_user_prompt,
)


def test_system_prompt_states_strict_concede_rule() -> None:
    """The single largest behavior change vs legacy: weak defenses
    must CONCEDE rather than DEFEND with caveat.
    """
    text = make_v2_defender_system_prompt()
    assert "CONCEDED is honest" in text or "CONCEDE" in text


def test_system_prompt_forbids_softening() -> None:
    text = make_v2_defender_system_prompt()
    # The prompt must explicitly tell the LLM not to soften CONCEDED
    # with "but it's not really a problem because…"
    assert "soften" in text.lower() or "but it's not really" in text


def test_system_prompt_demands_axis_tag_preservation() -> None:
    """The Steward needs the axis tag echoed back to route labels
    per §21 RULE 3.
    """
    text = make_v2_defender_system_prompt()
    assert "AXIS TAG PRESERVATION" in text or "echo that tag" in text.lower()


def test_user_prompt_specifies_attack_count() -> None:
    """The Defender must emit exactly N labeled responses where N is
    set by the Skeptic's attack distribution.
    """
    out = make_v2_defender_user_prompt(
        symbol="NVDA",
        analyst_output="A", valuer_output="V", skeptic_output="S",
        n_total_attacks=7,
    )
    # The prompt mentions 7 (3-axis case) in multiple places.
    assert "7 numbered" in out or "exactly 7" in out


def test_user_prompt_requires_tally_line() -> None:
    """The audit + Steward consume the Tally line directly — formatting
    matters.
    """
    out = make_v2_defender_user_prompt(
        symbol="NVDA", analyst_output="A", valuer_output="V",
        skeptic_output="S", n_total_attacks=5,
    )
    assert "Tally:" in out
    assert "X DEFENDED, Y CONCEDED" in out


def test_user_prompt_describes_audit_downgrade_triggers() -> None:
    """A defense built on the wrong kind of citation gets downgraded
    by the audit; the prompt warns the LLM about this so it concedes
    pre-emptively rather than gambling on a tangential citation.
    """
    out = make_v2_defender_user_prompt(
        symbol="NVDA", analyst_output="A", valuer_output="V",
        skeptic_output="S", n_total_attacks=5,
    )
    assert "WHAT WILL FAIL THE AUDIT" in out
    assert "tangential" in out.lower() or "tangential" in out.lower()


def test_user_prompt_includes_skeptic_section() -> None:
    out = make_v2_defender_user_prompt(
        symbol="NVDA", analyst_output="A", valuer_output="V",
        skeptic_output="UNIQUE_SKEPTIC_MARKER", n_total_attacks=5,
    )
    assert "UNIQUE_SKEPTIC_MARKER" in out


def test_user_prompt_response_format_includes_axis_tag() -> None:
    out = make_v2_defender_user_prompt(
        symbol="NVDA", analyst_output="A", valuer_output="V",
        skeptic_output="S", n_total_attacks=5,
    )
    assert "[axis: <axis tag from Skeptic" in out
