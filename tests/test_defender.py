"""Tests for the Defender agent (Phase 2 debate round)."""

from __future__ import annotations

from wise_investor.agents.defender import (
    DEFENDER_BACKSTORY,
    DEFENDER_GOAL,
    defender_model,
    make_defender_system_prompt,
)
from wise_investor.agents.tasks import (
    DEFENDER_REPORT_TEMPLATE,
    make_defender_user_prompt,
    make_steward_user_prompt,
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def test_defender_system_prompt_names_both_labels() -> None:
    p = make_defender_system_prompt()
    assert "DEFENDED" in p
    assert "CONCEDED" in p


def test_defender_backstory_bans_speculative_language() -> None:
    # The list of banned phrases must be present so the model gets
    # explicit examples of what NOT to use as a defense.
    bans = [
        "could",
        "may",
        "should",
        "is working on",
        "is well",  # "is well[- ]positioned"
        "historical averages",
    ]
    for phrase in bans:
        assert phrase in DEFENDER_BACKSTORY, f"missing banned phrase: {phrase}"


def test_defender_model_defaults_to_analyst_model() -> None:
    # Defender shares Qwen with Analyst to avoid a model swap between
    # Skeptic (Llama) → Defender → Steward (Qwen).
    from wise_investor.config import settings

    assert defender_model() == settings.analyst_model


# ---------------------------------------------------------------------------
# Report template
# ---------------------------------------------------------------------------


def test_defender_template_requires_five_responses() -> None:
    assert "Exactly 5 responses" in DEFENDER_REPORT_TEMPLATE
    assert "DEFENDED or CONCEDED" in DEFENDER_REPORT_TEMPLATE


def test_defender_template_has_speculative_ban_section() -> None:
    assert "Speculative-defense ban" in DEFENDER_REPORT_TEMPLATE


def test_defender_template_requires_citations_on_defended() -> None:
    assert "Citation requirement" in DEFENDER_REPORT_TEMPLATE
    assert "[Source:" in DEFENDER_REPORT_TEMPLATE


def test_defender_template_mandates_closing_tally() -> None:
    assert "Closing tally" in DEFENDER_REPORT_TEMPLATE
    assert "Tally:" in DEFENDER_REPORT_TEMPLATE
    assert "X + Y = 5" in DEFENDER_REPORT_TEMPLATE


def test_defender_template_refusal_phrase_required_on_concession() -> None:
    # CONCEDED responses must end with the canonical refusal phrase so
    # the downstream Steward/audit can detect them reliably. The phrase
    # is wrapped across lines in the prompt markdown, so normalize
    # whitespace before comparing.
    import re

    normalized = re.sub(r"\s+", " ", DEFENDER_REPORT_TEMPLATE)
    assert "No concrete Bull counter-evidence available in current facts" in normalized


# ---------------------------------------------------------------------------
# User-prompt builder
# ---------------------------------------------------------------------------


def test_make_defender_user_prompt_injects_all_inputs() -> None:
    out = make_defender_user_prompt(
        symbol="NVDA",
        analyst_output="ANALYST_BODY",
        valuer_output="VALUER_BODY",
        skeptic_output="SKEPTIC_BODY",
    )
    assert "NVDA" in out
    assert "<analyst_section>" in out and "ANALYST_BODY" in out
    assert "<valuer_section>" in out and "VALUER_BODY" in out
    assert "<skeptic_section>" in out and "SKEPTIC_BODY" in out
    # Template must be appended at the end.
    assert "Exactly 5 responses" in out


def test_make_defender_user_prompt_uppercases_symbol() -> None:
    out = make_defender_user_prompt(
        symbol="nvda",
        analyst_output="a",
        valuer_output="v",
        skeptic_output="s",
    )
    assert "NVDA" in out


# ---------------------------------------------------------------------------
# Steward integration — Steward now takes optional defender_output
# ---------------------------------------------------------------------------


def test_steward_prompt_accepts_defender_output() -> None:
    out = make_steward_user_prompt(
        symbol="NVDA",
        value_chain_text="VC",
        analyst_output="A",
        valuer_output="V",
        skeptic_output="S",
        defender_output="D",
    )
    assert "<defender_section>" in out
    assert "D\n" in out


def test_steward_prompt_backwards_compatible_without_defender() -> None:
    # Legacy callers (5-agent scripts, older tests) must still work.
    out = make_steward_user_prompt(
        symbol="NVDA",
        value_chain_text="VC",
        analyst_output="A",
        valuer_output="V",
        skeptic_output="S",
    )
    # No defender_section block when defender output is empty.
    assert "<defender_section>" not in out


def test_steward_template_copies_defender_labels() -> None:
    # The Steward template now instructs "copy the Defender's label
    # VERBATIM" instead of judging for itself.
    from wise_investor.agents.tasks import STEWARD_REPORT_TEMPLATE

    assert "copy the Defender's label VERBATIM" in STEWARD_REPORT_TEMPLATE
    assert "DEFENDED" in STEWARD_REPORT_TEMPLATE
    assert "CONCEDED" in STEWARD_REPORT_TEMPLATE


# ---------------------------------------------------------------------------
# Goal text surfaced in system prompt
# ---------------------------------------------------------------------------


def test_defender_goal_mentions_concrete_evidence() -> None:
    assert "concrete" in DEFENDER_GOAL.lower()
    assert "CONCEDED" in DEFENDER_GOAL or "concede" in DEFENDER_GOAL.lower()
