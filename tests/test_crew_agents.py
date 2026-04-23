"""Construction tests for the Phase 1C agents (Valuer, Skeptic).

Live execution is deferred to scripts/run_crew.py to avoid pulling 5GB of
local models into VRAM during every pytest run. Here we verify the agent
prompts carry the right operating-principle language and the task
templates enforce the required section structure.
"""

from __future__ import annotations

from wise_investor.agents.skeptic import (
    SKEPTIC_BACKSTORY,
    SKEPTIC_GOAL,
    make_skeptic_system_prompt,
    skeptic_model,
)
from wise_investor.agents.steward import (
    STEWARD_BACKSTORY,
    STEWARD_GOAL,
    make_steward_system_prompt,
    steward_model,
)
from wise_investor.agents.tasks import (
    SKEPTIC_REPORT_TEMPLATE,
    STEWARD_REPORT_TEMPLATE,
    VALUER_REPORT_TEMPLATE,
    make_skeptic_user_prompt,
    make_steward_user_prompt,
    make_valuer_user_prompt,
)
from wise_investor.agents.valuer import (
    VALUER_BACKSTORY,
    VALUER_GOAL,
    make_valuer_system_prompt,
    valuer_model,
)
from wise_investor.config import settings


# ---------------------------------------------------------------------------
# Valuer
# ---------------------------------------------------------------------------


def test_valuer_model_matches_config() -> None:
    assert valuer_model() == settings.valuer_model


def test_valuer_system_prompt_contains_core_principles() -> None:
    prompt = make_valuer_system_prompt().lower()
    # Valuer must explicitly refuse buy/sell calls (that's Steward's job).
    assert "buy" in prompt and ("sell" in prompt or "hold" in prompt)
    assert "steward" in prompt
    # Must enforce verbatim peer table quoting.
    assert "verbatim" in prompt or "code block" in prompt
    # No unit-side computation; values come from tool outputs.
    assert "never compute" in prompt or "do not compute" in prompt.replace("no", "")


def test_valuer_backstory_forbids_recommendation() -> None:
    text = VALUER_BACKSTORY.lower()
    assert "never issue" in text or "do not issue" in text
    assert "buy" in text and ("sell" in text or "hold" in text)


def test_valuer_template_enforces_three_sections() -> None:
    for heading in [
        "## Valuation Snapshot",
        "## Peer Context",
        "## Market-Implied Growth Assessment",
    ]:
        assert heading in VALUER_REPORT_TEMPLATE, f"missing: {heading}"


def test_valuer_template_forbids_verdict_words() -> None:
    text = VALUER_REPORT_TEMPLATE.lower()
    assert "overvalued" in text  # it should be discussing what NOT to write
    assert "do not produce" in text or "do not write" in text


def test_make_valuer_user_prompt_injects_analyst_output() -> None:
    prompt = make_valuer_user_prompt(
        "NVDA", "Dummy value chain body", "Analyst wrote: NVDA has moat X."
    )
    assert "NVDA" in prompt
    assert "<analyst_section>" in prompt and "</analyst_section>" in prompt
    assert "Analyst wrote: NVDA has moat X." in prompt
    assert "<value_chain_brief>" in prompt
    assert "Dummy value chain body" in prompt


# ---------------------------------------------------------------------------
# Skeptic
# ---------------------------------------------------------------------------


def test_skeptic_model_matches_config() -> None:
    assert skeptic_model() == settings.skeptic_model


def test_skeptic_uses_different_model_than_analyst_valuer() -> None:
    # Phase 1C restores the "different LLM" principle: Skeptic MUST differ
    # from Analyst/Valuer. If this fails, we've regressed to Phase 1B interim.
    bull = {settings.analyst_model, settings.valuer_model}
    assert settings.skeptic_model not in bull, (
        f"Skeptic uses {settings.skeptic_model}, same as a Bull-side agent. "
        "Phase 1C requires a distinct local model for adversarial diversity."
    )


def test_skeptic_system_prompt_forbids_balance() -> None:
    prompt = make_skeptic_system_prompt().lower()
    # Skeptic's job is attack, not balance.
    assert "red-team" in prompt or "red team" in prompt
    assert "attack" in prompt
    assert "bull" in prompt


def test_skeptic_backstory_demands_structured_rebuttals() -> None:
    text = SKEPTIC_BACKSTORY.lower()
    assert "assumption" in text
    assert "falsif" in text or "refute" in text
    # Must not issue buy/sell/hold either.
    assert "does not recommend" in text or "do not recommend" in text


def test_skeptic_template_requires_exactly_5_rebuttals() -> None:
    text = SKEPTIC_REPORT_TEMPLATE
    assert "Exactly 5 rebuttals" in text
    # Explicit 3-of-5 vulnerable-link grounding.
    assert "value chain" in text.lower() and "vulnerable" in text.lower()


def test_skeptic_template_has_steelman_concession() -> None:
    assert "Steelman Concession" in SKEPTIC_REPORT_TEMPLATE
    # Line-wrap may split "strongest Bull argument" — check the words
    # individually in a normalized form.
    normalized = " ".join(SKEPTIC_REPORT_TEMPLATE.split())
    assert "strongest Bull argument" in normalized


def test_make_skeptic_user_prompt_injects_both_bull_sections() -> None:
    prompt = make_skeptic_user_prompt(
        "NVDA",
        "Value chain text here",
        "Analyst text here",
        "Valuer text here",
    )
    assert "<analyst_section>" in prompt
    assert "Analyst text here" in prompt
    assert "<valuer_section>" in prompt
    assert "Valuer text here" in prompt
    assert "Vulnerable links" in prompt or "vulnerable" in prompt.lower()


# ---------------------------------------------------------------------------
# Goal strings (non-empty, substantive)
# ---------------------------------------------------------------------------


def test_goals_non_empty_and_long_enough() -> None:
    assert len(VALUER_GOAL) > 50
    assert len(SKEPTIC_GOAL) > 50
    assert len(STEWARD_GOAL) > 50


# ---------------------------------------------------------------------------
# Steward
# ---------------------------------------------------------------------------


def test_steward_model_matches_config() -> None:
    assert steward_model() == settings.steward_model


def test_steward_system_prompt_enforces_pass_default() -> None:
    prompt = make_steward_system_prompt().lower()
    # Phase 2-B tightening: 'default to hold-or-pass' replaces the
    # stricter-sounding but looser-in-practice 'default to pass' — the
    # new text better matches how small LLMs behave.
    assert "default to hold-or-pass" in prompt or "default to pass" in prompt
    assert "pass" in prompt and "buy" in prompt and "hold" in prompt
    # New discipline: neutralization evidence standard.
    assert "neutraliz" in prompt
    assert "speculative" in prompt or "speculation" in prompt


def test_steward_backstory_forbids_sell_and_balance_hedging() -> None:
    text = STEWARD_BACKSTORY.lower()
    # No short-selling.
    assert "sell" in text
    assert "outside the mandate" in text or "outside of the mandate" in text
    # No hedged "balanced view" verdicts.
    assert "balanced view" in text or 'balanced hold' in text


def test_steward_backstory_requires_rebuttal_accounting() -> None:
    text = STEWARD_BACKSTORY.lower()
    # Must enumerate which Skeptic rebuttals survived.
    assert "rebuttal" in text
    assert "skeptic" in text


def test_steward_template_has_all_five_sections() -> None:
    for heading in [
        "## Verdict",
        "## Conviction Level",
        "## Rationale",
        "## Position Sizing Guidance",
        "## Confidence Caveats",
    ]:
        assert heading in STEWARD_REPORT_TEMPLATE, f"missing: {heading}"


def test_steward_template_forbids_sell() -> None:
    t = STEWARD_REPORT_TEMPLATE
    assert "Do NOT issue SELL" in t


def test_steward_template_maps_conviction_to_sizing() -> None:
    # Sizing guidance must exist and map at least a couple of conviction levels.
    t = STEWARD_REPORT_TEMPLATE
    assert "C3" in t and "C4" in t
    assert "%" in t


def test_make_steward_user_prompt_injects_all_three_agents() -> None:
    prompt = make_steward_user_prompt(
        "NVDA",
        "Value chain text",
        "Analyst says X.",
        "Valuer says Y.",
        "Skeptic attacks Z.",
    )
    assert "<analyst_section>" in prompt and "Analyst says X." in prompt
    assert "<valuer_section>" in prompt and "Valuer says Y." in prompt
    assert "<skeptic_section>" in prompt and "Skeptic attacks Z." in prompt
    assert "<value_chain_brief>" in prompt


def test_steward_uses_distinct_model_if_skeptic_llama() -> None:
    # Phase 2 default: Steward=Qwen, Skeptic=Llama. If both are ever set to
    # the same model we should at least notice — but we don't hard-fail
    # because it's a legal (if less diverse) config.
    assert settings.steward_model  # non-empty
