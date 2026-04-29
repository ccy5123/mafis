"""Stage 3 prompt-builder tests.

The point of these tests is to catch DRIFT between the constitution
text in `docs/constitution.md` and what the prompts surface to the
LLM. If the constitution is edited (with a version bump), the
prompts must be re-validated; these tests fail loudly when key
phrases go missing.
"""

from __future__ import annotations

import pytest

from wise_investor.screening.stage3_prompts import (
    CONSTITUTION_PATH,
    build_stage3_prompt,
    extract_axis_section,
)
from wise_investor.screening.types import (
    AxisVerdict,
    PrefilterResult,
    SegmentBreakdown,
    Segment,
    TickerFundamentals,
)


# ---------------------------------------------------------------------------
# Constitution availability
# ---------------------------------------------------------------------------


def test_constitution_file_exists() -> None:
    assert CONSTITUTION_PATH.exists(), (
        "docs/constitution.md is required for Stage 3 prompt assembly. "
        "If you removed it, you broke the screening pipeline."
    )


# ---------------------------------------------------------------------------
# extract_axis_section — robustness against constitution edits
# ---------------------------------------------------------------------------


def test_extract_moat_section_contains_definition_and_pass_conditions() -> None:
    section = extract_axis_section("moat")
    # Definition sentence — must remain extractable for the prompt.
    assert "structural reason" in section.lower()
    assert "3 years" in section or "three" in section.lower()
    # Pass conditions are the heart of the screen.
    assert "Pass conditions" in section
    # Auto-PASS conditions guard against the most common bull traps.
    assert "Auto-PASS" in section


def test_extract_new_frontier_section_carries_imitation_requirement() -> None:
    section = extract_axis_section("new_frontier")
    # The single most important word — without imitation evidence
    # the axis collapses into hype. If this assertion fails, the
    # constitution edit dropped the constraint and the screen is
    # silently broken.
    assert "imitat" in section.lower()


def test_extract_bottleneck_section_requires_replacement_difficulty() -> None:
    section = extract_axis_section("bottleneck")
    assert "replac" in section.lower()  # replaceability test
    assert "1-2 years" in section or "1 to 2 years" in section.lower()


def test_extract_unknown_axis_raises() -> None:
    with pytest.raises(ValueError):
        extract_axis_section("not_a_real_axis")


# ---------------------------------------------------------------------------
# Full prompt assembly
# ---------------------------------------------------------------------------


def _stub_funds() -> TickerFundamentals:
    return TickerFundamentals(
        symbol="STUB",
        industry_classification="Test Sub-Industry",
        annual=(),
        quarterly_margins=(),
        segments_history=(),
        top5_customer_share=0.5,
        diversification_attempt_signals=0,
        industry_roic_3y_median=0.10,
        industry_gross_margin_3y_std=0.02,
    )


def _stub_prefilter(symbol: str = "STUB") -> PrefilterResult:
    return PrefilterResult(
        symbol=symbol,
        constitution_version="2.0",
        moat=AxisVerdict(
            axis="moat",
            verdict="NEED_LLM",
            reason="quant clean; needs Stage 3 to verify structural reason",
            details={
                "roic_3y_avg": 0.22,
                "roic_advantage": 0.12,
                "roic_advantage_trend": 0.005,
                "gross_margin_3y_std": 0.015,
                "gross_margin_industry_ratio": 0.75,
                "customer_concentration_trend": None,
            },
        ),
        new_frontier=AxisVerdict(
            axis="new_frontier",
            verdict="NEED_LLM",
            reason="time threshold satisfied",
            details={
                "years_since_first_segment_introduction": 8,
                "new_segments_added_5y": 2,
            },
        ),
        bottleneck=AxisVerdict(
            axis="bottleneck",
            verdict="NEED_LLM",
            reason="customer concentration above threshold",
            details={
                "top5_customer_share": 0.55,
                "diversification_attempt_signals": 0,
            },
        ),
        primary_segment=SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Data Center",
            primary_segment_revenue_share=0.60,
            all_segments=(Segment(name="Data Center", revenue=None, share_of_total=0.60),),
            fiscal_year=2024,
            source="stub",
        ),
        excluded_reason=None,
        hierarchy_decision="ADVANCE_TO_STAGE_3",
        passed_axes=tuple(),
        need_llm_axes=("moat", "new_frontier", "bottleneck"),
    )


def test_prompt_includes_all_three_axis_definitions() -> None:
    funds = _stub_funds()
    prefilter = _stub_prefilter()
    prompt = build_stage3_prompt(funds, prefilter)
    assert "AXIS 1 — MOAT" in prompt
    assert "AXIS 2 — NEW FRONTIER" in prompt
    assert "AXIS 3 — BOTTLENECK" in prompt


def test_prompt_includes_quant_proxies() -> None:
    funds = _stub_funds()
    prefilter = _stub_prefilter()
    prompt = build_stage3_prompt(funds, prefilter)
    # ROIC numbers from the moat block surface in the prompt.
    assert "0.2200" in prompt or "0.22" in prompt
    # Bottleneck top-5 share also surfaces.
    assert "0.55" in prompt


def test_prompt_includes_stage2_verdicts() -> None:
    """Stage 2's verdicts are passed through as framing context."""
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "NEED_LLM" in prompt


def test_prompt_includes_precision_over_recall_reminder() -> None:
    """Commitment 3 reminder must be in every prompt."""
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "FAIL the axis" in prompt
    assert "miss some good companies" in prompt


def test_prompt_specifies_json_output_format() -> None:
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "OUTPUT FORMAT" in prompt
    assert '"verdict"' in prompt
    assert '"hierarchy_decision"' in prompt


def test_prompt_states_caller_recomputes_hierarchy() -> None:
    """The prompt must tell the LLM that its hierarchy_decision is
    informational only — the gate is rule-based on the caller side.
    """
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "recompute" in prompt or "informational" in prompt


def test_prompt_includes_industry_classification() -> None:
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "Test Sub-Industry" in prompt


def test_prompt_includes_primary_segment_label() -> None:
    prompt = build_stage3_prompt(_stub_funds(), _stub_prefilter())
    assert "Data Center" in prompt
