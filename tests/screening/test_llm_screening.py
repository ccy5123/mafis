"""Stage 3 orchestrator tests.

LLM is stubbed throughout — no Ollama / network calls. Calibration
(Step 4) is what exercises the real LLM end-to-end.
"""

from __future__ import annotations

import json

from wise_investor.screening.llm_screening import (
    _apply_hierarchy_gate,
    _extract_axis_outcome,
    _parse_response,
    screen_ticker,
)
from wise_investor.screening.types import (
    AxisVerdict,
    PrefilterResult,
    Segment,
    SegmentBreakdown,
    Stage3AxisOutcome,
    TickerFundamentals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _funds(symbol: str = "STUB") -> TickerFundamentals:
    return TickerFundamentals(
        symbol=symbol,
        industry_classification="Test Sub-Industry",
        annual=(),
        quarterly_margins=(),
        segments_history=(),
        top5_customer_share=0.5,
        diversification_attempt_signals=0,
        industry_roic_3y_median=0.10,
        industry_gross_margin_3y_std=0.02,
    )


def _advancing_prefilter(symbol: str = "STUB") -> PrefilterResult:
    return PrefilterResult(
        symbol=symbol,
        constitution_version="2.0",
        moat=AxisVerdict(
            axis="moat", verdict="NEED_LLM", reason="x",
            details={"roic_3y_avg": 0.22, "roic_advantage": 0.12,
                     "roic_advantage_trend": 0.005,
                     "gross_margin_3y_std": 0.015,
                     "gross_margin_industry_ratio": 0.75,
                     "customer_concentration_trend": None},
        ),
        new_frontier=AxisVerdict(
            axis="new_frontier", verdict="NEED_LLM", reason="x",
            details={"years_since_first_segment_introduction": 8,
                     "new_segments_added_5y": 2},
        ),
        bottleneck=AxisVerdict(
            axis="bottleneck", verdict="NEED_LLM", reason="x",
            details={"top5_customer_share": 0.55,
                     "diversification_attempt_signals": 0},
        ),
        primary_segment=SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Primary",
            primary_segment_revenue_share=0.60,
            all_segments=(Segment(name="Primary", revenue=None, share_of_total=0.60),),
            fiscal_year=2024,
            source="stub",
        ),
        excluded_reason=None,
        hierarchy_decision="ADVANCE_TO_STAGE_3",
        passed_axes=tuple(),
        need_llm_axes=("moat", "new_frontier", "bottleneck"),
    )


def _llm_stub(payload: str | dict):
    """Wrap a payload in a callable. Dict payload is JSON-serialized."""
    if isinstance(payload, dict):
        rendered = json.dumps(payload)
    else:
        rendered = payload

    def _call(prompt: str) -> str:
        return rendered

    return _call


def _verdict(verdict: str, qualifier=None, reasoning="") -> dict:
    return {"verdict": verdict, "reasoning": reasoning, "bucket": qualifier,
            "type": qualifier, "imitation_evidence": qualifier}


# ---------------------------------------------------------------------------
# Stage 2 passthrough — Stage 3 doesn't call LLM if Stage 2 rejected
# ---------------------------------------------------------------------------


def test_stage2_rejected_passthrough_does_not_call_llm() -> None:
    """If Stage 2 rejected, Stage 3 must not invoke the LLM."""
    rejected = PrefilterResult(
        symbol="X",
        constitution_version="2.0",
        moat=AxisVerdict(axis="moat", verdict="FAIL", reason="r", details={}),
        new_frontier=AxisVerdict(axis="new_frontier", verdict="FAIL", reason="r", details={}),
        bottleneck=AxisVerdict(axis="bottleneck", verdict="FAIL", reason="r", details={}),
        primary_segment=None,
        excluded_reason="Stage 2 said no",
        hierarchy_decision="REJECT",
        passed_axes=tuple(),
        need_llm_axes=tuple(),
    )

    def _should_not_be_called(prompt: str) -> str:
        raise AssertionError("LLM must not be invoked after Stage 2 rejection")

    result = screen_ticker(_funds("X"), rejected, llm_call=_should_not_be_called)
    assert result.hierarchy_decision == "REJECT"
    assert result.rejection_reason == "Stage 2 said no"
    # All axes are FAIL with "Stage 2 rejected" reasoning.
    assert result.moat.verdict == "FAIL"
    assert result.moat.reasoning.startswith("Stage 2")


# ---------------------------------------------------------------------------
# Happy paths — well-formed LLM output
# ---------------------------------------------------------------------------


def test_three_axis_pass_advances_to_stage_4() -> None:
    payload = {
        "moat": {"verdict": "PASS", "bucket": "intangible", "reasoning": "ok"},
        "new_frontier": {"verdict": "PASS", "imitation_evidence": ["AMD"], "reasoning": "ok"},
        "bottleneck": {"verdict": "PASS", "type": "technical", "reasoning": "ok"},
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",
        "rejection_reason": None,
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"
    assert result.rejection_reason is None
    assert result.moat.qualifier == "intangible"
    assert result.new_frontier.qualifier == "AMD"
    assert result.bottleneck.qualifier == "technical"


def test_moat_plus_growth_advances() -> None:
    """Moat + Bottleneck = allowed pair (constitution §9)."""
    payload = {
        "moat": {"verdict": "PASS", "bucket": "switching", "reasoning": "ok"},
        "new_frontier": {"verdict": "FAIL", "imitation_evidence": [], "reasoning": "no"},
        "bottleneck": {"verdict": "PASS", "type": "resource", "reasoning": "ok"},
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",
        "rejection_reason": None,
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"


def test_two_growth_axes_advances() -> None:
    """New Frontier + Bottleneck = allowed pair (constitution §9)."""
    payload = {
        "moat": {"verdict": "FAIL", "bucket": None, "reasoning": "no"},
        "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},
        "bottleneck": {"verdict": "PASS", "type": "regulatory", "reasoning": "ok"},
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",
        "rejection_reason": None,
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"


# ---------------------------------------------------------------------------
# Hierarchy gate REJECTs
# ---------------------------------------------------------------------------


def test_only_moat_pass_rejects_no_growth_axis() -> None:
    """Moat alone fails the gate (constitution §9 — needs growth)."""
    payload = {
        "moat": {"verdict": "PASS", "bucket": "cost", "reasoning": "ok"},
        "new_frontier": {"verdict": "FAIL", "imitation_evidence": [], "reasoning": "no"},
        "bottleneck": {"verdict": "FAIL", "type": None, "reasoning": "no"},
        "hierarchy_decision": "REJECT",
        "rejection_reason": "no growth",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "REJECT"
    assert result.rejection_reason is not None
    assert "growth" in result.rejection_reason


def test_single_axis_pass_rejects() -> None:
    """One axis pass < 2 → REJECT regardless of which axis."""
    payload = {
        "moat": {"verdict": "FAIL", "bucket": None, "reasoning": "no"},
        "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},
        "bottleneck": {"verdict": "FAIL", "type": None, "reasoning": "no"},
        "hierarchy_decision": "REJECT",
        "rejection_reason": "single axis",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "REJECT"


def test_all_fail_rejects() -> None:
    payload = {
        "moat": {"verdict": "FAIL", "bucket": None, "reasoning": "no"},
        "new_frontier": {"verdict": "FAIL", "imitation_evidence": [], "reasoning": "no"},
        "bottleneck": {"verdict": "FAIL", "type": None, "reasoning": "no"},
        "hierarchy_decision": "REJECT",
        "rejection_reason": "all fail",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.hierarchy_decision == "REJECT"


# ---------------------------------------------------------------------------
# LLM hierarchy_decision is NOT trusted — caller recomputes
# ---------------------------------------------------------------------------


def test_caller_overrides_llm_hierarchy_when_inconsistent() -> None:
    """LLM says ADVANCE but only one axis passed → caller recomputes
    REJECT and preserves LLM's claim in llm_reported_decision.
    """
    payload = {
        "moat": {"verdict": "PASS", "bucket": "cost", "reasoning": "ok"},
        "new_frontier": {"verdict": "FAIL", "imitation_evidence": [], "reasoning": "no"},
        "bottleneck": {"verdict": "FAIL", "type": None, "reasoning": "no"},
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",  # WRONG — only 1 pass, no growth
        "rejection_reason": None,
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    # Ground truth from rules (§9): REJECT.
    assert result.hierarchy_decision == "REJECT"
    # LLM's claim is preserved for audit.
    assert result.llm_reported_decision == "ADVANCE_TO_STAGE_4"


# ---------------------------------------------------------------------------
# Malformed output → INVALID per axis → REJECT
# ---------------------------------------------------------------------------


def test_unparseable_output_yields_reject_with_invalid_axes() -> None:
    result = screen_ticker(
        _funds(),
        _advancing_prefilter(),
        llm_call=_llm_stub("this is not json at all"),
    )
    assert result.hierarchy_decision == "REJECT"
    assert result.moat.verdict == "INVALID"
    assert result.new_frontier.verdict == "INVALID"
    assert result.bottleneck.verdict == "INVALID"
    assert result.raw_llm_output == "this is not json at all"


def test_missing_axis_block_yields_invalid_for_that_axis() -> None:
    payload = {
        "moat": {"verdict": "PASS", "bucket": "cost", "reasoning": "ok"},
        "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},
        # Bottleneck block missing entirely.
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.bottleneck.verdict == "INVALID"
    # Two passes + INVALID treated as fail → REJECT (no two real growth-aligned passes).
    # With moat PASS + new_frontier PASS that's 2 axes pass + new_frontier IS growth:
    # so hierarchy DOES advance. Confirm logic:
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"


def test_unknown_verdict_literal_treated_as_invalid() -> None:
    payload = {
        "moat": {"verdict": "MAYBE", "bucket": "intangible", "reasoning": "uncertain"},
        "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},
        "bottleneck": {"verdict": "PASS", "type": "technical", "reasoning": "ok"},
        "hierarchy_decision": "ADVANCE_TO_STAGE_4",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.moat.verdict == "INVALID"
    # Two real passes, both growth-aligned → still advances.
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"


def test_llm_exception_yields_reject() -> None:
    def _flaky(prompt: str) -> str:
        raise RuntimeError("mock Ollama outage")

    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_flaky)
    assert result.hierarchy_decision == "REJECT"
    assert result.rejection_reason is not None
    assert "LLM call failed" in result.rejection_reason


def test_empty_response_yields_reject() -> None:
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(""))
    assert result.hierarchy_decision == "REJECT"


# ---------------------------------------------------------------------------
# Output parsing tolerance
# ---------------------------------------------------------------------------


def test_parser_accepts_markdown_code_fence() -> None:
    raw = """```json
{
  "moat": {"verdict": "PASS", "bucket": "cost", "reasoning": "ok"},
  "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},
  "bottleneck": {"verdict": "FAIL", "type": null, "reasoning": "weak"},
  "hierarchy_decision": "ADVANCE_TO_STAGE_4",
  "rejection_reason": null
}
```"""
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(raw))
    assert result.moat.verdict == "PASS"
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_4"


def test_parser_accepts_preamble_before_json() -> None:
    raw = (
        "Sure, here is my analysis:\n"
        '{"moat": {"verdict": "PASS", "bucket": "switching", "reasoning": "ok"},'
        ' "new_frontier": {"verdict": "PASS", "imitation_evidence": ["x"], "reasoning": "ok"},'
        ' "bottleneck": {"verdict": "FAIL", "type": null, "reasoning": "no"},'
        ' "hierarchy_decision": "ADVANCE_TO_STAGE_4", "rejection_reason": null}'
    )
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(raw))
    assert result.moat.verdict == "PASS"


def test_parse_response_helper_handles_non_object_root() -> None:
    """Top-level array or string must NOT pass the parser."""
    assert _parse_response("[1, 2, 3]") is None
    assert _parse_response('"a string"') is None
    assert _parse_response("null") is None


def test_extract_axis_outcome_missing_field() -> None:
    parsed: dict = {}
    out = _extract_axis_outcome(parsed, "moat", "bucket")
    assert out.verdict == "INVALID"


def test_extract_axis_outcome_array_qualifier_joined() -> None:
    parsed = {
        "new_frontier": {
            "verdict": "PASS",
            "imitation_evidence": ["AMD copies CUDA", "Intel oneAPI"],
            "reasoning": "two industry imitations",
        }
    }
    out = _extract_axis_outcome(parsed, "new_frontier", "imitation_evidence")
    assert out.verdict == "PASS"
    assert out.qualifier == "AMD copies CUDA; Intel oneAPI"


# ---------------------------------------------------------------------------
# Hierarchy gate helper — direct unit
# ---------------------------------------------------------------------------


def test_apply_hierarchy_gate_three_pass_advances() -> None:
    moat = Stage3AxisOutcome("moat", "PASS", None, "")
    nf = Stage3AxisOutcome("new_frontier", "PASS", None, "")
    bn = Stage3AxisOutcome("bottleneck", "PASS", None, "")
    decision, reason = _apply_hierarchy_gate(moat, nf, bn)
    assert decision == "ADVANCE_TO_STAGE_4"
    assert reason is None


def test_apply_hierarchy_gate_invalid_counts_as_fail() -> None:
    moat = Stage3AxisOutcome("moat", "INVALID", None, "")
    nf = Stage3AxisOutcome("new_frontier", "PASS", None, "")
    bn = Stage3AxisOutcome("bottleneck", "PASS", None, "")
    decision, _ = _apply_hierarchy_gate(moat, nf, bn)
    # 2 real passes + growth axis covered → advance.
    assert decision == "ADVANCE_TO_STAGE_4"


def test_apply_hierarchy_gate_only_moat_invalid_growth_rejects() -> None:
    moat = Stage3AxisOutcome("moat", "PASS", None, "")
    nf = Stage3AxisOutcome("new_frontier", "INVALID", None, "")
    bn = Stage3AxisOutcome("bottleneck", "INVALID", None, "")
    decision, reason = _apply_hierarchy_gate(moat, nf, bn)
    assert decision == "REJECT"
    assert reason is not None


# ---------------------------------------------------------------------------
# Constitution version stamping
# ---------------------------------------------------------------------------


def test_result_carries_constitution_version() -> None:
    payload = {
        "moat": {"verdict": "FAIL", "bucket": None, "reasoning": "no"},
        "new_frontier": {"verdict": "FAIL", "imitation_evidence": [], "reasoning": "no"},
        "bottleneck": {"verdict": "FAIL", "type": None, "reasoning": "no"},
        "hierarchy_decision": "REJECT",
    }
    result = screen_ticker(_funds(), _advancing_prefilter(), llm_call=_llm_stub(payload))
    assert result.constitution_version == "2.0"
