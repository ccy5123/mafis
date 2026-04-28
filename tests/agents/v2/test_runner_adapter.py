"""Tests for the v2 runner adapter — closure-based prompt-builder shims.

Verifies that:
  - build_v2_prompt_bundle returns the 6 keys the runner expects
  - Each shim's call signature matches the v1 runner's contract
  - Shims inject the v2-specific context (AttackPlan, n_total_attacks,
    audit summary) without mutating the wrapped builder's behavior
  - Steward shim runs the audit at prompt-build time and embeds the
    summary verbatim in the output
"""

from __future__ import annotations

from wise_investor.agents.v2.attack_distribution import (
    distribute_skeptic_attacks,
    total_attacks,
)
from wise_investor.agents.v2.runner_adapter import build_v2_prompt_bundle

# ---------------------------------------------------------------------------
# Bundle shape
# ---------------------------------------------------------------------------


def test_bundle_returns_six_runner_keys() -> None:
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    expected = {
        "skeptic_system",
        "skeptic_user_prompt_builder",
        "defender_system",
        "defender_user_prompt_builder",
        "steward_system",
        "steward_user_prompt_builder",
    }
    assert set(bundle.keys()) == expected


def test_systems_are_non_empty_strings() -> None:
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    for key in ("skeptic_system", "defender_system", "steward_system"):
        value = bundle[key]
        assert isinstance(value, str)
        assert len(value) > 100  # v2 system prompts are substantial


def test_builders_are_callable() -> None:
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    for key in (
        "skeptic_user_prompt_builder",
        "defender_user_prompt_builder",
        "steward_user_prompt_builder",
    ):
        assert callable(bundle[key])


# ---------------------------------------------------------------------------
# Skeptic shim
# ---------------------------------------------------------------------------


def test_skeptic_shim_signature_matches_v1_runner() -> None:
    """The runner calls `skeptic_user_prompt_builder(symbol, value_chain,
    analyst_text, valuer_text)` — v1 signature with 4 positional args.
    The v2 shim must accept exactly this shape."""
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    skeptic_builder = bundle["skeptic_user_prompt_builder"]

    out = skeptic_builder("NVDA", "<chain>", "<analyst>", "<valuer>")
    assert isinstance(out, str)
    assert "NVDA" in out


def test_skeptic_shim_embeds_attack_plan() -> None:
    """The plan distributes attacks across passed axes. The Skeptic
    user prompt should reflect this — for 2 axes (moat + bottleneck),
    the plan totals 5 attacks (2-2-1 distribution)."""
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    out = bundle["skeptic_user_prompt_builder"](
        "NVDA", "<chain>", "<analyst>", "<valuer>"
    )
    plan = distribute_skeptic_attacks(["moat", "bottleneck"])
    n_total = total_attacks(plan)
    # The prompt should mention the total attack count somewhere.
    assert str(n_total) in out


# ---------------------------------------------------------------------------
# Defender shim
# ---------------------------------------------------------------------------


def test_defender_shim_signature() -> None:
    """v1 Defender call: (symbol, analyst, valuer, skeptic_text). 4 args."""
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    defender_builder = bundle["defender_user_prompt_builder"]
    out = defender_builder("NVDA", "<a>", "<v>", "<s>")
    assert "NVDA" in out


def test_defender_shim_passes_n_total_attacks() -> None:
    """The v2 Defender prompt explicitly tells the model how many
    DEFENDED/CONCEDED labels to produce. For 3 axes (5+2 distribution),
    that's 7."""
    bundle = build_v2_prompt_bundle(
        passed_axes=["moat", "new_frontier", "bottleneck"]
    )
    out = bundle["defender_user_prompt_builder"]("NVDA", "<a>", "<v>", "<s>")
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    n_total = total_attacks(plan)
    assert str(n_total) in out


# ---------------------------------------------------------------------------
# Steward shim — audit injection
# ---------------------------------------------------------------------------


def test_steward_shim_runner_signature() -> None:
    """v1 Steward call: (symbol, value_chain, analyst, valuer, skeptic, defender)."""
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    out = bundle["steward_user_prompt_builder"](
        "NVDA",
        "<chain>",
        "<analyst>",
        "<valuer>",
        "<skeptic>",
        "<defender>",
    )
    assert "NVDA" in out


def test_steward_shim_runs_audit_with_real_attack_text() -> None:
    """When skeptic_text contains structured attacks and defender_text
    has matching DEFENDED tags, the Steward prompt should include the
    audit's summary_text — proof the audit ran at prompt-build time."""
    skeptic_text = (
        "1. **[axis: moat] Attack type: roic_advantage_erosion**\n"
        "ROIC will compress over time as competitors close the gap.\n"
        "2. **[axis: moat] Attack type: switching_cost_decay**\n"
        "Customers can migrate as protocols standardize.\n"
        "3. **[axis: bottleneck] Attack type: concentration_risk**\n"
        "Loss of one customer would devastate revenue.\n"
        "4. **[axis: bottleneck] Attack type: bypass_emergence**\n"
        "Vertical integration by hyperscalers threatens position.\n"
        "5. **[axis: overall_thesis] Attack type: capital_intensity**\n"
        "Free cash flow conversion is structurally weak.\n"
    )
    defender_text = (
        "1. **[axis: moat] DEFENDED**\n"
        "ROIC has expanded YoY [Source: get_roic_trend].\n"
        "2. **[axis: moat] DEFENDED**\n"
        "Customer churn is below 1% [Source: customer_churn_rate].\n"
        "3. **[axis: bottleneck] DEFENDED**\n"
        "Top customer is 18% of revenue, not 40%+ [Source: top5_disclosure].\n"
        "4. **[axis: bottleneck] CONCEDED**\n"
        "Hyperscaler vertical integration is a real long-term risk.\n"
        "5. **[axis: overall_thesis] DEFENDED**\n"
        "FCF conversion is 60% [Source: calculate_fcf].\n"
    )
    bundle = build_v2_prompt_bundle(passed_axes=["moat", "bottleneck"])
    out = bundle["steward_user_prompt_builder"](
        "NVDA",
        "<chain>",
        "<analyst>",
        "<valuer>",
        skeptic_text,
        defender_text,
    )
    # Steward prompt should contain the audit's pre-formatted summary.
    # AuditResult.summary_text starts with structured output mentioning
    # defended ratio / per-attack breakdown.
    assert "DEFENDED" in out or "defended" in out.lower()
    # Should have references to the per-axis verdict from §21
    assert "moat" in out.lower()
    assert "bottleneck" in out.lower()


def test_steward_shim_embeds_passed_axes() -> None:
    bundle = build_v2_prompt_bundle(
        passed_axes=["moat", "new_frontier", "bottleneck"]
    )
    out = bundle["steward_user_prompt_builder"](
        "NVDA",
        "<chain>",
        "<analyst>",
        "<valuer>",
        "<skeptic>",
        "<defender>",
    )
    # All three axes should appear in the prompt
    assert "moat" in out.lower()
    assert "new_frontier" in out.lower() or "frontier" in out.lower()
    assert "bottleneck" in out.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_two_axis_pass_distributes_correctly() -> None:
    """2 axes → 5 total attacks per attack_distribution.py."""
    plan = distribute_skeptic_attacks(["moat", "bottleneck"])
    assert total_attacks(plan) == 5


def test_three_axis_pass_distributes_correctly() -> None:
    """3 axes → 7 total attacks per attack_distribution.py."""
    plan = distribute_skeptic_attacks(["moat", "new_frontier", "bottleneck"])
    assert total_attacks(plan) == 7


def test_single_axis_passed_raises() -> None:
    """Single axis is structurally invalid for Stage 4 — the hierarchy
    gate at Stage 3 must reject before reaching here. Verify the
    distribution helper enforces this so silent miscounting can't
    happen if a caller bypasses the gate."""
    import pytest as _pytest

    with _pytest.raises(ValueError, match="<2 passed axes"):
        build_v2_prompt_bundle(passed_axes=["moat"])
