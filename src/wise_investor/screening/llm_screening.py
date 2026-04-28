"""Stage 3 LLM screening orchestrator.

Implements constitution v2.0 §18: take a Stage 2 PrefilterResult,
build the §18 prompt, call the LLM, parse the JSON, recompute the
hierarchy gate (§9) deterministically, return a Stage3Result.

Key design principles:

  - **Don't trust the LLM's hierarchy decision.** The LLM is asked
    to provide one for diagnostic purposes, but the gate (§9) is
    rule-based. We recompute the decision from individual axis
    verdicts and use the LLM's reported value only for audit.
  - **Malformed output → REJECT.** Constitution Commitment 3 says
    when in doubt, fail. A response that doesn't parse, lacks a
    required axis, or names an unknown verdict literal is treated
    as INVALID per axis and the hierarchy gate sees that axis as
    failed.
  - **No retries on parse failure.** Retrying on bad output is a
    self-deception channel — if the LLM produces garbage, the
    correct response is to record it and reject the candidate, not
    to keep prompting until it produces something acceptable.
  - **Constitution version stamped.** Every Stage3Result carries
    the version it was scored against (§13 + §22).

LLM call is injectable so tests run without Ollama. The default
production path goes through `wise_investor.llm.get_backend()` which
resolves the per-agent config for the "stage3_screener" agent name —
adding `agents.stage3_screener:` to `config/agent_models.yaml` lets
users tune the model + sampling without touching this code.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.stage3_prompts import build_stage3_prompt
from wise_investor.screening.types import (
    HierarchyDecision,
    PrefilterResult,
    Stage3AxisOutcome,
    Stage3Result,
    Stage3VerdictKind,
    TickerFundamentals,
)


logger = logging.getLogger(__name__)


# Canonical agent name used for `config/agent_models.yaml` resolution.
# Users wanting a different model for screening (e.g. cheaper / faster)
# add `agents.stage3_screener:` to that file.
STAGE3_AGENT_NAME = "stage3_screener"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def screen_ticker(
    funds: TickerFundamentals,
    prefilter: PrefilterResult,
    llm_call: Callable[[str], str] | None = None,
) -> Stage3Result:
    """Run §18 LLM screening on one ticker.

    `llm_call(prompt) -> str` is injectable. Tests pass a stub. The
    production path uses `_default_llm_call` which routes through the
    LLMBackend abstraction with model + sampling resolved from
    `agents.stage3_screener` in agent_models.yaml.
    """
    if prefilter.hierarchy_decision == "REJECT":
        # Stage 2 already rejected — Stage 3 doesn't run. Build a
        # Stage3Result that reflects this without calling the LLM at all.
        return _stage2_passthrough(funds.symbol, prefilter)

    if llm_call is None:
        llm_call = _default_llm_call

    prompt = build_stage3_prompt(funds, prefilter)

    try:
        raw = llm_call(prompt)
    except Exception as e:
        logger.warning(
            "Stage 3 LLM call failed for %s: %s; treating as REJECT.",
            funds.symbol,
            e,
        )
        return _failure_result(
            funds.symbol,
            raw_output="",
            rejection_reason=f"LLM call failed: {e}",
        )

    parsed = _parse_response(raw)
    if parsed is None:
        return _failure_result(
            funds.symbol,
            raw_output=raw,
            rejection_reason="LLM output not parseable as JSON",
        )

    moat = _extract_axis_outcome(parsed, "moat", "bucket")
    frontier = _extract_axis_outcome(parsed, "new_frontier", "imitation_evidence")
    bottleneck = _extract_axis_outcome(parsed, "bottleneck", "type")

    decision, rejection_reason = _apply_hierarchy_gate(moat, frontier, bottleneck)

    return Stage3Result(
        symbol=funds.symbol,
        constitution_version=CONSTITUTION_VERSION,
        moat=moat,
        new_frontier=frontier,
        bottleneck=bottleneck,
        hierarchy_decision=decision,
        rejection_reason=rejection_reason,
        llm_reported_decision=parsed.get("hierarchy_decision")
        if isinstance(parsed.get("hierarchy_decision"), str)
        else None,
        raw_llm_output=raw,
    )


# ---------------------------------------------------------------------------
# Hierarchy gate (constitution §9) — deterministic recompute
# ---------------------------------------------------------------------------


def _apply_hierarchy_gate(
    moat: Stage3AxisOutcome,
    frontier: Stage3AxisOutcome,
    bottleneck: Stage3AxisOutcome,
) -> tuple[HierarchyDecision, str | None]:
    """Constitution §9: 2+ axes PASSED AND a growth axis (new_frontier
    or bottleneck) is among them.

    INVALID counts as FAIL for the gate (precision-over-recall: when
    in doubt about an axis, fail it).
    """

    def _is_pass(o: Stage3AxisOutcome) -> bool:
        return o.verdict == "PASS"

    passes = [name for name, o in (
        ("moat", moat),
        ("new_frontier", frontier),
        ("bottleneck", bottleneck),
    ) if _is_pass(o)]
    growth_passes = [n for n in passes if n in ("new_frontier", "bottleneck")]

    if len(passes) >= 2 and growth_passes:
        return ("ADVANCE_TO_STAGE_4", None)

    reasons: list[str] = []
    if len(passes) < 2:
        reasons.append(f"only {len(passes)} axis(es) passed (need 2+)")
    if not growth_passes:
        reasons.append("no growth axis (new_frontier or bottleneck) passed")
    return ("REJECT", "; ".join(reasons))


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


_VALID_VERDICTS: set[str] = {"PASS", "FAIL"}


def _parse_response(raw: str) -> dict[str, Any] | None:
    """Best-effort JSON parse of the LLM response.

    Tolerates:
      - Leading/trailing whitespace
      - Markdown code fences (```json ... ```)
      - One leading explanatory line BEFORE a JSON block

    Rejects:
      - Multiple JSON objects (we expect exactly one)
      - Non-object root (must be a dict)
    """
    if not raw or not raw.strip():
        return None

    cleaned = raw.strip()

    # Strip code fences. Any kind: ```json, ```JSON, ```.
    fence = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
    fence_match = fence.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    # Sometimes models emit a one-line preamble before the JSON.
    # Try to locate the first { ... } block on the assumption that
    # the LLM produced a single object.
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        cleaned = obj_match.group(0)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_axis_outcome(
    parsed: dict[str, Any],
    axis: str,
    qualifier_field: str,
) -> Stage3AxisOutcome:
    """Pull one axis's outcome from the parsed JSON, with strict shape
    checks. Anything off-shape lands as INVALID.
    """
    block = parsed.get(axis)
    if not isinstance(block, dict):
        return Stage3AxisOutcome(
            axis=axis,  # type: ignore[arg-type]
            verdict="INVALID",
            qualifier=None,
            reasoning=f"missing or non-object axis block: {axis!r}",
        )

    raw_verdict = block.get("verdict")
    if not isinstance(raw_verdict, str) or raw_verdict not in _VALID_VERDICTS:
        return Stage3AxisOutcome(
            axis=axis,  # type: ignore[arg-type]
            verdict="INVALID",
            qualifier=None,
            reasoning=f"axis verdict missing or unrecognized: {raw_verdict!r}",
        )

    verdict: Stage3VerdictKind = raw_verdict  # type: ignore[assignment]

    raw_qualifier = block.get(qualifier_field)
    if isinstance(raw_qualifier, list):
        # Frontier returns imitation_evidence as list.
        qualifier_str = "; ".join(str(x) for x in raw_qualifier) or None
    elif isinstance(raw_qualifier, str):
        qualifier_str = raw_qualifier or None
    else:
        qualifier_str = None

    raw_reason = block.get("reasoning")
    reasoning = str(raw_reason) if raw_reason is not None else ""

    return Stage3AxisOutcome(
        axis=axis,  # type: ignore[arg-type]
        verdict=verdict,
        qualifier=qualifier_str,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# Failure / passthrough helpers
# ---------------------------------------------------------------------------


def _stage2_passthrough(symbol: str, prefilter: PrefilterResult) -> Stage3Result:
    """When Stage 2 already rejected, build a Stage3Result that
    reflects the rejection without invoking the LLM. The Stage3
    record is still produced so downstream ledger code sees a
    uniform shape per ticker.
    """
    sentinel_outcome = lambda axis: Stage3AxisOutcome(  # noqa: E731
        axis=axis,
        verdict="FAIL",
        qualifier=None,
        reasoning="Stage 2 rejected; Stage 3 did not run",
    )
    return Stage3Result(
        symbol=symbol,
        constitution_version=CONSTITUTION_VERSION,
        moat=sentinel_outcome("moat"),
        new_frontier=sentinel_outcome("new_frontier"),
        bottleneck=sentinel_outcome("bottleneck"),
        hierarchy_decision="REJECT",
        rejection_reason=prefilter.excluded_reason
        or "Stage 2 hierarchy gate rejected",
        llm_reported_decision=None,
        raw_llm_output="",
    )


def _failure_result(
    symbol: str, raw_output: str, rejection_reason: str
) -> Stage3Result:
    """Build a Stage3Result for an LLM failure or unparseable output.
    Every axis is INVALID, hierarchy decision is REJECT.
    """
    invalid = lambda axis: Stage3AxisOutcome(  # noqa: E731
        axis=axis,
        verdict="INVALID",
        qualifier=None,
        reasoning=rejection_reason,
    )
    return Stage3Result(
        symbol=symbol,
        constitution_version=CONSTITUTION_VERSION,
        moat=invalid("moat"),
        new_frontier=invalid("new_frontier"),
        bottleneck=invalid("bottleneck"),
        hierarchy_decision="REJECT",
        rejection_reason=rejection_reason,
        llm_reported_decision=None,
        raw_llm_output=raw_output,
    )


# ---------------------------------------------------------------------------
# Production LLM call
# ---------------------------------------------------------------------------


def _default_llm_call(prompt: str) -> str:
    """Production Stage 3 LLM call.

    Routes through the LLMBackend abstraction with model + sampling
    resolved for the `stage3_screener` agent. Users override the
    routing in `config/agent_models.yaml` without editing this code.
    """
    from wise_investor.llm import get_agent_config, get_backend

    backend = get_backend()
    cfg = get_agent_config(STAGE3_AGENT_NAME, backend=backend.name)
    response = backend.chat(
        messages=[{"role": "user", "content": prompt}],
        model=cfg.model,
        sampling=cfg.sampling,
    )
    return response.content


__all__ = [
    "STAGE3_AGENT_NAME",
    "screen_ticker",
]
