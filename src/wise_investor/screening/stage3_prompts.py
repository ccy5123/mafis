"""Stage 3 LLM screening prompt builder.

Implements constitution v2.0 §18: per-ticker binary verdict on each
axis using a focused LLM call. The prompt embeds axis definitions
read directly from `docs/constitution.md` so there is exactly one
source of truth — when the constitution is updated (and its version
bumped per §13), the prompt automatically reflects the change.

Design principles (from constitution):

  - Definitions are copied verbatim. Paraphrasing risks losing the
    sentence-level guidance ("not a moat: high margin alone";
    "auto-PASS 4: less than 3 years…") that prevents the LLM from
    drifting into permissive interpretations.
  - The prompt closes with a precision-over-recall reminder
    (Commitment 3). The LLM is explicitly told that the user has
    accepted missing some good companies as the price of not
    admitting bad ones.
  - JSON output format is mandated, with shape validated downstream.
  - The hierarchy gate is computed deterministically by the caller
    after parsing — we ASK the LLM to provide a hierarchy_decision
    field, but we never trust it; the gate (§9) is rule-based.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from wise_investor.config import PROJECT_ROOT
from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.types import (
    BottleneckProxies,
    FrontierProxies,
    MoatProxies,
    PrefilterResult,
    TickerFundamentals,
)


CONSTITUTION_PATH: Path = PROJECT_ROOT / "docs" / "constitution.md"


# Heading text exactly as it appears under `## N. <heading>` in the
# constitution. Section numbers may shift with revisions; the heading
# text is the stable anchor. If a v3 constitution renames an axis we
# update both the heading map AND the prompt-alignment test together.
_AXIS_HEADINGS: dict[str, str] = {
    "moat": "Moat axis",
    "new_frontier": "New Frontier axis",
    "bottleneck": "Bottleneck axis",
}


# Cached after first read — constitution text doesn't change at runtime.
_CONSTITUTION_TEXT: str | None = None


def _load_constitution_text() -> str:
    global _CONSTITUTION_TEXT
    if _CONSTITUTION_TEXT is None:
        if not CONSTITUTION_PATH.exists():
            raise RuntimeError(
                f"Constitution file not found at {CONSTITUTION_PATH}. "
                "The Stage 3 screener cannot build prompts without it; "
                "this should never happen on a clean checkout."
            )
        _CONSTITUTION_TEXT = CONSTITUTION_PATH.read_text(encoding="utf-8")
    return _CONSTITUTION_TEXT


def extract_axis_section(axis: str) -> str:
    """Return the verbatim text of an axis section from the constitution.

    Match: `## <number>. <Heading>` through (but not including) the
    next `## <number>.` heading at the same level. The constitution
    is structured this way intentionally; if it ever isn't, the test
    in `test_stage3_prompts.py` catches the drift.
    """
    if axis not in _AXIS_HEADINGS:
        raise ValueError(f"Unknown axis: {axis!r}")
    heading = _AXIS_HEADINGS[axis]
    text = _load_constitution_text()

    pattern = re.compile(
        rf"^## \d+\. {re.escape(heading)}\n(.*?)(?=^## \d+\. |\Z)",
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            f"Could not locate axis section for {axis!r} (heading "
            f"{heading!r}) in {CONSTITUTION_PATH}. Has the heading "
            "format changed?"
        )
    body = match.group(1).strip()
    return f"## {heading}\n\n{body}"


def _format_moat_proxies(proxies: MoatProxies) -> str:
    parts = []
    if proxies.roic_3y_avg is not None:
        parts.append(f"  ROIC 3y avg: {proxies.roic_3y_avg:.4f}")
    if proxies.roic_advantage is not None:
        parts.append(
            f"  ROIC advantage over industry median: {proxies.roic_advantage:+.4f}"
        )
    if proxies.roic_advantage_trend is not None:
        parts.append(
            f"  ROIC advantage trend (slope/year): "
            f"{proxies.roic_advantage_trend:+.5f}"
        )
    if proxies.gross_margin_3y_std is not None:
        parts.append(
            f"  Gross margin std (recent quarters): "
            f"{proxies.gross_margin_3y_std:.4f}"
        )
    if proxies.gross_margin_industry_ratio is not None:
        parts.append(
            f"  Gross margin volatility ratio vs industry: "
            f"{proxies.gross_margin_industry_ratio:.2f}×"
        )
    if not parts:
        return "  (no quantitative proxies available)"
    return "\n".join(parts)


def _format_frontier_proxies(proxies: FrontierProxies) -> str:
    parts = []
    if proxies.years_since_first_segment_introduction is not None:
        parts.append(
            "  Years since earliest reported segment: "
            f"{proxies.years_since_first_segment_introduction}"
        )
    if proxies.new_segments_added_5y is not None:
        parts.append(f"  New segments added in last 5 fiscal years: {proxies.new_segments_added_5y}")
    if not parts:
        return "  (no quantitative proxies available)"
    return "\n".join(parts)


def _format_bottleneck_proxies(proxies: BottleneckProxies) -> str:
    parts = []
    if proxies.top5_customer_share is not None:
        parts.append(
            f"  Top-5 customer revenue share: {proxies.top5_customer_share:.2f}"
        )
    if proxies.hhi is not None:
        parts.append(
            f"  HHI (uniform top-5 approximation): {proxies.hhi}"
        )
    parts.append(
        f"  Diversification-attempt signals (recent count): "
        f"{proxies.diversification_attempt_signals}"
    )
    return "\n".join(parts)


def build_stage3_prompt(
    funds: TickerFundamentals,
    prefilter: PrefilterResult,
) -> str:
    """Assemble the full Stage 3 prompt for one ticker.

    The prompt embeds:
      - The verbatim axis definitions from docs/constitution.md.
      - The Stage 2 quant proxies for THIS ticker, formatted compactly.
      - The Stage 2 axis verdicts (PASS/FAIL/NEED_LLM) so the LLM
        sees what the quant gate concluded — useful framing without
        being binding.
      - The hierarchy gate logic.
      - The required JSON output format.
      - The precision-over-recall reminder (Commitment 3).
    """
    moat_def = extract_axis_section("moat")
    frontier_def = extract_axis_section("new_frontier")
    bottleneck_def = extract_axis_section("bottleneck")

    moat_p = _format_moat_proxies(prefilter.moat.details and MoatProxies(
        roic_3y_avg=prefilter.moat.details.get("roic_3y_avg"),
        roic_advantage=prefilter.moat.details.get("roic_advantage"),
        roic_advantage_trend=prefilter.moat.details.get("roic_advantage_trend"),
        gross_margin_3y_std=prefilter.moat.details.get("gross_margin_3y_std"),
        gross_margin_industry_ratio=prefilter.moat.details.get(
            "gross_margin_industry_ratio"
        ),
        customer_concentration_trend=prefilter.moat.details.get(
            "customer_concentration_trend"
        ),
    ))

    frontier_p = _format_frontier_proxies(FrontierProxies(
        years_since_first_segment_introduction=prefilter.new_frontier.details.get(
            "years_since_first_segment_introduction"
        ),
        new_segments_added_5y=prefilter.new_frontier.details.get(
            "new_segments_added_5y"
        ),
    ))

    bottleneck_p = _format_bottleneck_proxies(BottleneckProxies(
        top5_customer_share=prefilter.bottleneck.details.get("top5_customer_share"),
        hhi=prefilter.bottleneck.details.get("hhi"),
        diversification_attempt_signals=prefilter.bottleneck.details.get(
            "diversification_attempt_signals", 0
        ),
    ))

    primary_segment_note = (
        f"Primary segment (constitution §13): "
        f"{prefilter.primary_segment.primary_segment_name} "
        f"({(prefilter.primary_segment.primary_segment_revenue_share or 0):.0%} of "
        f"FY{prefilter.primary_segment.fiscal_year} revenue)"
        if prefilter.primary_segment is not None
        and prefilter.primary_segment.primary_segment_exists
        else "(no primary segment ≥30%)"
    )

    return f"""\
You are evaluating whether {funds.symbol} passes the user's investment rubric.

The user's rubric (constitution v{CONSTITUTION_VERSION}) has three axes. Each
axis has a precise definition. Evaluate each axis independently and return
PASS or FAIL. Verbatim definitions follow.

Industry classification: {funds.industry_classification}
{primary_segment_note}

=================================================================
AXIS 1 — MOAT
=================================================================
{moat_def}

Stage 2 quantitative proxies for {funds.symbol}:
{moat_p}

Stage 2 quant verdict: {prefilter.moat.verdict} — {prefilter.moat.reason}

=================================================================
AXIS 2 — NEW FRONTIER
=================================================================
{frontier_def}

Stage 2 quantitative proxies for {funds.symbol}:
{frontier_p}

Stage 2 quant verdict: {prefilter.new_frontier.verdict} — {prefilter.new_frontier.reason}

=================================================================
AXIS 3 — BOTTLENECK
=================================================================
{bottleneck_def}

Stage 2 quantitative proxies for {funds.symbol}:
{bottleneck_p}

Stage 2 quant verdict: {prefilter.bottleneck.verdict} — {prefilter.bottleneck.reason}

=================================================================
HIERARCHY GATE (constitution §9)
=================================================================
After evaluating all three axes:
- Count PASSes
- Check growth axis inclusion: New Frontier OR Bottleneck must be PASS

Final classification:
- 2+ PASSes AND growth axis included → ADVANCE_TO_STAGE_4
- All other combinations → REJECT

The user's caller will recompute this gate from your per-axis verdicts;
your `hierarchy_decision` field is informational. Be honest about each
axis even if the gate result feels disappointing.

=================================================================
OUTPUT FORMAT
=================================================================
Return ONLY a JSON object with this exact shape, no prose, no
markdown fences. Do not invent fields.

{{
  "moat":         {{"verdict": "PASS" or "FAIL",
                    "bucket":  "intangible" or "switching" or "network" or "cost" or null,
                    "reasoning": "<2-3 sentences>"}},
  "new_frontier": {{"verdict": "PASS" or "FAIL",
                    "imitation_evidence": ["...","..."] or [],
                    "reasoning": "<2-3 sentences>"}},
  "bottleneck":   {{"verdict": "PASS" or "FAIL",
                    "type":    "technical" or "resource" or "regulatory" or "division-of-labor" or null,
                    "reasoning": "<2-3 sentences>"}},
  "hierarchy_decision": "ADVANCE_TO_STAGE_4" or "REJECT",
  "rejection_reason": "<short>" or null
}}

=================================================================
PRECISION OVER RECALL (Commitment 3)
=================================================================
Be conservative. When uncertain, FAIL the axis. The user has explicitly
accepted that this system will miss some good companies in exchange for
not admitting bad ones. Saying PASS on weak evidence is far worse than
saying FAIL on a real candidate that you'll re-encounter on a future
re-screening when the evidence has firmed up.
"""


def render_prompt_for_inspection(
    funds: TickerFundamentals, prefilter: PrefilterResult
) -> str:
    """Convenience for the CLI / debugging — wraps build_stage3_prompt
    with a header that's not part of the LLM prompt itself.
    """
    return (
        f"=== STAGE 3 PROMPT for {funds.symbol} (constitution v{CONSTITUTION_VERSION}) ===\n"
        f"\n{build_stage3_prompt(funds, prefilter)}"
    )


__all__ = [
    "CONSTITUTION_PATH",
    "build_stage3_prompt",
    "extract_axis_section",
    "render_prompt_for_inspection",
]
