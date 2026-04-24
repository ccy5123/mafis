"""Stage 3 semantic relevance filter — reduce false positives from
keyword / graph-context matches.

Stage 1 (keyword) and Stage 2 (graph-context) in `pre_filter.py` are
cheap regex-based passes. They produce FilterHits for "NVIDIA" in
"NVIDIA Corp Reports Q1 Earnings" (correct) but ALSO for "NVIDIA
Corp Donates to Scholarship Fund" (noise). For a Tier-3 promotion
decision, we want to see only hits that are MATERIALLY relevant to
the ticker's investment thesis.

This module layers a local-LLM semantic filter on top: pass each
FilterHit's news title + ticker context to Qwen 2.5 7B with a
yes/no classification prompt, keep only the YES hits.

Design rules:
  - LLM output is binary (YES | NO) with a one-sentence reason.
    Anything other than a strict YES → treated as NO.
  - Temperature 0 + seed 42, per the project reproducibility
    contract.
  - The LLM call is injectable so tests run without Ollama.
  - Empty / tiny hit lists short-circuit — no LLM call when there's
    nothing to filter.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from wise_investor.config import settings
from wise_investor.filters.pre_filter import FilterHit


logger = logging.getLogger(__name__)


_YES_RE = re.compile(r"\b(?:YES|MATERIAL|RELEVANT)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(?:NO|IMMATERIAL|NOT\s+RELEVANT|NOISE)\b", re.IGNORECASE)


_SYSTEM_PROMPT = (
    "You are a triage classifier for an investment research "
    "pipeline. For each (ticker, news headline) pair, decide whether "
    "the headline is MATERIAL to the ticker's investment thesis. "
    "Material means: earnings / guidance, supply chain shocks, "
    "regulatory actions, M&A activity, leadership changes, or direct "
    "product/competitor announcements. Immaterial means: routine PR, "
    "analyst price-target updates without new facts, ETF inclusion "
    "news, generic macro without ticker-specific impact, or unrelated "
    "mentions (e.g., charity, sports sponsorships). "
    "Output format: exactly two lines. Line 1: one token, YES or NO. "
    "Line 2: ≤ 15 words explaining the decision."
)


@dataclass
class SemanticDecision:
    """One LLM classification result for a single FilterHit."""

    hit: FilterHit
    is_material: bool
    reason: str
    raw_response: str


def _build_user_prompt(hit: FilterHit) -> str:
    """Render a single hit as a user prompt line."""
    return (
        f"Ticker: {hit.symbol}\n"
        f"Matched term: {hit.matched_term}\n"
        f"Stage: {hit.stage}\n"
        f"Headline: {hit.news_title}\n"
        f"Source: {hit.news_source}\n"
        f"Published: {hit.news_published}\n"
        "\n"
        "Material for the ticker's investment thesis? YES or NO."
    )


def _parse_decision(response: str) -> tuple[bool, str]:
    """Extract (is_material, reason) from an LLM response.

    Decision is conservative — anything without a strict YES token
    before the first NO is treated as NO. Empty response → NO.
    """
    text = (response or "").strip()
    if not text:
        return (False, "empty LLM response")

    # First non-empty line is the verdict token.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    verdict_line = lines[0] if lines else ""
    reason_line = " ".join(lines[1:])[:160] if len(lines) > 1 else ""

    # Unambiguous tokens win; otherwise treat as NO.
    if _YES_RE.search(verdict_line) and not _NO_RE.search(verdict_line):
        return (True, reason_line or "marked material")
    return (False, reason_line or "not marked material")


def filter_hits_semantically(
    hits: list[FilterHit],
    llm_call: Callable[[str, str], str] | None = None,
    max_hits: int | None = None,
) -> list[SemanticDecision]:
    """Run each hit through the LLM classifier, return decisions.

    `max_hits` caps the number of LLM calls to prevent runaway cost
    on a noisy pre-filter batch. Hits beyond the cap are returned as
    automatic NO decisions with a truncation reason.

    `llm_call` defaults to a direct Ollama call at temp 0, seed 42.
    Tests pass a stub.
    """
    if not hits:
        return []
    if llm_call is None:
        llm_call = _default_llm_call

    decisions: list[SemanticDecision] = []
    for i, hit in enumerate(hits):
        if max_hits is not None and i >= max_hits:
            decisions.append(
                SemanticDecision(
                    hit=hit,
                    is_material=False,
                    reason="dropped: beyond max_hits cap",
                    raw_response="",
                )
            )
            continue
        try:
            resp = llm_call(_SYSTEM_PROMPT, _build_user_prompt(hit))
        except Exception as e:
            logger.warning(
                "Semantic filter LLM call failed for %s/%s: %s",
                hit.symbol, hit.news_title[:40], e,
            )
            decisions.append(
                SemanticDecision(
                    hit=hit,
                    is_material=False,
                    reason=f"llm error: {e}",
                    raw_response="",
                )
            )
            continue
        is_material, reason = _parse_decision(resp)
        decisions.append(
            SemanticDecision(
                hit=hit, is_material=is_material, reason=reason, raw_response=resp
            )
        )
    return decisions


def materials_only(decisions: list[SemanticDecision]) -> list[FilterHit]:
    """Shortcut: extract only the FilterHits the LLM classified as material."""
    return [d.hit for d in decisions if d.is_material]


def _default_llm_call(system: str, user: str) -> str:
    """Production Ollama call at temp 0, seed 42."""
    import ollama

    resp = ollama.chat(
        model=settings.analyst_model,  # share Qwen with Analyst
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
    )
    return resp["message"]["content"]


__all__ = [
    "SemanticDecision",
    "filter_hits_semantically",
    "materials_only",
]
