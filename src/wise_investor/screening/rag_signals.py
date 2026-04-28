"""Extract structured RAG-derived signals for the Stage 2 prefilter.

Two signals feed back into `TickerFundamentals`:

  - `top5_customer_share`: float | None — fraction in [0, 1]. Pulled
    from 10-K Item 1A (Risk Factors) via ChromaDB semantic search +
    LLM numeric extraction. The bottleneck axis (constitution §15)
    fires PASS at >= 0.40.

  - `diversification_attempt_signals`: int — count of diversification
    announcements in 10-K Item 1 (Business). Constitution §15 treats
    diversification attempts as a bottleneck-erosion signal (anyone
    trying to diversify is at risk of breaking the focus that made
    them defensible).

Two extraction strategies, deliberately different:

  - top-5 customer share is a NUMERIC EXTRACTION problem. LLMs handle
    "find the percent in this passage" reliably; regex doesn't,
    because the 10-K phrasing varies wildly ("represented approximately
    25%", "five largest customers accounted for 27%", "concentrated in
    a small number of customers (over 30%)", etc.).

  - diversification signals is a COUNTING problem. A keyword counter
    is more reliable than an LLM here — LLMs over- or under-count
    depending on prompt phrasing, and we don't need semantic nuance.
    Just: how many times did the company announce expansion?

Both functions degrade gracefully (Commitment 3):
  - No filing indexed for the symbol → top5 returns None, diversif. returns 0
  - LLM call fails → top5 returns None
  - LLM returns malformed JSON → top5 returns None

Korean tickers are out of scope here — DART filings have a different
structure and aren't indexed in the same ChromaDB collection. The
dispatcher in live_adapter routes KR symbols through the KR adapter
which doesn't consume RAG signals.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# Diversification-attempt keywords (case-insensitive) for the §15
# bottleneck-erosion signal. Each pattern fires at most once across the
# pooled passages — we count distinct *types* of diversification language,
# not raw keyword frequency. The patterns allow optional articles and
# "new" qualifier to absorb 10-K phrasing variation.
_DIVERSIFICATION_KEYWORDS: tuple[str, ...] = (
    r"\bentered\s+(?:the\s+|a\s+|our\s+)?(?:new\s+)?(?:markets?|businesses?|segments?)\b",
    r"\blaunched\s+(?:a\s+|our\s+)?(?:new\s+)?(?:products?|product\s+lines?|businesses?|segments?)\b",
    r"\bacquired\s+(?:a\s+|our\s+)?(?:new\s+)?(?:business|company|line|operations?)\b",
    r"\bexpanded\s+(?:our\s+)?(?:operations\s+)?into\s+(?:the\s+|a\s+)?(?:new\s+)?(?:markets?|segments?|businesses?)\b",
    r"\bnew\s+(?:business|product|operating)\s+segments?\b",
    r"\bdiversif(?:y|ying|ication)\s+(?:into|our|the|of)\b",
)

# Cap on diversification count to avoid noise. Past 5 mentions the
# signal is already strong; counting higher doesn't change the verdict.
DIVERSIFICATION_CAP: int = 5

# Cap on top-5 share. Past 100% is nonsensical; some 10-Ks say "over
# 50%" without a precise number — we accept that lower bound.
TOP5_MAX: float = 1.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RagSignals:
    """The two RAG-derived fields the live US adapter can populate."""

    top5_customer_share: float | None
    diversification_attempt_signals: int
    top5_evidence: str  # short snippet from the LLM's source passage; "" when None


# ---------------------------------------------------------------------------
# Injectable function shapes (Protocol-style aliases)
# ---------------------------------------------------------------------------


class SearchFn(Protocol):
    """Signature compatible with `wise_investor.rag.index.search`."""

    def __call__(
        self,
        query: str,
        symbol: str | None = None,
        section: str | None = None,
        k: int = 5,
    ) -> list[Any]: ...


LLMCallFn = Callable[[str], str]


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def extract_rag_signals(
    symbol: str,
    *,
    search_fn: SearchFn | None = None,
    llm_call: LLMCallFn | None = None,
) -> RagSignals:
    """Run both extractors and return a single bundle.

    Convenience for the live adapter / CLI which want to populate both
    fields in one call. The two extractors don't share state, so this
    is just sequential invocation.
    """
    top5, evidence = extract_top5_customer_share(
        symbol, search_fn=search_fn, llm_call=llm_call
    )
    diversif = extract_diversification_signals(symbol, search_fn=search_fn)
    return RagSignals(
        top5_customer_share=top5,
        diversification_attempt_signals=diversif,
        top5_evidence=evidence,
    )


def extract_top5_customer_share(
    symbol: str,
    *,
    search_fn: SearchFn | None = None,
    llm_call: LLMCallFn | None = None,
    max_passages: int = 5,
) -> tuple[float | None, str]:
    """Pull top-5 customer concentration from 10-K Item 1A via RAG + LLM.

    Returns (share, evidence). `share` is None when:
      - No passages indexed for `symbol` (or no risk_factors section)
      - LLM declined to extract a number (e.g. text doesn't disclose
        concentration)
      - LLM returned malformed output

    `evidence` is a short snippet from the source passage when share is
    not None; otherwise an empty string.
    """
    sf = _resolve_search_fn(search_fn)
    if sf is None:
        return (None, "")

    queries = [
        "top customers concentration percentage of revenue",
        "five largest customers accounted for revenue",
        "customer concentration risk material customer",
    ]
    hits: list[Any] = []
    seen_ids: set[str] = set()
    for q in queries:
        try:
            results = sf(q, symbol=symbol, section="risk_factors", k=max_passages)
        except Exception as e:
            logger.warning("RAG search failed for %s/%s: %s", symbol, q, e)
            continue
        for h in results:
            # Build a stable hit ID: search results from different queries
            # often overlap. Dedup by the concatenated text prefix to
            # avoid prompting the LLM with redundant content.
            text = getattr(h, "text", "") or ""
            key = text[:200]
            if not key or key in seen_ids:
                continue
            seen_ids.add(key)
            hits.append(h)

    if not hits:
        return (None, "")

    passages_block = "\n\n---\n\n".join(
        f"[{getattr(h, 'symbol', '?')}/{getattr(h, 'section', '?')}/{getattr(h, 'filing_date', '?')}]\n{getattr(h, 'text', '')}"
        for h in hits[:max_passages]
    )

    prompt = _build_top5_prompt(symbol, passages_block)

    lc = llm_call if llm_call is not None else _default_llm_call
    try:
        raw = lc(prompt)
    except Exception as e:
        logger.warning("top5 LLM call failed for %s: %s", symbol, e)
        return (None, "")

    parsed = _parse_top5_response(raw)
    if parsed is None:
        return (None, "")

    share = parsed.get("top5_share")
    evidence = parsed.get("evidence") or ""

    if share is None:
        return (None, "")

    try:
        share_f = float(share)
    except (TypeError, ValueError):
        return (None, "")

    if not 0.0 <= share_f <= TOP5_MAX:
        # Out of plausible range — refuse rather than fabricate.
        logger.warning(
            "top5 share for %s out of [0,1] range: %r — discarding", symbol, share
        )
        return (None, "")

    return (share_f, str(evidence)[:500])


def extract_diversification_signals(
    symbol: str,
    *,
    search_fn: SearchFn | None = None,
    max_passages: int = 10,
) -> int:
    """Count diversification-attempt keywords in 10-K Business passages.

    Returns 0 when no passages indexed for the symbol.
    """
    sf = _resolve_search_fn(search_fn)
    if sf is None:
        return 0

    queries = [
        "expansion new market segment business launch",
        "acquired entered new business",
        "diversification strategy new product line",
    ]

    seen_ids: set[str] = set()
    pooled_text: list[str] = []
    for q in queries:
        try:
            results = sf(q, symbol=symbol, section="business", k=max_passages)
        except Exception as e:
            logger.warning("RAG search failed for %s/%s: %s", symbol, q, e)
            continue
        for h in results:
            text = getattr(h, "text", "") or ""
            key = text[:200]
            if not key or key in seen_ids:
                continue
            seen_ids.add(key)
            pooled_text.append(text)

    if not pooled_text:
        return 0

    joined = "\n\n".join(pooled_text)
    count = 0
    for pattern in _DIVERSIFICATION_KEYWORDS:
        # Each pattern contributes at most one signal — we don't double-count
        # if the same phrase appears multiple times in the same filing.
        if re.search(pattern, joined, re.IGNORECASE):
            count += 1

    return min(count, DIVERSIFICATION_CAP)


# ---------------------------------------------------------------------------
# LLM prompt + response parsing
# ---------------------------------------------------------------------------


def _build_top5_prompt(symbol: str, passages: str) -> str:
    return f"""You are extracting a single numeric fact from 10-K Risk Factor disclosures.

TICKER: {symbol}

PASSAGES (joined; from the company's own filings):
{passages}

TASK: Determine what fraction of revenue the top 5 customers account for.

DECISION RULES:
1. Return a number in [0, 1] when the passage discloses a specific figure
   for the largest customer(s) (e.g. "top 5 customers accounted for 27%"
   → 0.27; "our largest customer represented 18%" → at least 0.18 — return 0.18).
2. Return null when:
   - The passage doesn't disclose customer concentration at all.
   - The passage talks about customer concentration qualitatively but
     gives no number ("concentrated in a small number of customers").
   - You're unsure what the number refers to.
3. NEVER fabricate or estimate. NEVER round up. If unsure, return null.

OUTPUT FORMAT — JSON only, no commentary:
{{"top5_share": <number in [0,1] or null>, "evidence": "<short quote from the passage that justifies the number, or empty string if null>"}}

JSON output:"""


def _parse_top5_response(raw: str) -> dict[str, Any] | None:
    """Extract the JSON dict from a possibly-noisy LLM response.

    Handles:
      - Plain JSON
      - JSON wrapped in ```json ... ``` fences
      - JSON with leading/trailing prose (find the first balanced {...})
    """
    if not raw or not raw.strip():
        return None

    # Strip markdown fences.
    s = raw.strip()
    if s.startswith("```"):
        # Drop fence and optional language tag
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)

    # Try direct parse first.
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    # Find the first {...} balanced block.
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                snippet = s[start : i + 1]
                try:
                    return json.loads(snippet)
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


# ---------------------------------------------------------------------------
# Default-injection helpers
# ---------------------------------------------------------------------------


def _resolve_search_fn(search_fn: SearchFn | None) -> SearchFn | None:
    """Return the supplied stub or fall back to the production search.

    The fallback imports lazily so this module stays importable in
    environments that don't have ChromaDB configured (e.g. test
    runs that always inject a stub).
    """
    if search_fn is not None:
        return search_fn
    try:
        from wise_investor.rag.index import search as production_search
    except Exception as e:
        logger.warning("RAG production search unavailable: %s", e)
        return None
    return production_search


def _default_llm_call(prompt: str) -> str:
    """Reuse the same LLMBackend wiring as Stage 3 screening.

    Reading the agent config separately from llm_screening.py would
    drift the two paths over time; instead we delegate to that module's
    helper directly.
    """
    from wise_investor.screening.llm_screening import _default_llm_call as s3
    return s3(prompt)


__all__ = [
    "DIVERSIFICATION_CAP",
    "LLMCallFn",
    "RagSignals",
    "SearchFn",
    "TOP5_MAX",
    "extract_diversification_signals",
    "extract_rag_signals",
    "extract_top5_customer_share",
]
