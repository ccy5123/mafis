"""Pre-filter stages 1 and 2 — keyword match + graph context match.

Input: a batch of news items (duck-typed NewsItemLike from the
chain_alerts module) plus a set of candidate tickers with their
registered keyword profiles + optional value chain graph.

Output: a list of FilterHit records (one per (ticker, news, reason)
tuple), aggregated scores per ticker, and promotion recommendations
against the current tier assignment.

Design philosophy:
  - Stage 1 is the cheapest signal — symbol / company name / notes
    tokens as literal keyword matches. Works for Tier 3 names with
    no brief at all.
  - Stage 2 layers on value-chain graph context for Tier 1/2 names
    (or any ticker that has a brief). Reuses the chain_alerts
    match machinery to find news mentioning upstream / peer /
    downstream graph nodes.
  - Scoring is per-ticker count of hits in the batch — intentionally
    crude. A more sophisticated model (source credibility weighting,
    time decay, sentiment) is Phase 4 work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from wise_investor.alerts.chain_alerts import (
    NODE_ALIASES,
    NewsItemLike,
    find_target_paths,
)
from wise_investor.geopolitics.snapshot import SYMBOL_KEYWORDS
from wise_investor.value_chain.graph import ValueChainGraph


@dataclass
class FilterHit:
    """One (ticker, news, match-reason) triple from a pre-filter scan."""

    symbol: str
    stage: str  # "keyword" / "graph_context"
    matched_term: str
    news_title: str
    news_source: str
    news_published: str
    reason: str = ""  # optional human-readable context
    graph_path: list[str] = field(default_factory=list)


@dataclass
class PromotionRecommendation:
    """Suggested tier change based on accumulated hits."""

    symbol: str
    current_tier: int | None  # None = not in registry
    suggested_tier: int
    score: int
    sample_titles: list[str] = field(default_factory=list)
    reason: str = ""


# Thresholds chosen conservatively — a ticker needs real signal before
# the recommender suggests escalating its tier.
DEFAULT_THRESHOLDS: dict[int, int] = {
    # Promote TO tier key WHEN hits >= value
    2: 3,   # Tier 3 → Tier 2 after 3 hits
    1: 8,   # Tier 2 → Tier 1 after 8 hits
}


def _keyword_variants(symbol: str) -> list[str]:
    """Build the literal keyword list to scan for a given ticker.

    Union of:
      - the symbol itself
      - entries in SYMBOL_KEYWORDS (company name + extra domain terms)
      - the NODE_ALIASES entry, if any
    """
    sym = symbol.upper()
    keywords: list[str] = [sym]
    entry = SYMBOL_KEYWORDS.get(sym)
    if entry is not None:
        company, extras = entry
        keywords.append(company)
        keywords.extend(extras)
    if sym in NODE_ALIASES:
        keywords.extend(NODE_ALIASES[sym])
    # Dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        low = k.lower().strip()
        if not low or low in seen:
            continue
        seen.add(low)
        out.append(k)
    return out


def _text_contains(
    text: str, phrase: str, word_boundary: bool = True
) -> bool:
    """Substring / word-boundary test, case-insensitive."""
    if not phrase:
        return False
    if word_boundary:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        return bool(re.search(pattern, text.lower()))
    return phrase.lower() in text.lower()


def scan_keywords(
    news_items: Iterable[NewsItemLike],
    symbol: str,
    extra_keywords: list[str] | None = None,
) -> list[FilterHit]:
    """Stage 1 — literal keyword match against news headlines.

    `extra_keywords` lets the caller augment the registered keyword
    set (e.g., for a ticker not in SYMBOL_KEYWORDS yet, pass the
    `notes` string tokenized).
    """
    keywords = _keyword_variants(symbol)
    if extra_keywords:
        keywords = keywords + [
            k for k in extra_keywords if k and k.strip()
        ]
    hits: list[FilterHit] = []
    seen_titles: set[str] = set()
    for item in news_items:
        title = getattr(item, "title", "") or ""
        if not title:
            continue
        key = (symbol.upper(), title.strip().lower())
        if key in seen_titles:
            continue
        for kw in keywords:
            # Short tokens (1-2 chars) use loose substring, longer ones
            # word-boundary. Prevents "AMD" matching "amderson" while
            # still letting a 3-letter ticker survive as a word token.
            word_boundary = len(kw) >= 3
            if _text_contains(title, kw, word_boundary=word_boundary):
                hits.append(
                    FilterHit(
                        symbol=symbol.upper(),
                        stage="keyword",
                        matched_term=kw,
                        news_title=title,
                        news_source=getattr(item, "source", "") or "",
                        news_published=getattr(item, "published", "") or "",
                        reason=f"direct keyword match on {kw!r}",
                    )
                )
                seen_titles.add(key)
                break  # one hit per news item, not per keyword
    return hits


def scan_graph_context(
    news_items: Iterable[NewsItemLike],
    graph: ValueChainGraph,
    symbol: str,
    max_hops: int = 2,
) -> list[FilterHit]:
    """Stage 2 — match headlines against value chain nodes within
    `max_hops` of `symbol` in the graph.

    If `symbol` is not in the graph (never onboarded with a brief),
    returns an empty list — caller should fall back to stage 1 only.
    """
    if not graph.has_company(symbol):
        return []

    # Which graph nodes are "close" to this target? BFS up to max_hops.
    close_nodes: set[str] = {symbol.upper()}
    # We reuse chain_alerts BFS semantics by walking out/in edges.
    from collections import deque

    for direction in ("out", "in"):
        queue: deque[tuple[str, int]] = deque([(symbol, 0)])
        visited: set[str] = {symbol}
        while queue:
            node, depth = queue.popleft()
            if depth >= max_hops:
                continue
            if direction == "out":
                edges = list(graph._g.out_edges(node, data=True))  # noqa: SLF001
                neighbors = [tgt for _, tgt, _ in edges]
            else:
                edges = list(graph._g.in_edges(node, data=True))  # noqa: SLF001
                neighbors = [src for src, _, _ in edges]
            for neighbor in neighbors:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                close_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Remove the target itself — its own mentions are stage-1 territory.
    close_nodes.discard(symbol.upper())

    hits: list[FilterHit] = []
    for node in close_nodes:
        aliases = NODE_ALIASES.get(node, [node])
        # Also include ticker if the node carries one.
        company = graph.get_company(node)
        if company and company.ticker and company.ticker not in aliases:
            aliases = aliases + [company.ticker]
        # Compute once per-node.
        try:
            paths = find_target_paths(graph, node, max_hops=max_hops)
            # The path to `symbol` specifically, if it exists.
            path_to_symbol = next(
                (p for t, p, _ in paths if t == symbol.upper()),
                [node, symbol.upper()],
            )
        except Exception:
            path_to_symbol = [node, symbol.upper()]

        for item in news_items:
            title = getattr(item, "title", "") or ""
            if not title:
                continue
            for alias in aliases:
                if _text_contains(title, alias, word_boundary=len(alias) >= 3):
                    hits.append(
                        FilterHit(
                            symbol=symbol.upper(),
                            stage="graph_context",
                            matched_term=alias,
                            news_title=title,
                            news_source=getattr(item, "source", "") or "",
                            news_published=getattr(item, "published", "") or "",
                            reason=f"graph-context match on {alias!r} (node: {node})",
                            graph_path=path_to_symbol,
                        )
                    )
                    break  # one hit per (news, node)
    return hits


def aggregate_scores(hits: Iterable[FilterHit]) -> dict[str, int]:
    """Return {symbol: hit_count}, deduped on (symbol, news_title)."""
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}
    for h in hits:
        key = (h.symbol, h.news_title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        counts[h.symbol] = counts.get(h.symbol, 0) + 1
    return counts


def recommend_promotions(
    scores: dict[str, int],
    current_tiers: dict[str, int | None],
    thresholds: dict[int, int] | None = None,
    sample_hits: dict[str, list[FilterHit]] | None = None,
) -> list[PromotionRecommendation]:
    """Given per-ticker scores and current tier assignments, emit
    promotion recommendations in descending score order.

    A ticker in tier T with score S suggests promotion to tier T'
    where T' is the highest tier (lowest number) whose threshold S
    crosses. Tickers already in Tier 1 are never "promoted" further.
    Unknown tickers (current_tier=None) can be promoted to Tier 3.
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS
    # Sort scores descending so the most-hit ticker shows up first.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    recs: list[PromotionRecommendation] = []
    for symbol, score in ranked:
        current = current_tiers.get(symbol)
        suggested = _suggest_tier(current, score, thresholds)
        if suggested is None or suggested == current:
            continue
        titles: list[str] = []
        if sample_hits and symbol in sample_hits:
            titles = [h.news_title for h in sample_hits[symbol][:3]]
        recs.append(
            PromotionRecommendation(
                symbol=symbol,
                current_tier=current,
                suggested_tier=suggested,
                score=score,
                sample_titles=titles,
                reason=(
                    f"{score} filter hit(s) in the scan window; "
                    f"current tier={current}, threshold crossed for "
                    f"tier_{suggested}."
                ),
            )
        )
    return recs


def _suggest_tier(
    current: int | None, score: int, thresholds: dict[int, int]
) -> int | None:
    """Pick the highest tier whose threshold the score crosses.

    Returns the suggested tier (1/2/3) or None if nothing changes.
    """
    # If not in the registry, dropping below any threshold → Tier 3.
    if current is None:
        if score >= 1:
            return 3
        return None
    # Walk promotion thresholds from highest-tier (1) downward. First
    # one the score crosses AND is a real promotion wins.
    for target_tier in sorted(thresholds.keys()):
        needed = thresholds[target_tier]
        if score >= needed and target_tier < current:
            return target_tier
    return None


__all__ = [
    "DEFAULT_THRESHOLDS",
    "FilterHit",
    "PromotionRecommendation",
    "aggregate_scores",
    "recommend_promotions",
    "scan_graph_context",
    "scan_keywords",
]
