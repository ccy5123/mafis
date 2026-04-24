"""News-driven chain alerts over the value chain graph.

Given a value chain graph (nodes = companies, edges typed supplies /
peer / infrastructure) and a stream of news items (Google News
headlines + GDELT articles), identify events that impact Tier 1 target
tickers via graph paths and emit actionable alerts.

Design rationale (§5.1): hand-curated value chain briefs list
vulnerable links like "TSMC Taiwan single point" and "HBM supply
oligopoly". Without alerts, the human has to notice those events in
the news themselves. With alerts, a Reuters headline "TSMC Kaohsiung
fab outage" propagates to "NVDA re-review required, chain: NVDA
← supplies ← TSMC [Source: Reuters, 2026-04-23]".

The matcher is intentionally simple:
  - For each graph node, build a set of alias strings (the node name,
    its ticker if present, plus any aliases we register manually).
  - For each news item, check if any alias appears in the title or
    source country / domain string.
  - A match produces (node, news_item) pairs.

For path-finding we BFS from the matched node OUTWARD (reverse edges)
up to `max_hops` to find the nearest `is_target=True` nodes. We also
check FORWARD in case the target IS the matched node or downstream.

Telegram push lives in a thin wrapper to keep this module pure; a
dry-run (no notifier) just returns the alerts list so the caller can
print them, save to JSON, or batch-send.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from wise_investor.value_chain.graph import ValueChainGraph


logger = logging.getLogger(__name__)


@dataclass
class NewsItemLike:
    """Duck-typed news item. Accepts GoogleNewsItem or GdeltArticle.

    Attribute access only — no isinstance checks so we can take either
    source without pulling in the geopolitics package at module level.
    """

    title: str
    source: str  # outlet name for Google News, domain for GDELT
    published: str  # ISO YYYY-MM-DD when possible
    kind: str = "news"  # "google_news" / "gdelt" / "news"


@dataclass
class ChainAlert:
    """One actionable alert: news event → affected target ticker."""

    target_symbol: str
    matched_node: str
    chain_path: list[str]  # [target, ..., matched_node] or [matched_node, ..., target]
    hops: int  # path length - 1
    relation: str  # "supplies" / "peer" / "infrastructure" / "self"
    news_title: str
    news_source: str
    news_published: str
    news_kind: str
    notes: list[str] = field(default_factory=list)


# Optional manual alias map — some nodes go by multiple names in the
# news. Expand as the registry grows.
NODE_ALIASES: dict[str, list[str]] = {
    "TSMC": ["TSMC", "Taiwan Semiconductor", "Taiwan Semiconductor Manufacturing"],
    "SK hynix": ["SK hynix", "SK Hynix", "Hynix"],
    "Samsung": ["Samsung", "Samsung Electronics"],
    "Micron": ["Micron", "Micron Technology"],
    "ASML": ["ASML"],
    "NVIDIA": ["NVIDIA", "NVDA", "Nvidia"],
    "NVDA": ["NVIDIA", "NVDA", "Nvidia"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "Broadcom": ["Broadcom", "AVGO"],
    "Intel": ["Intel", "INTC"],
    "GE Vernova": ["GE Vernova", "GEV"],
    "GEV": ["GE Vernova", "GEV"],
    "Eaton": ["Eaton", "ETN"],
    "Siemens": ["Siemens", "SMNEY"],
    "ABB": ["ABB", "ABBNY"],
    "Vestas": ["Vestas", "VWDRY"],
    "Hitachi": ["Hitachi", "HTHIY"],
    "Foxconn": ["Foxconn", "Hon Hai"],
    "Wiwynn": ["Wiwynn"],
    "Quanta": ["Quanta"],
    "Supermicro": ["Supermicro", "Super Micro"],
    "Synopsys": ["Synopsys"],
    "Cadence": ["Cadence", "Cadence Design"],
}


def _aliases_for(node_name: str, graph: ValueChainGraph) -> list[str]:
    """Return the alias list for a node. Defaults to [name, ticker] when
    no entry is in NODE_ALIASES.
    """
    if node_name in NODE_ALIASES:
        return NODE_ALIASES[node_name]
    aliases = [node_name]
    node = graph.get_company(node_name)
    if node and node.ticker and node.ticker != node_name:
        aliases.append(node.ticker)
    return aliases


def _title_matches(title: str, aliases: list[str]) -> str | None:
    """Return the first alias that appears as a word in `title`, else None."""
    lowered_title = title.lower()
    for alias in aliases:
        if not alias:
            continue
        # Word-boundary match to avoid "Micron" inside "micronesia".
        pattern = r"\b" + re.escape(alias.lower()) + r"\b"
        if re.search(pattern, lowered_title):
            return alias
    return None


def find_matching_nodes(
    graph: ValueChainGraph, news_items: list[NewsItemLike]
) -> list[tuple[str, NewsItemLike, str]]:
    """For each (node, news_item) pair where a node alias appears in the
    item's title, emit (node_name, news_item, matched_alias).
    """
    matches: list[tuple[str, NewsItemLike, str]] = []
    all_nodes = [n for n, _ in graph._g.nodes(data=True)]  # noqa: SLF001
    for node in all_nodes:
        aliases = _aliases_for(node, graph)
        for item in news_items:
            alias = _title_matches(item.title, aliases)
            if alias is not None:
                matches.append((node, item, alias))
    return matches


def _bfs_to_target(
    graph: ValueChainGraph,
    start: str,
    max_hops: int,
    direction: str,
) -> list[tuple[str, list[str], str]]:
    """BFS from `start` toward target nodes within `max_hops`. Returns
    list of (target_name, path_nodes, relation_type).

    `direction` is one of "out" (follow outgoing edges — start is
    upstream of the targets) or "in" (follow incoming edges — start
    is downstream / impacted by the targets).
    """
    results: list[tuple[str, list[str], str]] = []
    visited: set[str] = {start}
    queue: deque[tuple[str, list[str], str]] = deque([(start, [start], "")])

    while queue:
        node, path, first_relation = queue.popleft()
        if len(path) - 1 > max_hops:
            continue
        # Is the current node itself a target (other than start)?
        node_meta = graph.get_company(node)
        if node_meta is not None and node_meta.is_target and node != start:
            results.append((node, path, first_relation))
            # Don't return yet — other shorter paths may be at this
            # level still, and BFS guarantees shortest; if we wanted
            # only the closest we could prune deeper but one-target-
            # per-start would miss multi-hop structures. Keep going
            # to max_hops.
            continue

        if len(path) - 1 >= max_hops:
            continue

        if direction == "out":
            edges = list(graph._g.out_edges(node, data=True))  # noqa: SLF001
            neighbors = [(tgt, data.get("relation", "")) for _, tgt, data in edges]
        else:  # "in"
            edges = list(graph._g.in_edges(node, data=True))  # noqa: SLF001
            neighbors = [(src, data.get("relation", "")) for src, _, data in edges]

        for neighbor, relation in neighbors:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            rel = first_relation or relation
            queue.append((neighbor, path + [neighbor], rel))
    return results


def find_target_paths(
    graph: ValueChainGraph, matched_node: str, max_hops: int = 2
) -> list[tuple[str, list[str], str]]:
    """Return every target (is_target=True) node reachable from
    `matched_node` within `max_hops`, along with the path and the
    primary edge relation on the first hop.

    We check both directions:
      - OUT: matched_node → ... → target  (matched is upstream of target)
      - IN:  target → ... → matched_node  (matched is downstream of target)

    If `matched_node` is itself a target, emit a zero-hop self entry.
    """
    results: list[tuple[str, list[str], str]] = []
    node_meta = graph.get_company(matched_node)
    if node_meta is not None and node_meta.is_target:
        results.append((matched_node, [matched_node], "self"))

    out = _bfs_to_target(graph, matched_node, max_hops, "out")
    for target, path, relation in out:
        results.append((target, path, relation))

    in_ = _bfs_to_target(graph, matched_node, max_hops, "in")
    for target, path, relation in in_:
        # Reverse path so reader sees target → matched direction.
        results.append((target, list(reversed(path)), relation))

    return results


def scan_for_alerts(
    graph: ValueChainGraph,
    news_items: list[NewsItemLike],
    max_hops: int = 2,
) -> list[ChainAlert]:
    """Find every (news, target) pair where the news mentions a node in
    the graph within `max_hops` of a target ticker.

    Deduplicates on (target, matched_node) pairs — if the same news
    mentions two aliases of the same node, only the first match
    survives.
    """
    seen: set[tuple[str, str, str]] = set()
    alerts: list[ChainAlert] = []
    for matched_node, news, alias in find_matching_nodes(graph, news_items):
        paths = find_target_paths(graph, matched_node, max_hops=max_hops)
        for target, path, relation in paths:
            key = (target, matched_node, news.title)
            if key in seen:
                continue
            seen.add(key)
            alerts.append(
                ChainAlert(
                    target_symbol=target,
                    matched_node=matched_node,
                    chain_path=path,
                    hops=max(0, len(path) - 1),
                    relation=relation or "self",
                    news_title=news.title,
                    news_source=news.source,
                    news_published=news.published,
                    news_kind=news.kind,
                    notes=[f"Matched on alias: {alias!r}"],
                )
            )
    return alerts


def compose_alert_markdown(alerts: list[ChainAlert]) -> str:
    """Render a list of alerts as a single markdown block suitable for
    Telegram push or report append. Groups by target ticker.
    """
    if not alerts:
        return "_No chain alerts at this scan._"

    by_target: dict[str, list[ChainAlert]] = {}
    for a in alerts:
        by_target.setdefault(a.target_symbol, []).append(a)

    lines = ["# Chain alerts"]
    for target, items in sorted(by_target.items()):
        lines.append("")
        lines.append(f"## {target} ({len(items)} event(s))")
        for a in items:
            chain_str = " → ".join(a.chain_path)
            lines.append(
                f"- **{a.news_title}** — {a.news_source} "
                f"({a.news_published})"
            )
            lines.append(
                f"  - Chain (hops={a.hops}, relation={a.relation}): "
                f"`{chain_str}`"
            )
            if a.notes:
                for n in a.notes:
                    lines.append(f"  - _{n}_")
    return "\n".join(lines) + "\n"


__all__ = [
    "NODE_ALIASES",
    "ChainAlert",
    "NewsItemLike",
    "compose_alert_markdown",
    "find_matching_nodes",
    "find_target_paths",
    "scan_for_alerts",
]
