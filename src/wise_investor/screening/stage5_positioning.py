"""Stage 5 — value chain positioning of rubric-passing survivors.

Constitution Sec 6 defines Stage 5 as the post-Stage-4 placement of
survivors onto a global value chain graph. The graph itself is a
hand-authored artifact (see `wise_investor.value_chain.graph` and the
briefs under `docs/value_chains/`); this module is the consumer that
takes a list of survivor tickers and produces:

  - per-survivor position: which node they map to on the graph and
    who their peers, suppliers, customers, and infrastructure providers
    are
  - cluster occupancy: which graph clusters contain how many survivors
  - over- / under-representation: clusters where survivors concentrate
    disproportionately, plus clusters with graph nodes but no survivors

Why the cluster analysis matters:

  1. Stage 6 (HRP) uses the cluster information for post-hoc
     down-weighting — two HRP-favored names sitting on the same
     cluster get the smaller position trimmed (constitution Sec 6).

  2. The user reads the over/under-representation report to spot
     systematic blind spots — a constitution that reliably picks
     semiconductor names while ignoring industrial bottlenecks is
     useful information that the price-correlation HRP layer can't
     surface.

Cluster detection uses NetworkX greedy modularity communities on the
undirected projection of the graph. The algorithm is deterministic for
a given input so calibration runs are reproducible across constitution
versions.

Survivors that don't map to any graph node are reported separately —
the graph is hand-curated and won't cover every passable ticker. The
report carries that count so callers can see how much of the survivor
pool is positionable today.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import networkx as nx

from wise_investor.value_chain.graph import ValueChainGraph

logger = logging.getLogger(__name__)


# Threshold above which a cluster is flagged as "over-represented"
# in the survivor pool. 1.0 means the cluster's survivor fraction
# matches its graph fraction; 1.5 means survivors are 1.5× more
# concentrated in this cluster than uniform distribution would imply.
DEFAULT_OVER_REPRESENTATION_THRESHOLD: float = 1.5

# Below this, a cluster is flagged as "under-represented." 0.5 means
# survivors land in this cluster at half the rate uniform distribution
# would imply.
DEFAULT_UNDER_REPRESENTATION_THRESHOLD: float = 0.5


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurvivorPosition:
    """One survivor's position within the value chain graph."""

    ticker: str
    node_name: str | None       # None when ticker doesn't map to any graph node
    industry: str | None
    cluster_id: int | None
    peers: tuple[str, ...]
    suppliers: tuple[str, ...]
    customers: tuple[str, ...]
    infrastructure: tuple[str, ...]


@dataclass(frozen=True)
class ClusterStat:
    """Representation stats for one cluster of the graph."""

    cluster_id: int
    member_names: tuple[str, ...]
    n_members: int                # total graph nodes in the cluster
    survivor_tickers: tuple[str, ...]
    n_survivors: int              # how many survivors map here
    expected_survivors: float     # uniform-distribution expectation
    over_representation: float    # n_survivors / expected_survivors;
                                  # 1.0 = neutral, > 1 = concentrated
    flag: str                     # "over", "under", or "neutral"


@dataclass(frozen=True)
class Stage5PositioningReport:
    """Aggregate output of Stage 5 positioning."""

    survivor_positions: tuple[SurvivorPosition, ...]
    clusters: tuple[ClusterStat, ...]
    n_total_survivors: int
    n_mapped_survivors: int
    n_unmapped_survivors: int
    total_graph_nodes: int

    @property
    def unmapped_tickers(self) -> tuple[str, ...]:
        return tuple(
            p.ticker for p in self.survivor_positions if p.node_name is None
        )

    @property
    def overrepresented_clusters(self) -> tuple[ClusterStat, ...]:
        return tuple(c for c in self.clusters if c.flag == "over")

    @property
    def underrepresented_clusters(self) -> tuple[ClusterStat, ...]:
        return tuple(c for c in self.clusters if c.flag == "under")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def position_survivors(
    survivors: list[str],
    graph: ValueChainGraph,
    *,
    over_representation_threshold: float = DEFAULT_OVER_REPRESENTATION_THRESHOLD,
    under_representation_threshold: float = DEFAULT_UNDER_REPRESENTATION_THRESHOLD,
) -> Stage5PositioningReport:
    """Position a list of survivor tickers on the value chain graph.

    Args:
        survivors: Stage 4 survivor tickers (uppercase preferred but
            case-insensitive lookup applies).
        graph: A loaded `ValueChainGraph`.
        over_representation_threshold: Ratio above which a cluster is
            flagged "over". Default 1.5×.
        under_representation_threshold: Ratio below which a cluster is
            flagged "under". Default 0.5×.
    """
    # Map each graph node to a cluster id. We do this once for the whole
    # graph; per-survivor lookup is then O(1).
    name_to_cluster, clusters_members = _detect_clusters(graph)

    # Resolve each survivor to a graph node name (when possible).
    positions: list[SurvivorPosition] = []
    cluster_to_survivors: dict[int, list[str]] = {
        cid: [] for cid in clusters_members
    }
    for ticker in survivors:
        node_name = _resolve_ticker_to_node(graph, ticker)
        if node_name is None:
            positions.append(
                SurvivorPosition(
                    ticker=ticker,
                    node_name=None,
                    industry=None,
                    cluster_id=None,
                    peers=(),
                    suppliers=(),
                    customers=(),
                    infrastructure=(),
                )
            )
            continue

        company = graph.get_company(node_name)
        cluster_id = name_to_cluster.get(node_name)
        positions.append(
            SurvivorPosition(
                ticker=ticker,
                node_name=node_name,
                industry=company.industry if company else None,
                cluster_id=cluster_id,
                peers=tuple(graph.peers_of(node_name)),
                suppliers=tuple(graph.suppliers_of(node_name)),
                customers=tuple(graph.customers_of(node_name)),
                infrastructure=tuple(graph.infrastructure_of(node_name)),
            )
        )
        if cluster_id is not None:
            cluster_to_survivors[cluster_id].append(ticker)

    n_total = len(survivors)
    n_mapped = sum(1 for p in positions if p.node_name is not None)
    n_unmapped = n_total - n_mapped
    total_nodes = graph.num_nodes

    # Compute representation stats per cluster.
    cluster_stats: list[ClusterStat] = []
    for cid, members in clusters_members.items():
        n_members = len(members)
        n_survivors = len(cluster_to_survivors[cid])
        # Expected survivors if mapped survivors were uniformly distributed:
        #   expected = (cluster_size / total_nodes) * n_mapped_survivors
        if total_nodes == 0 or n_mapped == 0:
            expected = 0.0
            ratio = 1.0
        else:
            expected = (n_members / total_nodes) * n_mapped
            # Avoid division by zero: when expected is 0 (cluster too
            # small for any survivor under uniform allocation), flag
            # over-representation only if n_survivors > 0.
            if expected == 0:
                ratio = float("inf") if n_survivors > 0 else 1.0
            else:
                ratio = n_survivors / expected

        if ratio >= over_representation_threshold and n_survivors > 0:
            flag = "over"
        elif (
            ratio <= under_representation_threshold
            and n_members > 0
        ):
            flag = "under"
        else:
            flag = "neutral"

        cluster_stats.append(
            ClusterStat(
                cluster_id=cid,
                member_names=tuple(sorted(members)),
                n_members=n_members,
                survivor_tickers=tuple(sorted(cluster_to_survivors[cid])),
                n_survivors=n_survivors,
                expected_survivors=expected,
                over_representation=ratio,
                flag=flag,
            )
        )

    # Sort clusters: over-represented first, then by ratio desc.
    cluster_stats.sort(
        key=lambda c: (
            0 if c.flag == "over" else 1 if c.flag == "neutral" else 2,
            -c.over_representation,
        )
    )

    return Stage5PositioningReport(
        survivor_positions=tuple(positions),
        clusters=tuple(cluster_stats),
        n_total_survivors=n_total,
        n_mapped_survivors=n_mapped,
        n_unmapped_survivors=n_unmapped,
        total_graph_nodes=total_nodes,
    )


# ---------------------------------------------------------------------------
# Ticker → node resolution
# ---------------------------------------------------------------------------


def _resolve_ticker_to_node(
    graph: ValueChainGraph, ticker: str
) -> str | None:
    """Find the graph node name corresponding to a ticker symbol.

    Resolution order:
      1. Direct name match (some briefs use the ticker as the company
         name, e.g. "NVDA" or "AMD").
      2. ticker attribute match — iterate nodes and pick the one whose
         `ticker` field matches case-insensitively.
      3. None when no match found.
    """
    if not ticker:
        return None
    sym = ticker.upper().strip()

    # Direct name match.
    if graph.has_company(sym):
        return sym

    # Ticker attribute match.
    for name, attrs in graph._g.nodes(data=True):  # noqa: SLF001
        node_ticker = attrs.get("ticker")
        if node_ticker and node_ticker.upper() == sym:
            return name

    return None


# ---------------------------------------------------------------------------
# Cluster detection
# ---------------------------------------------------------------------------


def _detect_clusters(
    graph: ValueChainGraph,
) -> tuple[dict[str, int], dict[int, list[str]]]:
    """Run greedy modularity community detection on the undirected
    projection of the graph.

    Returns:
      - name_to_cluster: node name → cluster id
      - clusters_members: cluster id → list of node names

    Empty graphs yield two empty dicts.
    """
    if graph.num_nodes == 0:
        return ({}, {})

    # Project to undirected for community detection — peer edges are
    # bidirectional in our model and supply/customer edges describe a
    # functional relationship that's still a cluster signal regardless
    # of direction.
    undirected: nx.Graph = graph._g.to_undirected()  # noqa: SLF001

    # Handle isolated nodes: greedy_modularity treats them as their own
    # singletons, which is fine for our representation analysis.
    try:
        communities = nx.community.greedy_modularity_communities(undirected)
    except (nx.NetworkXError, ValueError) as e:
        # Edge cases (very small graphs) can confuse the algorithm.
        # Fall back to connected components.
        logger.debug("greedy_modularity failed (%s); falling back to components", e)
        communities = list(nx.connected_components(undirected))

    name_to_cluster: dict[str, int] = {}
    clusters_members: dict[int, list[str]] = {}
    for cid, community in enumerate(communities):
        members = sorted(community)
        clusters_members[cid] = members
        for name in members:
            name_to_cluster[name] = cid

    return (name_to_cluster, clusters_members)


# ---------------------------------------------------------------------------
# Convenience: representation summary as a dict (for ledger / JSON output)
# ---------------------------------------------------------------------------


def representation_summary(report: Stage5PositioningReport) -> dict:
    """Project a positioning report into a flat JSON-friendly summary.

    Structure mirrors the calibration ledger's per-record shape so a
    Stage 5 run can be appended to the ledger without further reshaping.
    """
    industry_counter: Counter[str] = Counter()
    for p in report.survivor_positions:
        if p.industry:
            industry_counter[p.industry] += 1

    return {
        "n_total_survivors": report.n_total_survivors,
        "n_mapped_survivors": report.n_mapped_survivors,
        "n_unmapped_survivors": report.n_unmapped_survivors,
        "unmapped_tickers": list(report.unmapped_tickers),
        "total_graph_nodes": report.total_graph_nodes,
        "industry_distribution": dict(industry_counter.most_common()),
        "clusters": [
            {
                "cluster_id": c.cluster_id,
                "n_members": c.n_members,
                "n_survivors": c.n_survivors,
                "expected_survivors": c.expected_survivors,
                "over_representation": (
                    c.over_representation
                    if c.over_representation != float("inf")
                    else None
                ),
                "flag": c.flag,
                "survivor_tickers": list(c.survivor_tickers),
            }
            for c in report.clusters
        ],
    }


__all__ = [
    "ClusterStat",
    "DEFAULT_OVER_REPRESENTATION_THRESHOLD",
    "DEFAULT_UNDER_REPRESENTATION_THRESHOLD",
    "Stage5PositioningReport",
    "SurvivorPosition",
    "position_survivors",
    "representation_summary",
]
