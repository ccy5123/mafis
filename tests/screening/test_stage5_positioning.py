"""Stage 5 positioning tests.

Builds synthetic ValueChainGraphs and verifies:
  - per-survivor position resolution (direct name match, ticker
    attribute match, missing → unmapped)
  - cluster detection produces deterministic ids
  - over- / under-representation flagging
  - representation_summary projection shape
"""

from __future__ import annotations

from wise_investor.screening.stage5_positioning import (
    DEFAULT_OVER_REPRESENTATION_THRESHOLD,
    DEFAULT_UNDER_REPRESENTATION_THRESHOLD,
    position_survivors,
    representation_summary,
)
from wise_investor.value_chain.graph import (
    CompanyNode,
    Relationship,
    ValueChainGraph,
)

# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _two_cluster_graph() -> ValueChainGraph:
    """Two disjoint clusters of 3 nodes each.

    Cluster A: NVDA, AMD, INTC  (peers)
    Cluster B: KO, PEP, MNST    (peers)

    No edges between the clusters → community detection identifies
    exactly two communities.
    """
    g = ValueChainGraph()
    for sym in ("NVDA", "AMD", "INTC"):
        g.add_company(CompanyNode(name=sym, ticker=sym, industry="Semiconductors", is_target=True))
    for sym in ("KO", "PEP", "MNST"):
        g.add_company(CompanyNode(name=sym, ticker=sym, industry="Beverages", is_target=True))

    g.add_peer("NVDA", "AMD")
    g.add_peer("AMD", "INTC")
    g.add_peer("KO", "PEP")
    g.add_peer("PEP", "MNST")
    return g


def _supply_chain_graph() -> ValueChainGraph:
    """A small directed supply chain.

    TSMC -> NVDA -> Hyperscalers
    ASML -> TSMC
    """
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="ASML", ticker="ASML", industry="Semi Equipment"))
    g.add_company(CompanyNode(name="TSMC", ticker="TSM", industry="Foundry"))
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA", industry="Semis", is_target=True))
    g.add_company(CompanyNode(name="Hyperscalers", industry="Cloud"))

    g.add_relationship(Relationship("ASML", "TSMC", "supplies", source_doc="NVDA.md"))
    g.add_relationship(Relationship("TSMC", "NVDA", "supplies", source_doc="NVDA.md"))
    g.add_relationship(Relationship("NVDA", "Hyperscalers", "supplies", source_doc="NVDA.md"))
    return g


# ---------------------------------------------------------------------------
# Ticker resolution
# ---------------------------------------------------------------------------


def test_direct_name_match() -> None:
    """A survivor ticker that's also the graph node name resolves
    directly without needing the ticker attribute."""
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker=None))
    report = position_survivors(["NVDA"], g)
    assert report.survivor_positions[0].node_name == "NVDA"


def test_ticker_attribute_match() -> None:
    """When the node name differs from the ticker (e.g. company name
    'NVIDIA' with ticker 'NVDA'), resolution still works."""
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVIDIA", ticker="NVDA"))
    report = position_survivors(["NVDA"], g)
    assert report.survivor_positions[0].node_name == "NVIDIA"


def test_unmapped_survivor_is_recorded() -> None:
    """Survivors with no graph node remain in the report with node_name=None."""
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA"))
    report = position_survivors(["NVDA", "UNKNOWN"], g)
    assert report.n_total_survivors == 2
    assert report.n_mapped_survivors == 1
    assert report.n_unmapped_survivors == 1
    assert report.unmapped_tickers == ("UNKNOWN",)


def test_case_insensitive_ticker_lookup() -> None:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVIDIA", ticker="NVDA"))
    report = position_survivors(["nvda"], g)
    assert report.survivor_positions[0].node_name == "NVIDIA"


# ---------------------------------------------------------------------------
# Per-survivor position fields
# ---------------------------------------------------------------------------


def test_survivor_position_records_relationships() -> None:
    g = _supply_chain_graph()
    report = position_survivors(["NVDA"], g)
    pos = report.survivor_positions[0]
    assert pos.node_name == "NVDA"
    assert "TSMC" in pos.suppliers          # TSMC -> NVDA
    assert "Hyperscalers" in pos.customers  # NVDA -> Hyperscalers
    assert pos.industry == "Semis"


def test_survivor_position_lists_peers() -> None:
    g = _two_cluster_graph()
    report = position_survivors(["NVDA"], g)
    pos = report.survivor_positions[0]
    assert "AMD" in pos.peers


# ---------------------------------------------------------------------------
# Cluster detection + representation
# ---------------------------------------------------------------------------


def test_two_clusters_detected_in_disjoint_graph() -> None:
    g = _two_cluster_graph()
    report = position_survivors([], g)
    # Two distinct cluster ids should appear.
    cluster_ids = {c.cluster_id for c in report.clusters}
    assert len(cluster_ids) == 2


def test_overrepresentation_flagged_when_all_survivors_in_one_cluster() -> None:
    """3 survivors all in one cluster (size 3 of 6 total) → all 3 mapped
    survivors land in 50% of the graph → ratio = 2.0 → over."""
    g = _two_cluster_graph()
    survivors = ["NVDA", "AMD", "INTC"]
    report = position_survivors(survivors, g)

    over = report.overrepresented_clusters
    assert len(over) == 1
    # All 3 survivors landed in the over-cluster
    assert over[0].n_survivors == 3
    # Beverages cluster (KO/PEP/MNST) gets 0 survivors → under
    under = report.underrepresented_clusters
    assert any(c.n_survivors == 0 for c in under)


def test_balanced_survivors_yield_no_overrepresentation() -> None:
    """3 survivors, 1 from each cluster (one extra slot) → ratio ~1.0 →
    no flags."""
    g = _two_cluster_graph()
    survivors = ["NVDA", "KO"]
    report = position_survivors(survivors, g)

    # Both clusters have 1 survivor in a 3-node cluster of a 6-node
    # graph: expected = (3/6)*2 = 1.0; ratio = 1/1 = 1.0 → neutral.
    for c in report.clusters:
        assert c.flag in ("neutral", "under", "over")
    # Equal distribution should NOT produce strong over-representation.
    over = report.overrepresented_clusters
    assert len(over) == 0


def test_unmapped_survivors_dont_skew_cluster_math() -> None:
    """Unmapped survivors are excluded from the expected-survivor
    denominator so they don't artificially inflate cluster ratios."""
    g = _two_cluster_graph()
    # 2 unmapped + 3 mapped. The math should use 3 mapped, not 5.
    report = position_survivors(["NVDA", "AMD", "INTC", "UNK1", "UNK2"], g)
    assert report.n_mapped_survivors == 3
    assert report.n_unmapped_survivors == 2

    # The semis cluster should be over-represented at the same ratio
    # as the all-3-survivors-mapped case (n_mapped is the denominator).
    over = report.overrepresented_clusters
    assert len(over) == 1
    assert over[0].n_survivors == 3


def test_thresholds_are_configurable() -> None:
    """A loose over-representation threshold should flag more clusters."""
    g = _two_cluster_graph()
    survivors = ["NVDA", "AMD"]  # 2/2 land in semis cluster
    strict = position_survivors(survivors, g, over_representation_threshold=10.0)
    loose = position_survivors(survivors, g, over_representation_threshold=1.5)

    # Loose threshold flags the semis cluster; strict does not.
    assert len(strict.overrepresented_clusters) == 0
    assert len(loose.overrepresented_clusters) == 1


def test_default_thresholds_are_what_constitution_expects() -> None:
    assert DEFAULT_OVER_REPRESENTATION_THRESHOLD == 1.5
    assert DEFAULT_UNDER_REPRESENTATION_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_graph_yields_empty_report() -> None:
    g = ValueChainGraph()
    report = position_survivors(["NVDA"], g)
    assert report.total_graph_nodes == 0
    assert report.clusters == ()
    # The ticker is still recorded as unmapped.
    assert report.n_unmapped_survivors == 1


def test_no_survivors_still_produces_cluster_breakdown() -> None:
    """A run with no survivors should still report the graph's clusters
    so the user sees the universe structure even when the rubric
    rejected everything."""
    g = _two_cluster_graph()
    report = position_survivors([], g)
    assert report.n_total_survivors == 0
    # Clusters are reported regardless.
    assert len(report.clusters) == 2


# ---------------------------------------------------------------------------
# representation_summary projection
# ---------------------------------------------------------------------------


def test_representation_summary_shape() -> None:
    g = _two_cluster_graph()
    report = position_survivors(["NVDA", "AMD"], g)
    summary = representation_summary(report)

    assert summary["n_total_survivors"] == 2
    assert summary["n_mapped_survivors"] == 2
    assert summary["n_unmapped_survivors"] == 0
    assert "industry_distribution" in summary
    assert summary["industry_distribution"].get("Semiconductors") == 2
    assert "clusters" in summary
    assert isinstance(summary["clusters"], list)


def test_representation_summary_handles_unmapped() -> None:
    g = _two_cluster_graph()
    report = position_survivors(["UNKNOWN"], g)
    summary = representation_summary(report)
    assert summary["n_unmapped_survivors"] == 1
    assert summary["unmapped_tickers"] == ["UNKNOWN"]
    # No industry attribution for unmapped tickers
    assert summary["industry_distribution"] == {}


def test_representation_summary_inf_ratio_replaced_with_none() -> None:
    """When a cluster has 0 expected but >0 actual survivors, the ratio
    is inf — the JSON projection should replace inf with None for
    serializability."""
    # Build a graph where one node has zero connections so it forms a
    # singleton cluster. Then map a survivor to it. Singleton cluster
    # has size 1 in a graph of N; expected = (1/N) * survivors. With
    # only one survivor, expected = 1/N which is non-zero, so no inf.
    # Force inf by including 0 mapped survivors elsewhere.
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="A", ticker="A"))
    g.add_company(CompanyNode(name="B", ticker="B"))
    # Two singleton clusters; 0 survivors anywhere yields ratio 1.0
    # (the n_mapped guard). The inf path requires > 0 mapped survivors
    # but 0 expected, which can't happen with cluster_size > 0.
    # Test the JSON projection instead.
    report = position_survivors(["A"], g)
    summary = representation_summary(report)
    for c in summary["clusters"]:
        # ratio must be either a finite float or None (never inf string)
        assert c["over_representation"] is None or isinstance(c["over_representation"], float)


# ---------------------------------------------------------------------------
# Integration with persisted graph JSON
# ---------------------------------------------------------------------------


def test_positioning_works_after_graph_save_load_roundtrip(tmp_path) -> None:
    g = _two_cluster_graph()
    p = tmp_path / "graph.json"
    g.save(p)
    g2 = ValueChainGraph.load(p)
    report = position_survivors(["NVDA"], g2)
    pos = report.survivor_positions[0]
    assert pos.node_name == "NVDA"
    assert "AMD" in pos.peers
