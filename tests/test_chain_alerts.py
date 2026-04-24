"""Tests for chain alerts — news × value chain graph → actionable events."""

from __future__ import annotations

from wise_investor.alerts.chain_alerts import (
    NODE_ALIASES,
    ChainAlert,
    NewsItemLike,
    compose_alert_markdown,
    find_matching_nodes,
    find_target_paths,
    scan_for_alerts,
)
from wise_investor.value_chain.graph import (
    CompanyNode,
    Relationship,
    ValueChainGraph,
)


# ---------------------------------------------------------------------------
# Fixture graph: NVDA target, supplied by TSMC, peer AMD
# ---------------------------------------------------------------------------


def _fixture_graph() -> ValueChainGraph:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA", is_target=True))
    g.add_company(CompanyNode(name="TSMC", ticker="TSM"))
    g.add_company(CompanyNode(name="AMD", ticker="AMD"))
    g.add_company(CompanyNode(name="ASML", ticker="ASML"))
    # TSMC → supplies → NVDA
    g.add_relationship(Relationship("TSMC", "NVDA", "supplies", source_doc="NVDA.md"))
    # ASML → supplies → TSMC (two-hop from ASML to NVDA)
    g.add_relationship(Relationship("ASML", "TSMC", "supplies", source_doc="NVDA.md"))
    # NVDA ↔ peer ↔ AMD (both directions)
    g.add_relationship(Relationship("NVDA", "AMD", "peer", source_doc="NVDA.md"))
    g.add_relationship(Relationship("AMD", "NVDA", "peer", source_doc="NVDA.md"))
    return g


def _news(
    title: str, source: str = "Reuters", published: str = "2026-04-24", kind: str = "news"
) -> NewsItemLike:
    return NewsItemLike(title=title, source=source, published=published, kind=kind)


# ---------------------------------------------------------------------------
# Alias matching
# ---------------------------------------------------------------------------


def test_find_matching_nodes_hits_tsmc_alias() -> None:
    g = _fixture_graph()
    items = [
        _news("Taiwan Semiconductor reports Kaohsiung fab outage"),
        _news("Unrelated economic commentary"),
    ]
    matches = find_matching_nodes(g, items)
    # "Taiwan Semiconductor" is in TSMC's alias list.
    assert any(node == "TSMC" for node, _, _ in matches)
    # Unrelated headline shouldn't match anything.
    unrelated_matches = [m for m in matches if m[1].title.startswith("Unrelated")]
    assert unrelated_matches == []


def test_find_matching_nodes_word_boundary_avoids_substring_false_positive() -> None:
    """Avoid matching "AMD" inside something like "amphetamide" or "amderson"."""
    g = _fixture_graph()
    items = [_news("Scientists isolate amderson compound in soil samples")]
    matches = find_matching_nodes(g, items)
    # No AMD match because word boundary requires standalone "AMD".
    amd_matches = [m for m in matches if m[0] == "AMD"]
    assert amd_matches == []


def test_find_matching_nodes_case_insensitive() -> None:
    g = _fixture_graph()
    items = [_news("nvidia announces new GPU")]
    matches = find_matching_nodes(g, items)
    # "NVDA" alias list includes "Nvidia" / "NVIDIA"; match should fire.
    assert any(node == "NVDA" for node, _, _ in matches)


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------


def test_find_target_paths_self_when_node_is_target() -> None:
    g = _fixture_graph()
    paths = find_target_paths(g, "NVDA", max_hops=2)
    assert ("NVDA", ["NVDA"], "self") in paths


def test_find_target_paths_one_hop_supplier_to_target() -> None:
    g = _fixture_graph()
    paths = find_target_paths(g, "TSMC", max_hops=2)
    # TSMC supplies NVDA — one hop out.
    direct = [(t, p, r) for t, p, r in paths if t == "NVDA"]
    assert direct
    target, path, relation = direct[0]
    assert path == ["TSMC", "NVDA"]
    assert relation == "supplies"


def test_find_target_paths_two_hop_indirect() -> None:
    g = _fixture_graph()
    # ASML → TSMC → NVDA  (ASML is two hops from NVDA).
    paths = find_target_paths(g, "ASML", max_hops=2)
    paths_to_nvda = [(t, p, r) for t, p, r in paths if t == "NVDA"]
    assert paths_to_nvda
    _, path, relation = paths_to_nvda[0]
    assert path == ["ASML", "TSMC", "NVDA"]
    # First-hop relation propagates.
    assert relation == "supplies"


def test_find_target_paths_respects_max_hops() -> None:
    g = _fixture_graph()
    # With max_hops=1, ASML should NOT reach NVDA (distance 2).
    paths = find_target_paths(g, "ASML", max_hops=1)
    to_nvda = [(t, p, r) for t, p, r in paths if t == "NVDA"]
    assert to_nvda == []


def test_find_target_paths_via_peer_edge() -> None:
    """AMD is a peer of NVDA (target). AMD news → NVDA should fire."""
    g = _fixture_graph()
    paths = find_target_paths(g, "AMD", max_hops=1)
    to_nvda = [(t, p, r) for t, p, r in paths if t == "NVDA"]
    assert to_nvda
    _, path, relation = to_nvda[0]
    assert "AMD" in path and "NVDA" in path
    assert relation == "peer"


# ---------------------------------------------------------------------------
# End-to-end scan_for_alerts
# ---------------------------------------------------------------------------


def test_scan_for_alerts_emits_nvda_alert_on_tsmc_news() -> None:
    g = _fixture_graph()
    news = [_news("TSMC cuts Q2 outlook on demand softness")]
    alerts = scan_for_alerts(g, news)
    nvda_alerts = [a for a in alerts if a.target_symbol == "NVDA"]
    assert nvda_alerts
    a = nvda_alerts[0]
    assert a.matched_node == "TSMC"
    assert "TSMC" in a.chain_path
    assert "NVDA" in a.chain_path
    assert a.hops == 1
    assert a.relation == "supplies"


def test_scan_for_alerts_dedupes_identical_target_node_news() -> None:
    g = _fixture_graph()
    news = [
        _news("TSMC ramp issues"),
        _news("TSMC ramp issues"),  # same title
    ]
    alerts = scan_for_alerts(g, news)
    assert len(alerts) == 1


def test_scan_for_alerts_emits_self_alert_when_target_mentioned() -> None:
    """News about NVDA itself should produce a self-alert (hops=0)."""
    g = _fixture_graph()
    news = [_news("NVIDIA's Q1 guidance disappoints")]
    alerts = scan_for_alerts(g, news)
    self_alerts = [a for a in alerts if a.relation == "self"]
    assert self_alerts
    a = self_alerts[0]
    assert a.target_symbol == "NVDA"
    assert a.hops == 0


def test_scan_for_alerts_empty_on_irrelevant_news() -> None:
    g = _fixture_graph()
    news = [_news("Weather alert in Tokyo")]
    assert scan_for_alerts(g, news) == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_compose_alert_markdown_renders_grouped_by_target() -> None:
    alerts = [
        ChainAlert(
            target_symbol="NVDA",
            matched_node="TSMC",
            chain_path=["TSMC", "NVDA"],
            hops=1,
            relation="supplies",
            news_title="TSMC outage",
            news_source="Reuters",
            news_published="2026-04-24",
            news_kind="google_news",
            notes=["Matched on alias: 'TSMC'"],
        ),
        ChainAlert(
            target_symbol="GEV",
            matched_node="Siemens",
            chain_path=["Siemens", "GEV"],
            hops=1,
            relation="peer",
            news_title="Siemens profit warning",
            news_source="Bloomberg",
            news_published="2026-04-23",
            news_kind="google_news",
        ),
    ]
    md = compose_alert_markdown(alerts)
    assert "# Chain alerts" in md
    assert "## NVDA" in md
    assert "## GEV" in md
    assert "TSMC outage" in md
    assert "TSMC → NVDA" in md
    assert "supplies" in md


def test_compose_alert_markdown_empty_state() -> None:
    md = compose_alert_markdown([])
    assert "No chain alerts" in md


# ---------------------------------------------------------------------------
# Alias registry sanity
# ---------------------------------------------------------------------------


def test_node_aliases_contains_known_critical_nodes() -> None:
    """If someone removes TSMC / ASML / HBM suppliers from the alias map,
    chain alerts silently go dead for those nodes. Regression guard.
    """
    for required in ["TSMC", "ASML", "SK hynix", "Samsung", "Micron", "NVIDIA"]:
        assert required in NODE_ALIASES, f"missing alias entry: {required}"
