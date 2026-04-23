"""Tests for ValueChainGraph and the markdown parser."""

from __future__ import annotations

from pathlib import Path

from wise_investor.value_chain.graph import (
    CompanyNode,
    Relationship,
    ValueChainGraph,
)
from wise_investor.value_chain.parser import (
    _clean_name,
    _slice_section,
    build_graph_from_briefs,
    parse_value_chain_markdown,
)


# ---------------------------------------------------------------------------
# Graph mechanics
# ---------------------------------------------------------------------------


def test_graph_add_company_and_update() -> None:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA"))
    # Re-adding should not duplicate but should fill in missing fields.
    g.add_company(CompanyNode(name="NVDA", industry="Semiconductors"))
    node = g.get_company("NVDA")
    assert node is not None
    assert node.ticker == "NVDA"
    assert node.industry == "Semiconductors"
    assert g.num_nodes == 1


def test_graph_promote_to_target_monotonically() -> None:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA"))
    assert g.get_company("NVDA").is_target is False  # type: ignore[union-attr]
    g.add_company(CompanyNode(name="NVDA", is_target=True))
    assert g.get_company("NVDA").is_target is True  # type: ignore[union-attr]
    # Setting back to False must NOT downgrade.
    g.add_company(CompanyNode(name="NVDA", is_target=False))
    assert g.get_company("NVDA").is_target is True  # type: ignore[union-attr]


def test_graph_add_relationship_auto_creates_nodes() -> None:
    g = ValueChainGraph()
    g.add_relationship(Relationship("TSMC", "NVDA", "supplies"))
    assert g.has_company("TSMC")
    assert g.has_company("NVDA")
    assert g.suppliers_of("NVDA") == ["TSMC"]


def test_graph_peer_is_bidirectional() -> None:
    g = ValueChainGraph()
    g.add_peer("NVDA", "AMD")
    assert "AMD" in g.peers_of("NVDA")
    assert "NVDA" in g.peers_of("AMD")


def test_graph_customers_direction() -> None:
    g = ValueChainGraph()
    # NVDA supplies Microsoft (downstream from target's POV).
    g.add_relationship(Relationship("NVDA", "Microsoft Azure", "supplies"))
    assert g.customers_of("NVDA") == ["Microsoft Azure"]
    assert g.suppliers_of("Microsoft Azure") == ["NVDA"]


def test_graph_infrastructure() -> None:
    g = ValueChainGraph()
    g.add_relationship(Relationship("Power utility", "NVDA", "infrastructure"))
    assert g.infrastructure_of("NVDA") == ["Power utility"]


def test_graph_save_and_load_roundtrip(tmp_path: Path) -> None:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA", is_target=True))
    g.add_relationship(
        Relationship("TSMC", "NVDA", "supplies", source_doc="NVDA.md", notes="fab")
    )
    g.add_peer("NVDA", "AMD", source_doc="NVDA.md")

    path = tmp_path / "g.json"
    g.save(path)

    loaded = ValueChainGraph.load(path)
    assert loaded.num_nodes == g.num_nodes
    assert loaded.num_edges == g.num_edges
    assert loaded.suppliers_of("NVDA") == ["TSMC"]
    assert "AMD" in loaded.peers_of("NVDA")
    assert loaded.get_company("NVDA").is_target is True  # type: ignore[union-attr]


def test_graph_targets_lists_only_flagged_nodes() -> None:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", is_target=True))
    g.add_company(CompanyNode(name="GEV", is_target=True))
    g.add_company(CompanyNode(name="TSMC"))
    assert g.targets() == ["GEV", "NVDA"]


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


def test_clean_name_extracts_parenthesized_ticker() -> None:
    name, ticker = _clean_name("Siemens Energy (ENR.DE)")
    assert name == "Siemens Energy"
    assert ticker == "ENR.DE"


def test_clean_name_without_ticker() -> None:
    name, ticker = _clean_name("TSMC")
    assert name == "TSMC"
    assert ticker is None


def test_slice_section_returns_section_body() -> None:
    md = (
        "# Title\n\n"
        "## Upstream — Suppliers\n"
        "- **TSMC** — chip fab\n"
        "## Peers\n"
        "- **AMD** — rival\n"
    )
    body = _slice_section(md, "upstream")
    assert "TSMC" in body
    assert "AMD" not in body  # stops at next ## heading


def test_slice_section_missing_returns_empty() -> None:
    assert _slice_section("# Empty doc", "upstream") == ""


# ---------------------------------------------------------------------------
# Parser integration on synthetic brief
# ---------------------------------------------------------------------------


_FIXTURE_BRIEF = """\
# NVDA Value Chain

## Upstream — Suppliers

### Chip fabrication

- **TSMC** — single external foundry for all leading-edge nodes.
- **SK hynix** — HBM3e primary supplier.

## Peers — Direct and adjacent competition

| Peer | Ticker | Overlap | Threat |
|------|--------|---------|-------:|
| AMD | AMD | MI300 | High |
| Intel | INTC | Gaudi | Low |

## Downstream — Customers

- **Microsoft Azure** — hyperscale cloud.
- **AWS** — hyperscale cloud.

## Infrastructure / Regulatory

- **Power utility** — data center electricity.
"""


def test_parse_value_chain_markdown_picks_up_sections() -> None:
    rels = parse_value_chain_markdown(
        _FIXTURE_BRIEF, target_symbol="NVDA", source_doc="NVDA.md"
    )

    # Upstream -> NVDA
    upstream = [r.source for r in rels if r.target == "NVDA" and r.relation == "supplies"]
    assert "TSMC" in upstream
    assert "SK hynix" in upstream

    # Peers: bidirectional between NVDA and AMD/Intel
    peers_of_nvda = [r.target for r in rels if r.source == "NVDA" and r.relation == "peer"]
    assert "AMD" in peers_of_nvda
    assert "Intel" in peers_of_nvda

    # Downstream: NVDA -> customer
    customers = [r.target for r in rels if r.source == "NVDA" and r.relation == "supplies"]
    assert "Microsoft Azure" in customers
    assert "AWS" in customers

    # Infrastructure -> NVDA
    infra = [r.source for r in rels if r.target == "NVDA" and r.relation == "infrastructure"]
    assert "Power utility" in infra


def test_parse_picks_up_source_doc_attribution() -> None:
    rels = parse_value_chain_markdown(
        _FIXTURE_BRIEF, target_symbol="NVDA", source_doc="NVDA.md"
    )
    assert all(r.source_doc == "NVDA.md" for r in rels)


def test_build_graph_from_real_briefs_dir() -> None:
    briefs_dir = Path(__file__).resolve().parents[1] / "docs" / "value_chains"
    g = build_graph_from_briefs(briefs_dir)

    # Both Tier 1 tickers should be targets.
    targets = g.targets()
    assert "NVDA" in targets
    assert "GEV" in targets

    # NVDA should have some supplier + peer structure from the real doc.
    assert len(g.suppliers_of("NVDA")) > 0
    assert len(g.peers_of("NVDA")) > 0


def test_build_graph_skips_readme() -> None:
    briefs_dir = Path(__file__).resolve().parents[1] / "docs" / "value_chains"
    g = build_graph_from_briefs(briefs_dir)
    # docs/value_chains/README.md exists but must not become a node.
    assert not g.has_company("README")
