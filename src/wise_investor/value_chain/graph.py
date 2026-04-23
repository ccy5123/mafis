"""NetworkX-backed value chain graph (design-v2.2 §5.1).

Nodes are companies; each node carries a name (string), an optional
ticker symbol, and an optional industry label. Edges are directed and
typed:

  "supplies"       A -> B  means A supplies to B (A upstream of B).
                           Used for Upstream (supplier -> target) and
                           Downstream (target -> customer) sections.
  "peer"           A <-> B means A and B compete in the same market.
                           Emitted as two directed edges so either side
                           can find the other via graph.neighbors().
  "infrastructure" A -> B  means A provides infra support to B
                           (power, cooling, bearings, EDA, etc.).

Each edge records `source_doc` — the docs/value_chains/<SYMBOL>.md file
the relationship was parsed from — so the graph is always auditable
back to a curated human source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx


RelationType = str  # one of "supplies" / "peer" / "infrastructure"


@dataclass
class CompanyNode:
    """A vertex in the value chain graph.

    `name` is the required stable key. `ticker` is optional — many
    upstream suppliers (Synopsys, Cadence, SKF) lack a clean matching
    symbol in our data sources, and that is fine.
    """

    name: str
    ticker: str | None = None
    industry: str | None = None
    is_target: bool = False  # True when this node has its own value chain brief


@dataclass
class Relationship:
    source: str   # CompanyNode.name of the provider
    target: str   # CompanyNode.name of the recipient
    relation: RelationType
    source_doc: str | None = None  # e.g. "NVDA.md"
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class ValueChainGraph:
    """High-level wrapper around nx.DiGraph with typed edges."""

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()

    # ---- mutation -----------------------------------------------------

    def add_company(self, node: CompanyNode) -> None:
        """Add a company or update metadata on an existing one."""
        name = node.name.strip()
        if not name:
            raise ValueError("company name cannot be empty")
        if name in self._g:
            existing: dict[str, Any] = self._g.nodes[name]
            # Preserve fields that were already set; promote to target if either side says so.
            existing.setdefault("ticker", node.ticker)
            if node.ticker and not existing.get("ticker"):
                existing["ticker"] = node.ticker
            existing.setdefault("industry", node.industry)
            if node.industry and not existing.get("industry"):
                existing["industry"] = node.industry
            existing["is_target"] = existing.get("is_target", False) or node.is_target
        else:
            self._g.add_node(
                name,
                ticker=node.ticker,
                industry=node.industry,
                is_target=node.is_target,
            )

    def add_relationship(self, rel: Relationship) -> None:
        """Insert a typed edge. Ensures both endpoints exist as nodes first."""
        if rel.source not in self._g:
            self.add_company(CompanyNode(name=rel.source))
        if rel.target not in self._g:
            self.add_company(CompanyNode(name=rel.target))
        self._g.add_edge(
            rel.source,
            rel.target,
            relation=rel.relation,
            source_doc=rel.source_doc,
            notes=rel.notes,
            **rel.extra,
        )

    def add_peer(self, a: str, b: str, source_doc: str | None = None) -> None:
        """Peer relationships are bidirectional; store both directions."""
        self.add_relationship(Relationship(a, b, "peer", source_doc=source_doc))
        self.add_relationship(Relationship(b, a, "peer", source_doc=source_doc))

    # ---- queries ------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    def has_company(self, name: str) -> bool:
        return name in self._g

    def get_company(self, name: str) -> CompanyNode | None:
        if name not in self._g:
            return None
        d = self._g.nodes[name]
        return CompanyNode(
            name=name,
            ticker=d.get("ticker"),
            industry=d.get("industry"),
            is_target=d.get("is_target", False),
        )

    def edges_from(
        self, name: str, relation: RelationType | None = None
    ) -> list[Relationship]:
        out: list[Relationship] = []
        if name not in self._g:
            return out
        for src, tgt, data in self._g.out_edges(name, data=True):
            if relation is not None and data.get("relation") != relation:
                continue
            out.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relation=data.get("relation", ""),
                    source_doc=data.get("source_doc"),
                    notes=data.get("notes"),
                )
            )
        return out

    def edges_to(
        self, name: str, relation: RelationType | None = None
    ) -> list[Relationship]:
        out: list[Relationship] = []
        if name not in self._g:
            return out
        for src, tgt, data in self._g.in_edges(name, data=True):
            if relation is not None and data.get("relation") != relation:
                continue
            out.append(
                Relationship(
                    source=src,
                    target=tgt,
                    relation=data.get("relation", ""),
                    source_doc=data.get("source_doc"),
                    notes=data.get("notes"),
                )
            )
        return out

    def suppliers_of(self, name: str) -> list[str]:
        """Companies that supply to `name` (upstream)."""
        return [
            r.source for r in self.edges_to(name, relation="supplies")
        ]

    def customers_of(self, name: str) -> list[str]:
        """Companies that `name` supplies to (downstream)."""
        return [
            r.target for r in self.edges_from(name, relation="supplies")
        ]

    def peers_of(self, name: str) -> list[str]:
        """Bidirectional peers of `name`."""
        # edges_from gives the outgoing half of the bidirectional pair.
        seen: set[str] = set()
        for r in self.edges_from(name, relation="peer"):
            seen.add(r.target)
        for r in self.edges_to(name, relation="peer"):
            seen.add(r.source)
        return sorted(seen)

    def infrastructure_of(self, name: str) -> list[str]:
        """Infrastructure providers to `name`."""
        return [
            r.source for r in self.edges_to(name, relation="infrastructure")
        ]

    def targets(self) -> list[str]:
        """Every node flagged as is_target (i.e. has its own value chain brief)."""
        return sorted(
            n for n, d in self._g.nodes(data=True) if d.get("is_target")
        )

    def relationships(self) -> Iterable[Relationship]:
        for src, tgt, data in self._g.edges(data=True):
            yield Relationship(
                source=src,
                target=tgt,
                relation=data.get("relation", ""),
                source_doc=data.get("source_doc"),
                notes=data.get("notes"),
            )

    # ---- persistence --------------------------------------------------

    def to_json(self) -> str:
        payload = {
            "nodes": [
                {
                    "name": n,
                    **{
                        k: v
                        for k, v in d.items()
                        if v is not None and not (k == "is_target" and v is False)
                    },
                }
                for n, d in self._g.nodes(data=True)
            ],
            "edges": [asdict(r) for r in self.relationships()],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ValueChainGraph":
        data = json.loads(text)
        g = cls()
        for node in data.get("nodes", []):
            g.add_company(
                CompanyNode(
                    name=node["name"],
                    ticker=node.get("ticker"),
                    industry=node.get("industry"),
                    is_target=node.get("is_target", False),
                )
            )
        for edge in data.get("edges", []):
            g.add_relationship(
                Relationship(
                    source=edge["source"],
                    target=edge["target"],
                    relation=edge["relation"],
                    source_doc=edge.get("source_doc"),
                    notes=edge.get("notes"),
                )
            )
        return g

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "ValueChainGraph":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
