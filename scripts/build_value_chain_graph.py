"""Build the value chain graph from docs/value_chains/*.md and save JSON.

Usage:
    python scripts/build_value_chain_graph.py

Writes data/value_chain.graph.json containing every parsed node and
edge, with source_doc attribution per edge so the graph is always
auditable back to a hand-curated brief.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.value_chain.parser import build_graph_from_briefs  # noqa: E402


console = Console()


def main() -> int:
    briefs_dir = REPO_ROOT / "docs" / "value_chains"
    out_path = REPO_ROOT / "data" / "value_chain.graph.json"

    console.rule("[bold]Build value chain graph[/bold]")
    console.print(f"Briefs: [cyan]{briefs_dir}[/cyan]")

    graph = build_graph_from_briefs(briefs_dir)
    graph.save(out_path)

    console.print(
        f"[green]Saved {graph.num_nodes} nodes / {graph.num_edges} edges "
        f"to[/green] [cyan]{out_path}[/cyan]"
    )

    console.print("\n[bold]Targets[/bold] (have their own brief):")
    for t in graph.targets():
        peers = graph.peers_of(t)
        suppliers = graph.suppliers_of(t)
        customers = graph.customers_of(t)
        infra = graph.infrastructure_of(t)
        console.print(
            f"  {t}: suppliers={len(suppliers)} peers={len(peers)} "
            f"customers={len(customers)} infrastructure={len(infra)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
