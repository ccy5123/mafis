"""Query the saved value chain graph.

Usage:
    python scripts/value_chain_query.py NVDA
    python scripts/value_chain_query.py NVDA --show peers
    python scripts/value_chain_query.py NVDA --show suppliers
    python scripts/value_chain_query.py NVDA --show customers
    python scripts/value_chain_query.py NVDA --show infrastructure
    python scripts/value_chain_query.py --all-targets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.value_chain.graph import ValueChainGraph  # noqa: E402


console = Console()
GRAPH_PATH = REPO_ROOT / "data" / "value_chain.graph.json"


def _load() -> ValueChainGraph:
    if not GRAPH_PATH.exists():
        console.print(
            f"[red]{GRAPH_PATH} not found. "
            f"Run scripts/build_value_chain_graph.py first.[/red]"
        )
        sys.exit(1)
    return ValueChainGraph.load(GRAPH_PATH)


def _print_edges(title: str, names: list[str]) -> None:
    if not names:
        console.print(f"[dim]{title}: none[/dim]")
        return
    t = Table(title=title, show_header=False, show_edge=True)
    t.add_column("Name")
    for n in names:
        t.add_row(n)
    console.print(t)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", help="Target ticker (e.g. NVDA)")
    parser.add_argument(
        "--show",
        choices=["all", "peers", "suppliers", "customers", "infrastructure"],
        default="all",
    )
    parser.add_argument("--all-targets", action="store_true")
    args = parser.parse_args()

    g = _load()
    console.print(
        f"[dim]Graph: {g.num_nodes} nodes / {g.num_edges} edges[/dim]\n"
    )

    if args.all_targets or not args.symbol:
        for t in g.targets():
            console.print(f"[bold cyan]{t}[/bold cyan]")
            console.print(f"  suppliers ({len(g.suppliers_of(t))}): {', '.join(g.suppliers_of(t)[:6])}")
            console.print(f"  peers     ({len(g.peers_of(t))}): {', '.join(g.peers_of(t)[:6])}")
            console.print(f"  customers ({len(g.customers_of(t))}): {', '.join(g.customers_of(t)[:6])}")
            console.print(f"  infra     ({len(g.infrastructure_of(t))}): {', '.join(g.infrastructure_of(t)[:6])}")
            console.print()
        return 0

    symbol = args.symbol.upper()
    if not g.has_company(symbol):
        console.print(f"[red]{symbol} not in graph[/red]")
        return 1

    company = g.get_company(symbol)
    if company is not None:
        console.print(
            f"[bold cyan]{company.name}[/bold cyan]  ticker={company.ticker}  "
            f"industry={company.industry}  target={company.is_target}"
        )
        console.print()

    if args.show in ("all", "suppliers"):
        _print_edges(f"Suppliers of {symbol}", g.suppliers_of(symbol))
    if args.show in ("all", "peers"):
        _print_edges(f"Peers of {symbol}", g.peers_of(symbol))
    if args.show in ("all", "customers"):
        _print_edges(f"Customers of {symbol}", g.customers_of(symbol))
    if args.show in ("all", "infrastructure"):
        _print_edges(f"Infrastructure for {symbol}", g.infrastructure_of(symbol))

    return 0


if __name__ == "__main__":
    sys.exit(main())
