"""Ad-hoc semantic search over indexed 10-K sections.

Usage:
    python scripts/search_10k.py "supply chain concentration risk"
    python scripts/search_10k.py "data center capex" --symbol NVDA
    python scripts/search_10k.py "interest rate hedging" --section mdna --k 3

Requires the collection to be populated first via scripts/index_10k.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.rag.index import search, stats  # noqa: E402


console = Console()


SECTION_CHOICES = ["business", "risk_factors", "mdna", "quant_market_risk"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Free-text query (e.g. 'supply chain risk')")
    parser.add_argument(
        "--symbol", help="Restrict results to one ticker (e.g. NVDA)", default=None
    )
    parser.add_argument(
        "--section",
        choices=SECTION_CHOICES,
        default=None,
        help="Restrict results to one section key.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results (default 5)")
    args = parser.parse_args()

    summary = stats()
    total = summary.get("total_chunks", 0)
    if total == 0:
        console.print(
            "[red]Index is empty. Run `python scripts/index_10k.py <TICKER>` first.[/red]"
        )
        return 1

    console.rule(f"[bold]RAG search: {args.query!r}[/bold]")
    filters = []
    if args.symbol:
        filters.append(f"symbol={args.symbol.upper()}")
    if args.section:
        filters.append(f"section={args.section}")
    filter_desc = ", ".join(filters) if filters else "no filters"
    console.print(f"[dim]Index: {total} chunks   Filters: {filter_desc}   k={args.k}[/dim]")

    hits = search(
        query=args.query,
        symbol=args.symbol,
        section=args.section,
        k=args.k,
    )
    if not hits:
        console.print("[yellow]No hits.[/yellow]")
        return 0

    for i, hit in enumerate(hits, start=1):
        header = (
            f"[bold]{i}.[/bold] {hit.symbol}  "
            f"{hit.section}  "
            f"filed={hit.filing_date}  "
            f"chunk={hit.chunk_id}  "
            f"distance={hit.distance:.3f}"
        )
        excerpt = hit.text.strip()
        if len(excerpt) > 800:
            excerpt = excerpt[:800].rstrip() + " ..."
        console.print(Panel(excerpt, title=header, border_style="cyan"))

    return 0


if __name__ == "__main__":
    sys.exit(main())
