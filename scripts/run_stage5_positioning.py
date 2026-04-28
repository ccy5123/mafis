"""Stage 5 — value chain positioning of rubric-passing survivors.

Reads:
  - data/value_chain.graph.json  (from `scripts/build_value_chain_graph.py`)
  - A list of survivor tickers (positional args, --tickers-file, or
    --from-screening JSON output)

Emits:
  - Per-survivor position table: peers / suppliers / customers / infra
  - Cluster representation report: which graph clusters are
    over- or under-represented in the survivor pool

Usage:
    python scripts/run_stage5_positioning.py NVDA AMD INTC
    python scripts/run_stage5_positioning.py --tickers-file survivors.txt
    python scripts/run_stage5_positioning.py --from-screening screening.json

Constitution alignment: this is the §6 Stage 5 step. Survivors that
don't map to any graph node are reported separately so the user can
extend `docs/value_chains/` to fill the gaps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.screening.stage5_positioning import (  # noqa: E402
    Stage5PositioningReport,
    position_survivors,
    representation_summary,
)
from wise_investor.value_chain.graph import ValueChainGraph  # noqa: E402

console = Console()


DEFAULT_GRAPH_PATH = REPO_ROOT / "data" / "value_chain.graph.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Survivor tickers (positional). Mutually exclusive with --tickers-file / --from-screening.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Path to value chain graph JSON (default: data/value_chain.graph.json)",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        help="File with one survivor ticker per line.",
    )
    parser.add_argument(
        "--from-screening",
        type=Path,
        help=(
            "Path to a screening JSON output (run_screening.py --json). "
            "Will pull tickers whose Stage 2 hierarchy_decision is "
            "ADVANCE_TO_STAGE_3 — NOT actual Stage 4 survivors. Useful "
            "as a placeholder pool until the agent runner lands."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON summary instead of formatted tables.",
    )
    args = parser.parse_args()

    if not args.graph.exists():
        console.print(f"[red]Graph not found: {args.graph}[/red]")
        console.print(
            "[yellow]Run `python scripts/build_value_chain_graph.py` first.[/yellow]"
        )
        return 1

    survivors = _resolve_survivors(args)
    if not survivors:
        console.print("[red]No survivor tickers provided.[/red]")
        return 2

    graph = ValueChainGraph.load(args.graph)
    report = position_survivors(survivors, graph)

    if args.json:
        print(json.dumps(representation_summary(report), indent=2, default=str))
        return 0

    _print_report(report, graph_path=args.graph)
    return 0


def _resolve_survivors(args: argparse.Namespace) -> list[str]:
    """Pick survivor list from one of the three input modes.

    Order of precedence: positional args > --tickers-file > --from-screening.
    Mutually-exclusive enforcement is friendly (we don't error if more
    than one is given; we just pick the highest-precedence source).
    """
    if args.tickers:
        return [t.upper() for t in args.tickers]
    if args.tickers_file and args.tickers_file.exists():
        text = args.tickers_file.read_text(encoding="utf-8")
        return [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if args.from_screening and args.from_screening.exists():
        payload = json.loads(args.from_screening.read_text(encoding="utf-8"))
        out: list[str] = []
        for row in payload:
            pf = row.get("prefilter")
            if pf and pf.get("hierarchy_decision") == "ADVANCE_TO_STAGE_3":
                out.append(str(row["symbol"]).upper())
        return out
    return []


def _print_report(
    report: Stage5PositioningReport, *, graph_path: Path
) -> None:
    try:
        graph_display = graph_path.relative_to(REPO_ROOT)
    except ValueError:
        graph_display = graph_path

    console.print(
        Panel.fit(
            (
                f"[bold]Stage 5 positioning[/bold]\n"
                f"Graph: {graph_display}\n"
                f"Survivors: {report.n_total_survivors} "
                f"({report.n_mapped_survivors} mapped, "
                f"{report.n_unmapped_survivors} unmapped)\n"
                f"Graph nodes: {report.total_graph_nodes}"
            ),
            border_style="cyan",
        )
    )

    # Per-survivor positions.
    if report.survivor_positions:
        console.print()
        console.rule("[bold]Per-survivor positions[/bold]")
        table = Table(show_lines=False)
        table.add_column("Ticker", style="bold")
        table.add_column("Node")
        table.add_column("Industry")
        table.add_column("Cluster")
        table.add_column("Peers")
        table.add_column("Suppliers")
        table.add_column("Customers")

        for p in report.survivor_positions:
            cluster_str = (
                str(p.cluster_id)
                if p.cluster_id is not None
                else "[dim]—[/dim]"
            )
            node_str = p.node_name or "[red](unmapped)[/red]"
            table.add_row(
                p.ticker,
                node_str,
                p.industry or "[dim]—[/dim]",
                cluster_str,
                _short_list(p.peers),
                _short_list(p.suppliers),
                _short_list(p.customers),
            )
        console.print(table)

    # Cluster representation.
    if report.clusters:
        console.print()
        console.rule("[bold]Cluster representation[/bold]")
        ctable = Table(show_lines=False)
        ctable.add_column("ID", style="bold")
        ctable.add_column("Members")
        ctable.add_column("Survivors")
        ctable.add_column("Expected", justify="right")
        ctable.add_column("Ratio", justify="right")
        ctable.add_column("Flag")

        for c in report.clusters:
            ratio = (
                f"{c.over_representation:.2f}×"
                if c.over_representation != float("inf")
                else "∞"
            )
            color = {
                "over": "yellow",
                "under": "blue",
                "neutral": "white",
            }.get(c.flag, "white")
            ctable.add_row(
                str(c.cluster_id),
                str(c.n_members),
                str(c.n_survivors),
                f"{c.expected_survivors:.2f}",
                ratio,
                f"[{color}]{c.flag}[/{color}]",
            )
        console.print(ctable)

    if report.unmapped_tickers:
        console.print()
        console.print(
            f"[yellow]Unmapped survivors:[/yellow] "
            f"{', '.join(report.unmapped_tickers)}"
        )
        console.print(
            "[dim]Add `docs/value_chains/<TICKER>.draft.md` briefs and "
            "rebuild the graph to position these.[/dim]"
        )


def _short_list(items: tuple[str, ...], max_n: int = 3) -> str:
    if not items:
        return "[dim]—[/dim]"
    if len(items) <= max_n:
        return ", ".join(items)
    return ", ".join(items[:max_n]) + f" +{len(items) - max_n}"


if __name__ == "__main__":
    sys.exit(main())
