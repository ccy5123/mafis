"""Stage 6 — portfolio construction CLI.

Runs the full Stage 6 pipeline:
  1. Pulls historical price returns for the survivor pool (yfinance).
  2. Computes HRP weights.
  3. Applies Stage 5 cluster collision adjustment if a graph is supplied.
  4. Enforces 1% / 30% bounds.
  5. Compares to existing positions and emits a trade list.

Usage:
    python scripts/run_portfolio_construction.py NVDA AMD INTC KO PEP
    python scripts/run_portfolio_construction.py \\
        --tickers-file survivors.txt \\
        --graph data/value_chain.graph.json \\
        --positions positions.yaml

Existing positions YAML format:
    total_capital_usd: 100000.0  # optional; defaults to sum(values)
    positions:
      NVDA: 20000.0
      AAPL: 30000.0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.portfolio.construction import (  # noqa: E402
    DEFAULT_MAX_WEIGHT,
    DEFAULT_MIN_WEIGHT,
    PortfolioConstructionResult,
    construct_portfolio,
)
from wise_investor.screening.stage5_positioning import (  # noqa: E402
    position_survivors,
)
from wise_investor.value_chain.graph import ValueChainGraph  # noqa: E402

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Survivor tickers (positional). Mutually exclusive with --tickers-file.",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        help="File with one survivor ticker per line.",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        help=(
            "Path to value chain graph JSON. When supplied, Stage 5 "
            "cluster collision adjustment is applied to the HRP output."
        ),
    )
    parser.add_argument(
        "--positions",
        type=Path,
        help="YAML file with existing positions for trade computation.",
    )
    parser.add_argument(
        "--total-capital",
        type=float,
        help="Override for total capital. Defaults to sum of existing positions.",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=DEFAULT_MIN_WEIGHT,
        help=f"Minimum single-position weight (default {DEFAULT_MIN_WEIGHT:.2f}).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=DEFAULT_MAX_WEIGHT,
        help=f"Maximum single-position weight (default {DEFAULT_MAX_WEIGHT:.2f}).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=504,
        help="Trading-day window for the return matrix (default ~2 years).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of formatted tables.",
    )
    args = parser.parse_args()

    survivors = _resolve_survivors(args)
    if not survivors:
        console.print("[red]No survivor tickers provided.[/red]")
        return 2

    positioning_report = None
    if args.graph:
        if not args.graph.exists():
            console.print(f"[red]Graph not found: {args.graph}[/red]")
            return 1
        graph = ValueChainGraph.load(args.graph)
        positioning_report = position_survivors(survivors, graph)

    existing_positions: dict[str, float] | None = None
    total_capital = args.total_capital
    if args.positions:
        if not args.positions.exists():
            console.print(f"[red]Positions file not found: {args.positions}[/red]")
            return 1
        existing_positions, file_capital = _load_positions(args.positions)
        if total_capital is None:
            total_capital = file_capital

    try:
        graph_display = (
            args.graph.relative_to(REPO_ROOT) if args.graph else None
        )
    except ValueError:
        graph_display = args.graph

    console.print(
        Panel.fit(
            (
                f"[bold]Stage 6 portfolio construction[/bold]\n"
                f"Survivors: {len(survivors)}\n"
                f"Bounds: [{args.min_weight:.2%}, {args.max_weight:.2%}]\n"
                f"Lookback: {args.lookback_days} trading days\n"
                f"Graph: {graph_display or '[dim]none — no cluster adjustment[/dim]'}\n"
                f"Existing positions: "
                f"{len(existing_positions) if existing_positions else 0}\n"
                f"Total capital: "
                f"{f'${total_capital:,.0f}' if total_capital else '[dim]—[/dim]'}"
            ),
            border_style="cyan",
        )
    )

    result = construct_portfolio(
        survivors,
        positioning_report=positioning_report,
        existing_positions=existing_positions,
        total_capital_usd=total_capital,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        lookback_days=args.lookback_days,
    )

    if args.json:
        print(json.dumps(_to_json(result), indent=2, default=str))
        return 0

    _print_report(result)
    return 0


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def _resolve_survivors(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        return [t.upper() for t in args.tickers]
    if args.tickers_file and args.tickers_file.exists():
        text = args.tickers_file.read_text(encoding="utf-8")
        return [
            line.strip().upper()
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
    return []


def _load_positions(path: Path) -> tuple[dict[str, float], float | None]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    positions = {
        str(k).upper(): float(v)
        for k, v in (raw.get("positions") or {}).items()
    }
    capital = raw.get("total_capital_usd")
    return positions, float(capital) if capital is not None else None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_report(result: PortfolioConstructionResult) -> None:
    if not result.target_weights:
        console.print(
            "\n[red]No portfolio produced.[/red] "
            f"({result.n_excluded_no_data} tickers excluded for missing data)"
        )
        if result.excluded_tickers:
            console.print(
                f"[dim]Excluded: {', '.join(result.excluded_tickers)}[/dim]"
            )
        return

    # Target weights table
    console.print()
    console.rule("[bold]Target weights[/bold]")
    wt = Table(show_lines=False)
    wt.add_column("Symbol", style="bold")
    wt.add_column("HRP raw", justify="right")
    wt.add_column("After cluster", justify="right")
    wt.add_column("After bounds", justify="right")
    wt.add_column("Cluster trim", justify="right")

    raw = result.raw_hrp_weights
    target = result.target_weights
    trims = result.cluster_adjustments

    # Rebuild the post-cluster value for display: target ÷ post-bounds
    # rescaling isn't recoverable here, so we just show raw and final.
    for sym in sorted(target, key=lambda s: -target[s]):
        wt.add_row(
            sym,
            f"{raw.get(sym, 0):.2%}",
            "—",
            f"{target[sym]:.2%}",
            f"{trims[sym]:.2f}×" if sym in trims else "[dim]—[/dim]",
        )
    console.print(wt)

    if result.excluded_tickers:
        console.print()
        console.print(
            f"[yellow]Excluded for missing return data:[/yellow] "
            f"{', '.join(result.excluded_tickers)}"
        )

    # Trades
    if result.trades:
        console.print()
        console.rule("[bold]Rebalance trades[/bold]")
        tt = Table(show_lines=False)
        tt.add_column("Symbol", style="bold")
        tt.add_column("Target wt", justify="right")
        tt.add_column("Target $", justify="right")
        tt.add_column("Current $", justify="right")
        tt.add_column("Trade $", justify="right")

        for trade in sorted(
            result.trades, key=lambda t: -abs(t.trade_value_usd)
        ):
            sign_color = (
                "green"
                if trade.trade_value_usd > 0
                else "red"
                if trade.trade_value_usd < 0
                else "white"
            )
            tt.add_row(
                trade.symbol,
                f"{trade.target_weight:.2%}",
                f"${trade.target_value_usd:,.0f}",
                f"${trade.current_value_usd:,.0f}",
                f"[{sign_color}]${trade.trade_value_usd:+,.0f}[/{sign_color}]",
            )
        console.print(tt)

        # Summary
        n_buys = sum(1 for t in result.trades if t.trade_value_usd > 1e-3)
        n_sells = sum(1 for t in result.trades if t.trade_value_usd < -1e-3)
        n_holds = len(result.trades) - n_buys - n_sells
        console.print(
            f"\n[bold]Summary:[/bold] {n_buys} buys, {n_sells} sells, "
            f"{n_holds} holds"
        )


def _to_json(result: PortfolioConstructionResult) -> dict:
    return {
        "target_weights": result.target_weights,
        "raw_hrp_weights": result.raw_hrp_weights,
        "cluster_adjustments": result.cluster_adjustments,
        "bounds": {"min": result.bounds_min, "max": result.bounds_max},
        "total_capital_usd": result.total_capital_usd,
        "n_excluded_no_data": result.n_excluded_no_data,
        "excluded_tickers": list(result.excluded_tickers),
        "trades": [asdict(t) for t in result.trades],
    }


if __name__ == "__main__":
    sys.exit(main())
