"""Portfolio position management CLI.

Usage:
    python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1
    python scripts/portfolio_cli.py list
    python scripts/portfolio_cli.py delete NVDA
    python scripts/portfolio_cli.py weights           # queries Finnhub for live prices
    python scripts/portfolio_cli.py gap NVDA --low 3.0 --high 5.0

The SQLite file lives at the path in settings.sqlite_path (default
data/portfolio.sqlite). All operations are local; no cloud sync.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402
from wise_investor.portfolio.store import PortfolioStore  # noqa: E402


console = Console()


def _fetch_live_prices(symbols: list[str]) -> dict[str, float | None]:
    """Pull one quote per symbol from Finnhub. Missing/failing quotes map
    to None so the weight snapshot can report the gap.
    """
    if not symbols:
        return {}
    try:
        from wise_investor.data.finnhub import FinnhubClient
    except Exception as e:
        console.print(f"[red]Could not load Finnhub client: {e}[/red]")
        return {s: None for s in symbols}

    out: dict[str, float | None] = {}
    with FinnhubClient() as client:
        for s in symbols:
            try:
                q = client.quote(s)
                out[s.upper()] = q.price
            except Exception as e:
                console.print(f"[yellow]Quote for {s} failed: {e}[/yellow]")
                out[s.upper()] = None
    return out


def _cmd_add(args: argparse.Namespace) -> int:
    store = PortfolioStore()
    pos = store.upsert_position(
        symbol=args.symbol,
        shares=args.shares,
        cost_basis_usd=args.cost,
        tier=args.tier,
        notes=args.notes or "",
    )
    console.print(
        f"[green]Saved[/green] {pos.symbol}: {pos.shares:g} shares, "
        f"cost ${pos.cost_basis_usd:,.2f}, tier {pos.tier}, "
        f"first_bought={pos.first_bought}"
    )
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    store = PortfolioStore()
    if store.delete_position(args.symbol):
        console.print(f"[green]Deleted[/green] {args.symbol.upper()}")
        return 0
    console.print(f"[yellow]No position found for {args.symbol.upper()}[/yellow]")
    return 1


def _cmd_list(args: argparse.Namespace) -> int:
    store = PortfolioStore()
    positions = store.list_positions()
    if not positions:
        console.print("[dim]No positions recorded.[/dim]")
        return 0
    table = Table(title=f"Portfolio ({settings.sqlite_path})")
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Tier", justify="right")
    table.add_column("Shares", justify="right")
    table.add_column("Cost Basis", justify="right")
    table.add_column("Avg / share", justify="right")
    table.add_column("First bought", no_wrap=True)
    table.add_column("Notes")
    for p in positions:
        avg = p.avg_cost_per_share
        table.add_row(
            p.symbol,
            str(p.tier),
            f"{p.shares:g}",
            f"${p.cost_basis_usd:,.2f}",
            f"${avg:,.2f}" if avg is not None else "—",
            p.first_bought,
            p.notes,
        )
    console.print(table)
    return 0


def _cmd_weights(args: argparse.Namespace) -> int:
    store = PortfolioStore()
    positions = store.list_positions()
    if not positions:
        console.print("[dim]No positions recorded.[/dim]")
        return 0
    prices = _fetch_live_prices([p.symbol for p in positions])
    snaps = store.snapshot_weights(prices)

    total_mv = sum(s.market_value_usd for s in snaps if s.market_value_usd is not None)
    total_cost = sum(s.cost_basis_usd for s in snaps)
    console.rule(
        f"[bold]Portfolio weights — total MV "
        f"${total_mv:,.2f} (cost ${total_cost:,.2f})[/bold]"
    )

    table = Table()
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Shares", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Mkt Value", justify="right")
    table.add_column("Weight %", justify="right")
    table.add_column("Unreal P/L", justify="right")
    for s in snaps:
        weight_str = f"{s.weight_pct:.2f}%" if s.weight_pct is not None else "—"
        mv_str = f"${s.market_value_usd:,.2f}" if s.market_value_usd is not None else "—"
        pnl_str = (
            f"${s.unrealized_pnl_usd:,.2f}" if s.unrealized_pnl_usd is not None else "—"
        )
        price_str = f"${s.price:,.2f}" if s.price is not None else "—"
        table.add_row(s.symbol, f"{s.shares:g}", price_str, mv_str, weight_str, pnl_str)
    console.print(table)
    return 0


def _cmd_gap(args: argparse.Namespace) -> int:
    store = PortfolioStore()
    positions = store.list_positions()
    prices = _fetch_live_prices([p.symbol for p in positions] or [args.symbol])
    msg = store.sizing_gap(
        symbol=args.symbol,
        suggested_low_pct=args.low,
        suggested_high_pct=args.high,
        prices=prices,
    )
    console.print(msg)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Add or update a position")
    p_add.add_argument("symbol")
    p_add.add_argument("--shares", type=float, required=True)
    p_add.add_argument("--cost", type=float, required=True, help="Total cost basis in USD")
    p_add.add_argument("--tier", type=int, choices=[1, 2, 3], required=True)
    p_add.add_argument("--notes", default="")
    p_add.set_defaults(func=_cmd_add)

    p_del = sub.add_parser("delete", help="Remove a position")
    p_del.add_argument("symbol")
    p_del.set_defaults(func=_cmd_delete)

    p_list = sub.add_parser("list", help="List all positions")
    p_list.set_defaults(func=_cmd_list)

    p_weights = sub.add_parser("weights", help="Current weights from live Finnhub quotes")
    p_weights.set_defaults(func=_cmd_weights)

    p_gap = sub.add_parser("gap", help="Compare current weight to a suggested sizing band")
    p_gap.add_argument("symbol")
    p_gap.add_argument("--low", type=float, required=True, help="Suggested low pct, e.g. 3.0")
    p_gap.add_argument("--high", type=float, required=True, help="Suggested high pct, e.g. 5.0")
    p_gap.set_defaults(func=_cmd_gap)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
