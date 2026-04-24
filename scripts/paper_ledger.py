"""Paper trading ledger — record Steward verdicts, track returns over time.

Usage:
    python scripts/paper_ledger.py record reports/NVDA_20260424_1715.crew.md
    python scripts/paper_ledger.py record reports/NVDA_20260424_1715.crew.md \\
        --verdict-date 2026-04-24                # override the issue date
    python scripts/paper_ledger.py record reports/NVDA_20260424_1715.crew.md \\
        --no-live-quote --price 512.00           # skip Finnhub, supply entry price
    python scripts/paper_ledger.py list                    # all trades
    python scripts/paper_ledger.py list --symbol NVDA
    python scripts/paper_ledger.py returns                 # mark-to-market
    python scripts/paper_ledger.py summary                 # aggregate metrics
    python scripts/paper_ledger.py delete 3                # remove row #3

Storage is the same SQLite file as the positions ledger
(data/portfolio.sqlite); tables are separate (`paper_trades`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.paper_trading.ledger import PaperTradeLedger  # noqa: E402
from wise_investor.paper_trading.report_parser import parse_crew_report  # noqa: E402


console = Console()


def _live_price(symbol: str) -> float | None:
    try:
        from wise_investor.data.finnhub import FinnhubClient

        with FinnhubClient() as c:
            return c.quote(symbol).price
    except Exception as e:
        console.print(f"[yellow]Live quote for {symbol} failed: {e}[/yellow]")
        return None


def _cmd_record(args: argparse.Namespace) -> int:
    report_path = Path(args.report)
    if not report_path.exists():
        console.print(f"[red]Report not found: {report_path}[/red]")
        return 1

    text = report_path.read_text(encoding="utf-8")
    summary = parse_crew_report(text, symbol_hint=args.symbol)
    if not summary.symbol:
        console.print(
            "[red]Could not determine symbol from the report title. "
            "Pass --symbol explicitly.[/red]"
        )
        return 1
    if summary.verdict is None:
        console.print("[red]Report had no parseable Verdict heading.[/red]")
        return 1

    # Entry price.
    if args.price is not None:
        price = float(args.price)
    elif args.no_live_quote:
        price = None
    else:
        price = _live_price(summary.symbol)

    ledger = PaperTradeLedger()
    trade = ledger.record_trade(
        symbol=summary.symbol,
        verdict=summary.verdict,
        original_verdict=summary.original_verdict or summary.verdict,
        verdict_date=args.verdict_date,
        conviction=summary.conviction,
        original_conviction=summary.original_conviction,
        audit_downgraded=summary.audit_downgraded,
        price_at_verdict=price,
        report_path=str(report_path),
    )
    audit_note = (
        f" (audit: {summary.original_verdict} C{summary.original_conviction} "
        f"→ {summary.verdict} C{summary.conviction})"
        if summary.audit_downgraded
        else ""
    )
    console.print(
        f"[green]Recorded trade #{trade.id}[/green] "
        f"{trade.symbol} {trade.verdict} C{trade.conviction or '?'} "
        f"@ ${trade.price_at_verdict or '?':,.2f} on {trade.verdict_date}"
        f"{audit_note}"
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    ledger = PaperTradeLedger()
    trades = ledger.list_trades(symbol=args.symbol, verdict=args.verdict)
    if not trades:
        console.print("[dim]No trades recorded.[/dim]")
        return 0
    table = Table(title="Paper trade ledger")
    table.add_column("#", justify="right")
    table.add_column("Symbol", style="cyan")
    table.add_column("Verdict")
    table.add_column("C", justify="right")
    table.add_column("Entry $", justify="right")
    table.add_column("Issued")
    table.add_column("Audit?", justify="center")
    table.add_column("Report", style="dim")
    for t in trades:
        audit_flag = "↓" if t.audit_downgraded else ""
        price = f"${t.price_at_verdict:,.2f}" if t.price_at_verdict else "—"
        table.add_row(
            str(t.id),
            t.symbol,
            t.verdict,
            str(t.conviction) if t.conviction else "—",
            price,
            t.verdict_date,
            audit_flag,
            Path(t.report_path).name if t.report_path else "",
        )
    console.print(table)
    return 0


def _cmd_delete(args: argparse.Namespace) -> int:
    ledger = PaperTradeLedger()
    if ledger.delete_trade(args.trade_id):
        console.print(f"[green]Deleted trade #{args.trade_id}[/green]")
        return 0
    console.print(f"[yellow]No trade with id={args.trade_id}[/yellow]")
    return 1


def _cmd_returns(args: argparse.Namespace) -> int:
    ledger = PaperTradeLedger()
    trades = ledger.list_trades()
    if not trades:
        console.print("[dim]No trades recorded.[/dim]")
        return 0

    # Live prices for every unique symbol.
    symbols = sorted({t.symbol for t in trades})
    prices: dict[str, float | None] = {}
    for sym in symbols:
        if args.no_live_quote:
            prices[sym] = None
        else:
            prices[sym] = _live_price(sym)

    returns = ledger.current_returns(prices)
    table = Table(title="Mark-to-market returns")
    table.add_column("#", justify="right")
    table.add_column("Symbol", style="cyan")
    table.add_column("Verdict")
    table.add_column("Entry $", justify="right")
    table.add_column("Current $", justify="right")
    table.add_column("Return %", justify="right")
    table.add_column("Days held", justify="right")
    import datetime as dt

    today = dt.date.today()
    for r in returns:
        price_in = (
            f"${r.trade.price_at_verdict:,.2f}"
            if r.trade.price_at_verdict
            else "—"
        )
        price_now = f"${r.current_price:,.2f}" if r.current_price else "—"
        pct = f"{r.return_pct:+.2f}%" if r.return_pct is not None else "—"
        try:
            days = (today - dt.date.fromisoformat(r.trade.verdict_date)).days
        except Exception:
            days = "?"
        table.add_row(
            str(r.trade.id),
            r.trade.symbol,
            r.trade.verdict,
            price_in,
            price_now,
            pct,
            str(days),
        )
    console.print(table)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    ledger = PaperTradeLedger()
    trades = ledger.list_trades()
    if not trades:
        console.print("[dim]No trades recorded.[/dim]")
        return 0

    symbols = sorted({t.symbol for t in trades})
    prices: dict[str, float | None] = {}
    for sym in symbols:
        prices[sym] = None if args.no_live_quote else _live_price(sym)

    summary = ledger.performance_summary(prices)
    console.rule(f"[bold]Performance summary — {summary.n_trades} trades[/bold]")

    if summary.by_verdict:
        console.print()
        console.print("[bold]By verdict[/bold]")
        for v, stats in summary.by_verdict.items():
            console.print(
                f"  [cyan]{v}[/cyan]: n={int(stats['n'])}  "
                f"avg={stats['avg_return_pct']:+.2f}%  "
                f"win rate={stats['win_rate'] * 100:.1f}%"
            )

    if summary.by_conviction:
        console.print()
        console.print("[bold]By conviction[/bold]")
        for c, stats in sorted(summary.by_conviction.items()):
            console.print(
                f"  C{c}: n={int(stats['n'])}  "
                f"avg={stats['avg_return_pct']:+.2f}%"
            )

    if summary.audit_effect:
        console.print()
        console.print("[bold]Audit effect (original BUY verdicts)[/bold]")
        if "clean_avg_return_pct" in summary.audit_effect:
            console.print(
                f"  BUYs that cleared audit: "
                f"{summary.audit_effect['clean_avg_return_pct']:+.2f}%"
            )
        if "downgraded_avg_return_pct" in summary.audit_effect:
            console.print(
                f"  BUYs downgraded by audit: "
                f"{summary.audit_effect['downgraded_avg_return_pct']:+.2f}%"
            )

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rec = sub.add_parser("record", help="Record a verdict from a crew report file")
    p_rec.add_argument("report", help="Path to a <SYMBOL>_YYYYMMDD_HHMM.crew.md file")
    p_rec.add_argument("--symbol", default=None, help="Override ticker detection")
    p_rec.add_argument(
        "--verdict-date", default=None,
        help="Override the verdict date (ISO YYYY-MM-DD); defaults to today",
    )
    p_rec.add_argument(
        "--price", type=float, default=None,
        help="Supply entry price explicitly (skips Finnhub live quote)",
    )
    p_rec.add_argument(
        "--no-live-quote", action="store_true",
        help="Skip the live-quote fetch; entry price left NULL unless --price given",
    )
    p_rec.set_defaults(func=_cmd_record)

    p_list = sub.add_parser("list", help="List recorded trades")
    p_list.add_argument("--symbol", default=None)
    p_list.add_argument("--verdict", default=None, choices=["BUY", "HOLD", "PASS"])
    p_list.set_defaults(func=_cmd_list)

    p_del = sub.add_parser("delete", help="Delete a trade row")
    p_del.add_argument("trade_id", type=int)
    p_del.set_defaults(func=_cmd_delete)

    p_ret = sub.add_parser("returns", help="Mark-to-market every open trade")
    p_ret.add_argument("--no-live-quote", action="store_true")
    p_ret.set_defaults(func=_cmd_returns)

    p_sum = sub.add_parser("summary", help="Aggregate performance metrics")
    p_sum.add_argument("--no-live-quote", action="store_true")
    p_sum.set_defaults(func=_cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
