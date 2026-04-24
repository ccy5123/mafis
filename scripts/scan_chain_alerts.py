"""Scan recent news against the value chain graph → alert on impact.

Usage:
    python scripts/scan_chain_alerts.py                 # scan all targets
    python scripts/scan_chain_alerts.py --hops 3        # widen reach
    python scripts/scan_chain_alerts.py --telegram      # push to bot
    python scripts/scan_chain_alerts.py --symbol NVDA   # one target only

What it does:
  1. Load the persisted value chain graph from data/value_chain.graph.json.
  2. For each target ticker (is_target=True), pull a geopolitics
     snapshot (Google News + GDELT) via the symbol's keyword profile.
  3. Union every news item into one list.
  4. Scan the graph: any news title mentioning a graph-node alias
     within N hops of a target becomes an alert.
  5. Print a markdown report; optionally push to Telegram.

Designed for cron: `*/60 * * * * python scripts/scan_chain_alerts.py`.
Cooldown/dedup across runs is NOT implemented here — this is a stateless
scanner. The caller decides what to do with the results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.alerts.chain_alerts import (  # noqa: E402
    NewsItemLike,
    compose_alert_markdown,
    scan_for_alerts,
)
from wise_investor.value_chain.graph import ValueChainGraph  # noqa: E402


console = Console()

GRAPH_PATH = REPO_ROOT / "data" / "value_chain.graph.json"


def _pull_news_for(symbol: str) -> list[NewsItemLike]:
    """Fetch Google News + GDELT items for one symbol.

    Converts the geopolitics-package dataclasses to the duck-typed
    NewsItemLike the alerts module expects.
    """
    try:
        from wise_investor.geopolitics.snapshot import get_geopolitics_snapshot
    except Exception as e:
        console.print(f"[red]Geopolitics import failed: {e}[/red]")
        return []

    items: list[NewsItemLike] = []
    try:
        snap = get_geopolitics_snapshot(symbol)
    except Exception as e:
        console.print(f"[yellow]Snapshot for {symbol} failed: {e}[/yellow]")
        return items

    for gn in snap.google_news:
        items.append(
            NewsItemLike(
                title=gn.title,
                source=gn.source or "Google News",
                published=gn.published or "",
                kind="google_news",
            )
        )
    for theme in snap.gdelt_themes:
        for art in theme.articles:
            items.append(
                NewsItemLike(
                    title=art.title,
                    source=art.domain or "GDELT",
                    published=art.iso_date[:10] if art.iso_date else "",
                    kind="gdelt",
                )
            )
    return items


def run(args: argparse.Namespace) -> int:
    if not GRAPH_PATH.exists():
        console.print(
            f"[red]No graph at {GRAPH_PATH}. Run scripts/build_value_chain_graph.py first.[/red]"
        )
        return 1

    graph = ValueChainGraph.load(GRAPH_PATH)
    console.print(
        f"Loaded graph: [cyan]{graph.num_nodes} nodes, {graph.num_edges} edges[/cyan]"
    )

    targets = [args.symbol.upper()] if args.symbol else graph.targets()
    if not targets:
        console.print("[yellow]No target tickers in the graph.[/yellow]")
        return 1

    console.print(f"Scanning for targets: [cyan]{', '.join(targets)}[/cyan]")

    # Aggregate news from each target's keyword profile. Dedupe by title.
    seen_titles: set[str] = set()
    news: list[NewsItemLike] = []
    for sym in targets:
        console.print(f"  pulling news for {sym}...")
        for item in _pull_news_for(sym):
            key = item.title.strip().lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            news.append(item)

    console.print(f"Collected [cyan]{len(news)}[/cyan] unique news items")

    alerts = scan_for_alerts(graph, news, max_hops=args.hops)
    console.print(f"Emitted [cyan]{len(alerts)}[/cyan] alert(s)")

    md = compose_alert_markdown(alerts)
    console.print()
    console.print(Markdown(md))

    if args.telegram and alerts:
        try:
            from wise_investor.notify.telegram import TelegramNotifier

            notifier = TelegramNotifier()
            if notifier.configured:
                notifier.send(md)
                console.print("[green]Pushed to Telegram[/green]")
            else:
                console.print(
                    "[yellow]Telegram not configured (set TELEGRAM_BOT_TOKEN + "
                    "TELEGRAM_CHAT_ID in .env).[/yellow]"
                )
        except Exception as e:
            console.print(f"[red]Telegram push failed: {e}[/red]")

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        console.print(f"[green]Saved markdown to {args.output}[/green]")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hops",
        type=int,
        default=2,
        help="Max graph hops from matched node to target (default 2)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Restrict scan to one target ticker (default: all is_target=True nodes)",
    )
    parser.add_argument(
        "--telegram",
        action="store_true",
        help="Push alerts to Telegram bot when alerts are found",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the alert markdown (e.g. reports/chain_alerts_YYYYMMDD.md)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
