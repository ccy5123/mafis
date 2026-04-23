"""Standalone geopolitical macro inspector.

Usage:
    python scripts/probe_geopolitics.py NVDA
    python scripts/probe_geopolitics.py NVDA --timespan 3days
    python scripts/probe_geopolitics.py --query "Taiwan semiconductor"
    python scripts/probe_geopolitics.py --themes ECON_TRADE_SANCTIONS,TRADE_WAR

Prints one Rich table per GDELT theme + the Google News headline feed.
Uses only free, key-less public APIs (GDELT DOC 2.0 + Google News RSS).

This script is intentionally decoupled from the crew pre-gather pipeline
so users can inspect the raw data quality before we wire it into the
Economist context block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.geopolitics.gdelt import (  # noqa: E402
    GEOPOLITICAL_THEMES,
    GdeltClient,
)
from wise_investor.geopolitics.google_news import fetch_google_news  # noqa: E402
from wise_investor.geopolitics.snapshot import (  # noqa: E402
    DEFAULT_THEMES,
    build_google_news_query,
    get_geopolitics_snapshot,
)


console = Console()


def _print_google_news(query: str, max_items: int) -> None:
    console.rule(f"[bold]Google News — {query}[/bold]")
    try:
        items = fetch_google_news(query, max_items=max_items)
    except Exception as e:
        console.print(f"[red]Google News failed: {e}[/red]")
        return
    if not items:
        console.print("[yellow]No items.[/yellow]")
        return
    table = Table(show_lines=False)
    table.add_column("Date", style="dim", no_wrap=True)
    table.add_column("Source", no_wrap=True, style="cyan")
    table.add_column("Headline")
    for item in items:
        table.add_row(item.published or "?", item.source or "?", item.title)
    console.print(table)


def _print_gdelt_theme(theme: str, timespan: str, max_records: int) -> None:
    label = GEOPOLITICAL_THEMES.get(theme, theme)
    console.rule(f"[bold]GDELT {theme} — {label}[/bold]")
    with GdeltClient() as client:
        result = client.search_theme(
            theme=theme, timespan=timespan, max_records=max_records
        )
    if result.error:
        console.print(f"[red]Error: {result.error}[/red]")
        return
    if not result.articles:
        console.print(f"[yellow]No articles in last {timespan}.[/yellow]")
        return
    table = Table(show_lines=False)
    table.add_column("Seen (UTC)", style="dim", no_wrap=True)
    table.add_column("Country", no_wrap=True)
    table.add_column("Domain", style="cyan", no_wrap=True)
    table.add_column("Title")
    for art in result.articles:
        table.add_row(
            art.iso_date[:10] if art.iso_date else "?",
            art.source_country or "?",
            art.domain or "?",
            art.title,
        )
    console.print(table)


def _print_raw_query(query: str, timespan: str, max_records: int) -> None:
    console.rule(f"[bold]GDELT raw query — {query}[/bold]")
    with GdeltClient() as client:
        try:
            articles = client.search_articles(
                query=query, timespan=timespan, max_records=max_records
            )
        except Exception as e:
            console.print(f"[red]GDELT failed: {e}[/red]")
            return
    if not articles:
        console.print(f"[yellow]No articles in last {timespan}.[/yellow]")
        return
    table = Table(show_lines=False)
    table.add_column("Seen (UTC)", style="dim", no_wrap=True)
    table.add_column("Country", no_wrap=True)
    table.add_column("Domain", style="cyan", no_wrap=True)
    table.add_column("Title")
    for art in articles:
        table.add_row(
            art.iso_date[:10] if art.iso_date else "?",
            art.source_country or "?",
            art.domain or "?",
            art.title,
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Ticker to profile (e.g. NVDA). Omit when using --query or --themes.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Raw GDELT query string (bypasses symbol registry).",
    )
    parser.add_argument(
        "--themes",
        default=None,
        help="Comma-separated GDELT theme keys (e.g. ECON_TRADE_SANCTIONS,TRADE_WAR)",
    )
    parser.add_argument(
        "--timespan", default="7days", help="GDELT timespan (e.g. 24h, 3days, 7days)"
    )
    parser.add_argument(
        "--gdelt-max", type=int, default=10, help="Max articles per GDELT theme/query"
    )
    parser.add_argument(
        "--news-max", type=int, default=10, help="Max Google News headlines"
    )
    args = parser.parse_args()

    # Mode 1: raw query
    if args.query:
        _print_raw_query(args.query, args.timespan, args.gdelt_max)
        return 0

    # Mode 2: theme list without symbol
    if args.themes and not args.symbol:
        themes = [t.strip() for t in args.themes.split(",") if t.strip()]
        for theme in themes:
            _print_gdelt_theme(theme, args.timespan, args.gdelt_max)
        return 0

    # Mode 3: full per-symbol snapshot (default)
    if not args.symbol:
        parser.error("Provide a symbol, --query, or --themes")

    symbol = args.symbol.upper()
    themes = (
        tuple(t.strip() for t in args.themes.split(",") if t.strip())
        if args.themes
        else DEFAULT_THEMES
    )

    console.rule(f"[bold]Geopolitical snapshot — {symbol}[/bold]")
    snapshot = get_geopolitics_snapshot(
        symbol=symbol,
        themes=themes,
        gdelt_timespan=args.timespan,
        gdelt_max_per_theme=args.gdelt_max,
        google_max_items=args.news_max,
    )

    if snapshot.errors:
        console.print("[yellow]Partial data — errors on:[/yellow]")
        for src, msg in snapshot.errors.items():
            console.print(f"  [dim]{src}[/dim]: {msg}")

    _print_google_news(
        snapshot.google_news_query or build_google_news_query(symbol), args.news_max
    )

    for theme_result in snapshot.gdelt_themes:
        console.rule(
            f"[bold]GDELT {theme_result.theme} — {theme_result.label}[/bold]"
        )
        if theme_result.error:
            console.print(f"[red]Error: {theme_result.error}[/red]")
            continue
        if not theme_result.articles:
            console.print(f"[yellow]No articles in last {args.timespan}.[/yellow]")
            continue
        table = Table(show_lines=False)
        table.add_column("Seen (UTC)", style="dim", no_wrap=True)
        table.add_column("Country", no_wrap=True)
        table.add_column("Domain", style="cyan", no_wrap=True)
        table.add_column("Title")
        for art in theme_result.articles:
            table.add_row(
                art.iso_date[:10] if art.iso_date else "?",
                art.source_country or "?",
                art.domain or "?",
                art.title,
            )
        console.print(table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
