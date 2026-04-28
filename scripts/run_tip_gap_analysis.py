"""Tip → screening gap analysis (constitution Sec 7).

Compares tickers the user has mentioned via the Telegram tip channel
against tickers the screening pipeline surfaced. Three categories
emerge:

  - mentioned_and_surfaced: both sides agree
  - mentioned_only:         user attention NOT confirmed by rubric
  - surfaced_only:          system found these without user attention

The reason this report exists at all (per constitution v2.0 § 7):
> "the set of tickers the user mentioned but the system did not
>  surface is itself useful. It represents the gap between the user's
>  attention and the system's rubric."

The output is for the user to read. It is NOT fed into any LLM in any
stage. If you want to use this report to refine the rubric, do it by
inspecting the gap manually and deciding whether to revise the
constitution; do NOT thread the gap into model prompts.

Usage:
    python scripts/run_tip_gap_analysis.py NVDA AAPL MSFT
    python scripts/run_tip_gap_analysis.py --tickers-file survivors.txt
    python scripts/run_tip_gap_analysis.py --from-screening screening.json
    python scripts/run_tip_gap_analysis.py --window-days 60 NVDA AAPL
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.ingest.tip_annotation import (  # noqa: E402
    DEFAULT_WINDOW_DAYS,
    GapReport,
    compute_gap_analysis,
    lookup_tip_annotations,
)
from wise_investor.ingest.tip_store import TipStore  # noqa: E402

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tickers",
        nargs="*",
        help="System-surfaced tickers to compare against the tip log.",
    )
    parser.add_argument(
        "--tickers-file",
        type=Path,
        help="File with one surfaced ticker per line.",
    )
    parser.add_argument(
        "--from-screening",
        type=Path,
        help=(
            "Path to a screening JSON (run_screening.py --json). Pulls "
            "ADVANCE_TO_STAGE_3 tickers from the prefilter rows."
        ),
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"Tip log lookback window (default {DEFAULT_WINDOW_DAYS} days).",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Override TipStore database path (default uses settings.sqlite_path).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of formatted tables.",
    )
    args = parser.parse_args()

    surfaced = _resolve_surfaced(args)
    if not surfaced:
        console.print(
            "[yellow]No surfaced tickers provided.[/yellow] "
            "Gap analysis still useful — will list all mention-only tickers."
        )

    store = TipStore(db_path=args.db) if args.db else TipStore()

    report = compute_gap_analysis(
        surfaced, store, window_days=args.window_days
    )

    if args.json:
        print(json.dumps(_to_json(report), indent=2, default=str))
        return 0

    _print_report(report, store=store, surfaced_input=surfaced)
    return 0


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def _resolve_surfaced(args: argparse.Namespace) -> list[str]:
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


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_report(
    report: GapReport,
    *,
    store: TipStore,
    surfaced_input: list[str],
) -> None:
    console.print(
        Panel.fit(
            (
                f"[bold]Tip → screening gap analysis[/bold]\n"
                f"Window: last {report.window_days} days\n"
                f"Surfaced (system): {report.n_surfaced}\n"
                f"Mentioned (user): {report.n_mentioned}\n"
                f"Overlap: {len(report.mentioned_and_surfaced)} "
                f"({report.overlap_ratio:.1%} of mentions)"
            ),
            border_style="cyan",
        )
    )

    # Mentioned + surfaced (the "rubric agrees with attention" set)
    if report.mentioned_and_surfaced:
        console.print()
        console.rule("[bold green]Mentioned and surfaced[/bold green]")
        annotations = lookup_tip_annotations(
            list(report.mentioned_and_surfaced),
            store,
            window_days=report.window_days,
        )
        ms = Table(show_lines=False)
        ms.add_column("Symbol", style="bold")
        ms.add_column("Mentions")
        ms.add_column("Annotation")
        for sym in report.mentioned_and_surfaced:
            ann = annotations.get(sym)
            ms.add_row(
                sym,
                str(report.by_ticker_mentions.get(sym, 0)),
                ann.render() if ann else "[dim]—[/dim]",
            )
        console.print(ms)

    # Mentioned only — user attention without rubric confirmation
    if report.mentioned_only:
        console.print()
        console.rule("[bold yellow]Mentioned only (rubric did NOT surface)[/bold yellow]")
        console.print(
            "[dim]These tickers attracted the user's attention but the "
            "system either rejected them or didn't analyze them. Worth "
            "reviewing — either rubric blind spot, or noise in the user's "
            "attention pattern.[/dim]"
        )
        annotations = lookup_tip_annotations(
            list(report.mentioned_only),
            store,
            window_days=report.window_days,
        )
        mo = Table(show_lines=False)
        mo.add_column("Symbol", style="bold")
        mo.add_column("Mentions")
        mo.add_column("Annotation")
        for sym in report.mentioned_only:
            ann = annotations.get(sym)
            mo.add_row(
                sym,
                str(report.by_ticker_mentions.get(sym, 0)),
                ann.render() if ann else "[dim]—[/dim]",
            )
        console.print(mo)

    # Surfaced only — system found these on its own
    if report.surfaced_only:
        console.print()
        console.rule("[bold blue]Surfaced only (user did NOT mention)[/bold blue]")
        console.print(
            "[dim]System-discovered candidates outside the user's recent "
            "attention. This is the constitution's main reason to exist — "
            "ideas the rubric finds without user bias.[/dim]"
        )
        so = Table(show_lines=False)
        so.add_column("Symbol", style="bold")
        for sym in report.surfaced_only:
            so.add_row(sym)
        console.print(so)


def _to_json(report: GapReport) -> dict:
    payload = asdict(report)
    payload["mentioned_and_surfaced"] = list(report.mentioned_and_surfaced)
    payload["mentioned_only"] = list(report.mentioned_only)
    payload["surfaced_only"] = list(report.surfaced_only)
    payload["overlap_ratio"] = report.overlap_ratio
    payload["n_mentioned"] = report.n_mentioned
    payload["n_surfaced"] = report.n_surfaced
    return payload


if __name__ == "__main__":
    sys.exit(main())
