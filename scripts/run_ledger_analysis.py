"""Calibration ledger analysis CLI.

Reads `data/calibration_ledger/v*.json` entries and produces:
  - Confusion-matrix metrics (precision, recall, F1, accuracy)
  - Per-ticker TP/FP/TN/FN classification
  - Side-by-side diff between two entries (e.g., constitution v2.0 vs
    a hypothetical v2.1 on the same manifest + calibration date)

Usage:
    python scripts/run_ledger_analysis.py                       # latest entry
    python scripts/run_ledger_analysis.py --list                # all entries
    python scripts/run_ledger_analysis.py --entry path.json     # specific
    python scripts/run_ledger_analysis.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.screening.ledger_analysis import (  # noqa: E402
    LedgerAnalysis,
    LedgerComparison,
    analyze_entry,
    compare_entries,
    list_ledger_entries,
    load_ledger_entry,
)

console = Console()
DEFAULT_LEDGER_DIR = REPO_ROOT / "data" / "calibration_ledger"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=DEFAULT_LEDGER_DIR,
        help=f"Ledger directory (default: {DEFAULT_LEDGER_DIR.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all ledger entries with one-line summaries.",
    )
    parser.add_argument(
        "--entry",
        type=Path,
        help="Analyze a specific ledger JSON file.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("A", "B"),
        help="Compare two ledger entries side by side.",
    )
    args = parser.parse_args()

    if args.list:
        return _cmd_list(args.ledger_dir)

    if args.compare:
        return _cmd_compare(args.compare[0], args.compare[1])

    # Default: analyze the latest entry (or args.entry if specified)
    target = args.entry
    if target is None:
        entries = list_ledger_entries(args.ledger_dir)
        if not entries:
            console.print(
                f"[red]No ledger entries found in {args.ledger_dir}.[/red]"
            )
            console.print(
                "[yellow]Run `python scripts/run_back_validation.py` "
                "to populate the ledger.[/yellow]"
            )
            return 1
        target = entries[-1]
        console.print(
            f"[dim]No --entry specified; analyzing latest: "
            f"{target.relative_to(REPO_ROOT)}[/dim]\n"
        )

    return _cmd_analyze(target)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_list(ledger_dir: Path) -> int:
    entries = list_ledger_entries(ledger_dir)
    if not entries:
        console.print(f"[red]No entries in {ledger_dir}.[/red]")
        return 1

    table = Table(title=f"Ledger entries — {ledger_dir.relative_to(REPO_ROOT)}")
    table.add_column("File", style="cyan")
    table.add_column("Version")
    table.add_column("Cal. date")
    table.add_column("N tickers", justify="right")
    table.add_column("N advanced", justify="right")
    table.add_column("Adv. avg excess", justify="right")

    for path in entries:
        try:
            payload = load_ledger_entry(path)
        except Exception as e:
            table.add_row(path.name, "[red]ERROR[/red]", str(e)[:40], "", "", "")
            continue
        adv_avg = payload.get("advanced_avg_excess_return")
        adv_avg_str = f"{adv_avg:+.1%}" if adv_avg is not None else "—"
        table.add_row(
            path.name,
            payload.get("constitution_version", "?"),
            payload.get("calibration_date", "?"),
            str(payload.get("n_tickers", 0)),
            str(payload.get("n_advanced", 0)),
            adv_avg_str,
        )
    console.print(table)
    return 0


def _cmd_analyze(path: Path) -> int:
    if not path.exists():
        console.print(f"[red]Entry not found: {path}[/red]")
        return 1

    payload = load_ledger_entry(path)
    analysis = analyze_entry(payload)
    _print_analysis(analysis, path=path)
    return 0


def _cmd_compare(path_a: Path, path_b: Path) -> int:
    if not path_a.exists():
        console.print(f"[red]Entry A not found: {path_a}[/red]")
        return 1
    if not path_b.exists():
        console.print(f"[red]Entry B not found: {path_b}[/red]")
        return 1

    a = load_ledger_entry(path_a)
    b = load_ledger_entry(path_b)
    comparison = compare_entries(a, b)
    _print_comparison(comparison)
    return 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _print_analysis(a: LedgerAnalysis, *, path: Path) -> None:
    try:
        path_display = path.relative_to(REPO_ROOT)
    except ValueError:
        path_display = path

    console.print(
        Panel.fit(
            (
                f"[bold]Calibration ledger analysis[/bold]\n"
                f"File: {path_display}\n"
                f"Constitution: v{a.constitution_version}\n"
                f"Calibration date: {a.calibration_date}\n"
                f"Horizon date: {a.horizon_date}\n"
                f"N tickers: {a.n_tickers} "
                f"({a.n_advanced} advanced, {a.n_rejected} rejected)"
            ),
            border_style="cyan",
        )
    )

    # Confusion matrix
    cm = a.confusion
    console.print()
    console.rule("[bold]Confusion matrix[/bold]")
    cm_table = Table(show_header=True)
    cm_table.add_column("")
    cm_table.add_column("Outperformed", justify="right")
    cm_table.add_column("Underperformed", justify="right")
    cm_table.add_row(
        "[bold]Advanced[/bold]",
        f"[green]{cm.tp} (TP)[/green]",
        f"[red]{cm.fp} (FP)[/red]",
    )
    cm_table.add_row(
        "[bold]Rejected[/bold]",
        f"[red]{cm.fn} (FN)[/red]",
        f"[green]{cm.tn} (TN)[/green]",
    )
    console.print(cm_table)
    if cm.undefined > 0:
        console.print(
            f"[dim]Undefined (no return data): {cm.undefined}[/dim]"
        )

    # Metrics
    console.print()
    metrics = Table(show_header=False, title="Classifier metrics")
    metrics.add_column()
    metrics.add_column(justify="right")
    metrics.add_row("Precision (of advances, % beat)", _fmt_pct(cm.precision))
    metrics.add_row("Recall (of winners, % advanced)", _fmt_pct(cm.recall))
    metrics.add_row("F1", _fmt_pct(cm.f1))
    metrics.add_row("Accuracy", _fmt_pct(cm.accuracy))
    metrics.add_row("Adv. avg excess return", _fmt_pct(a.advanced_avg_excess_return))
    metrics.add_row("Rej. avg excess return", _fmt_pct(a.rejected_avg_excess_return))
    console.print(metrics)

    # Per-ticker classification, sorted by excess descending
    console.print()
    console.rule("[bold]Per-ticker classification[/bold]")
    tt = Table(show_header=True)
    tt.add_column("Symbol", style="bold")
    tt.add_column("Predicted")
    tt.add_column("Outcome")
    tt.add_column("Excess return", justify="right")
    tt.add_column("Class")

    sorted_classifications = sorted(
        a.classifications,
        key=lambda c: (
            c.excess_return is None,
            -(c.excess_return if c.excess_return is not None else 0),
        ),
    )
    for c in sorted_classifications:
        excess_str = (
            f"{c.excess_return:+.1%}" if c.excess_return is not None else "—"
        )
        cls_color = {
            "TP": "green",
            "TN": "green",
            "FP": "red",
            "FN": "red",
            "UNDEFINED": "dim",
        }.get(c.classification, "white")
        tt.add_row(
            c.symbol,
            "ADVANCE" if c.predicted_advance else "REJECT",
            "[green]beat[/green]" if c.actually_outperformed else "[red]missed[/red]",
            excess_str,
            f"[{cls_color}]{c.classification}[/{cls_color}]",
        )
    console.print(tt)


def _print_comparison(c: LedgerComparison) -> None:
    console.print(
        Panel.fit(
            (
                f"[bold]Ledger comparison[/bold]\n"
                f"A: {c.a_label}\n"
                f"B: {c.b_label}"
            ),
            border_style="cyan",
        )
    )

    console.print()
    metrics = Table(title="Classifier metrics A vs B", show_header=True)
    metrics.add_column("Metric")
    metrics.add_column("A", justify="right")
    metrics.add_column("B", justify="right")
    metrics.add_column("Δ", justify="right")
    for label, a_val, b_val in (
        ("Precision", c.a_metrics.precision, c.b_metrics.precision),
        ("Recall", c.a_metrics.recall, c.b_metrics.recall),
        ("F1", c.a_metrics.f1, c.b_metrics.f1),
        ("Accuracy", c.a_metrics.accuracy, c.b_metrics.accuracy),
    ):
        delta = (
            f"{(b_val - a_val):+.2%}"
            if (a_val is not None and b_val is not None)
            else "—"
        )
        metrics.add_row(
            label, _fmt_pct(a_val), _fmt_pct(b_val), delta
        )
    console.print(metrics)

    if c.flipped_to_advance:
        console.print()
        console.rule("[bold green]Flipped: REJECT → ADVANCE[/bold green]")
        console.print(", ".join(c.flipped_to_advance))

    if c.flipped_to_reject:
        console.print()
        console.rule("[bold red]Flipped: ADVANCE → REJECT[/bold red]")
        console.print(", ".join(c.flipped_to_reject))

    if c.a_only_symbols:
        console.print()
        console.print(
            f"[dim]Only in A: {', '.join(c.a_only_symbols)}[/dim]"
        )
    if c.b_only_symbols:
        console.print(
            f"[dim]Only in B: {', '.join(c.b_only_symbols)}[/dim]"
        )


def _fmt_pct(v: float | None) -> str:
    return f"{v:.2%}" if v is not None else "—"


if __name__ == "__main__":
    sys.exit(main())
