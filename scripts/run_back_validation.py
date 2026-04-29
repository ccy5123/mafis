"""Calibration runner — back-validate a manifest of tickers and append to the ledger.

Reads `data/calibration/manifest.yaml` (or a custom path), runs
`back_validate_universe` against it on a chosen calibration date, and
writes the structured result to `data/calibration_ledger/`.

Usage:
    python scripts/run_back_validation.py
    python scripts/run_back_validation.py --calibration-date 2017-06-30
    python scripts/run_back_validation.py --manifest path/to/manifest.yaml
    python scripts/run_back_validation.py --limit 5         # debug: first 5
    python scripts/run_back_validation.py --no-write        # don't append to ledger
    python scripts/run_back_validation.py --no-cache        # bypass historical cache

Constitution alignment: this is the §22 back-validation runner — it
executes the rubric on a fixed manifest and records outcomes. The
manifest is NOT a user-preference list (Commitment 1); see the manifest
file's `selection_principle` block for the audit trail.

RAG signal policy (P1a 2026-04):
This runner currently does NOT auto-invoke `extract_rag_signals` for
each ticker. As a result, `top5_customer_share` and
`diversification_attempt_signals` are None / 0 in calibration runs,
and the bottleneck axis routes uniformly to NEED_LLM (the §15 path
1-B threshold needs top-5 share to fire quantitatively). This is one
known contributor to the very high recall numbers in earlier ledger
entries — Stage 2 wasn't gating bottleneck.

Pre-indexing helper: `python scripts/index_universe.py` populates the
ChromaDB 10-K collection for the manifest's US/US-ADR tickers. Wiring
the back-validation runner to actually consume RAG signals is the
P1a-Full / P2-pre-calibration follow-up — once that lands, recall
metrics in the ledger become meaningful for the first time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.screening.back_validation import (  # noqa: E402
    BackValidationSummary,
    back_validate_universe,
)
from wise_investor.screening.calibration_ledger import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    write_ledger_entry,
)

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to manifest YAML (default: data/calibration/manifest.yaml)",
    )
    parser.add_argument(
        "--calibration-date",
        type=dt.date.fromisoformat,
        default=None,
        help="ISO date YYYY-MM-DD; defaults to manifest's default_calibration_date",
    )
    parser.add_argument(
        "--horizon-years",
        type=int,
        default=None,
        help="Defaults to manifest's default_horizon_years (5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to first N tickers (debug)",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the summary but don't append to the calibration ledger.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk historical cache (forces fresh yfinance calls).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        console.print(f"[red]Manifest not found: {args.manifest}[/red]")
        return 1

    manifest = load_manifest(args.manifest)
    calibration_date = args.calibration_date or manifest.default_calibration_date
    horizon_years = args.horizon_years or manifest.default_horizon_years

    symbols = manifest.symbols
    if args.limit:
        symbols = symbols[: args.limit]

    try:
        manifest_display = args.manifest.relative_to(REPO_ROOT)
    except ValueError:
        manifest_display = args.manifest

    console.print(
        Panel.fit(
            f"[bold]Calibration run[/bold]\n"
            f"Manifest: {manifest_display}\n"
            f"Tickers: {len(symbols)}\n"
            f"Calibration date: {calibration_date.isoformat()}\n"
            f"Horizon: {horizon_years} years\n"
            f"Cache: {'disabled' if args.no_cache else 'enabled'}",
            border_style="cyan",
        )
    )

    summary = back_validate_universe(
        symbols,
        calibration_date,
        horizon_years=horizon_years,
        cache=not args.no_cache,
    )

    _print_summary(summary)

    if args.no_write:
        console.print("\n[yellow]--no-write specified; ledger not updated.[/yellow]")
        return 0

    path = write_ledger_entry(summary, manifest=manifest)
    try:
        path_display = path.relative_to(REPO_ROOT)
    except ValueError:
        path_display = path
    console.print(f"\n[green]Ledger entry written:[/green] {path_display}")
    return 0


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _print_summary(summary: BackValidationSummary) -> None:
    if not summary.per_ticker_records:
        console.print("\n[red]No ticker records produced — every ticker errored.[/red]")
        return

    console.print()
    console.rule("[bold]Per-ticker results[/bold]")

    table = Table(show_lines=False)
    table.add_column("Symbol", style="bold")
    table.add_column("Decision")
    table.add_column("Moat")
    table.add_column("Frontier")
    table.add_column("Bottleneck")
    table.add_column("Excess return")

    for r in summary.per_ticker_records:
        decision = r.prefilter_result.hierarchy_decision
        decision_color = "green" if decision == "ADVANCE_TO_STAGE_3" else "red"
        excess = r.return_outcome.excess_return
        excess_str = f"{excess:+.1%}" if excess is not None else "—"
        # Strip "ADVANCE_TO_" prefix to keep the column narrow.
        decision_short = decision.replace("ADVANCE_TO_", "→") if decision != "REJECT" else "REJECT"
        table.add_row(
            r.symbol,
            f"[{decision_color}]{decision_short}[/{decision_color}]",
            _verdict_cell(r.prefilter_result.moat.verdict),
            _verdict_cell(r.prefilter_result.new_frontier.verdict),
            _verdict_cell(r.prefilter_result.bottleneck.verdict),
            excess_str,
        )
    console.print(table)

    console.print()
    console.rule("[bold]Aggregate[/bold]")
    agg = Table(show_header=False)
    agg.add_column()
    agg.add_column()
    agg.add_row("Constitution version", summary.constitution_version)
    agg.add_row("Calibration date", summary.calibration_date.isoformat())
    agg.add_row("Horizon date", summary.horizon_date.isoformat())
    agg.add_row("N tickers", str(summary.n_tickers))
    agg.add_row("N advanced", str(summary.n_advanced))
    agg.add_row("N rejected", str(summary.n_rejected))
    agg.add_row(
        "Advanced avg excess return",
        _fmt_pct(summary.advanced_avg_excess_return),
    )
    agg.add_row(
        "Rejected avg excess return",
        _fmt_pct(summary.rejected_avg_excess_return),
    )
    agg.add_row(
        "Moat persistence rate",
        _fmt_pct(summary.moat_persistence_rate),
    )
    agg.add_row(
        "Bottleneck persistence rate",
        _fmt_pct(summary.bottleneck_persistence_rate),
    )
    console.print(agg)


def _verdict_cell(verdict: str) -> str:
    color = {"PASS": "green", "FAIL": "red", "NEED_LLM": "yellow"}.get(verdict, "white")
    return f"[{color}]{verdict}[/{color}]"


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:+.1%}"


if __name__ == "__main__":
    sys.exit(main())
