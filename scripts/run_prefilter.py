"""Stage 2 quantitative pre-filter — single-ticker CLI for calibration.

Reads a ticker fundamentals fixture from JSON, runs the constitution
v2.0 §16 pipeline, prints the structured result.

Usage:
    python scripts/run_prefilter.py path/to/fixture.json

The fixture format is documented in
`src/wise_investor/screening/adapters.py::dump_fundamentals_template`.

This is the CLI used during Step 4 calibration: hand-curate a JSON
file per ticker, run the prefilter on each, compare verdicts to your
intuition, refine the constitution thresholds if a systematic
mismatch appears.
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

from wise_investor.screening.adapters import (  # noqa: E402
    load_fundamentals_from_json,
)
from wise_investor.screening.prefilter import evaluate_ticker  # noqa: E402
from wise_investor.screening.segments import (  # noqa: E402
    resolve_primary_segment,
    single_segment_default,
)


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture",
        type=Path,
        help="Path to a ticker fundamentals JSON fixture.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON output instead of the formatted table.",
    )
    args = parser.parse_args()

    if not args.fixture.exists():
        console.print(f"[red]Fixture not found: {args.fixture}[/red]")
        return 1

    funds = load_fundamentals_from_json(args.fixture)

    # Pick the latest fiscal year's segment breakdown for the §13
    # gate. If history is empty, fall back to single-segment default.
    if funds.segments_history:
        latest = max(funds.segments_history, key=lambda sb: sb.fiscal_year)
        # The fixture may have already set primary_segment_exists; if
        # so, trust it. Otherwise re-resolve from the segment list.
        if latest.primary_segment_exists:
            primary = latest
        else:
            primary = resolve_primary_segment(
                list(latest.all_segments),
                fiscal_year=latest.fiscal_year,
                source=latest.source,
            )
    else:
        primary = single_segment_default(
            funds.symbol,
            fiscal_year=funds.annual[-1].fiscal_year if funds.annual else 0,
        )

    result = evaluate_ticker(funds, primary)

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
        return 0

    # Pretty-print summary.
    header = (
        f"[bold]{result.symbol}[/bold]  "
        f"(constitution v{result.constitution_version})"
    )
    panel_color = "green" if result.hierarchy_decision == "ADVANCE_TO_STAGE_3" else "red"
    decision_text = (
        "[green]ADVANCE_TO_STAGE_3[/green]"
        if result.hierarchy_decision == "ADVANCE_TO_STAGE_3"
        else "[red]REJECT[/red]"
    )
    console.print(
        Panel.fit(
            f"{header}\nDecision: {decision_text}",
            border_style=panel_color,
        )
    )

    if result.excluded_reason:
        console.print(f"[yellow]Excluded reason:[/yellow] {result.excluded_reason}")

    # Per-axis table.
    table = Table(title="Axis verdicts", show_lines=True)
    table.add_column("Axis", style="bold")
    table.add_column("Verdict")
    table.add_column("Reason")
    for axis_verdict in (result.moat, result.new_frontier, result.bottleneck):
        color = {
            "PASS": "green",
            "FAIL": "red",
            "NEED_LLM": "yellow",
        }[axis_verdict.verdict]
        table.add_row(
            axis_verdict.axis,
            f"[{color}]{axis_verdict.verdict}[/{color}]",
            axis_verdict.reason,
        )
    console.print(table)

    # Primary segment summary.
    if result.primary_segment is not None and result.primary_segment.primary_segment_exists:
        ps = result.primary_segment
        console.print(
            f"\n[dim]Primary segment:[/dim] "
            f"{ps.primary_segment_name} "
            f"({(ps.primary_segment_revenue_share or 0):.1%}) "
            f"@ FY{ps.fiscal_year}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
