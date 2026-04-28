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
    parser.add_argument(
        "--with-stage3",
        action="store_true",
        help=(
            "After the Stage 2 quant pre-filter, also run the Stage 3 "
            "qualitative LLM screen via the active LLM backend. Requires "
            "Ollama (or the configured backend) to be reachable; falls "
            "back to REJECT on any LLM failure per Commitment 3."
        ),
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help=(
            "Print the Stage 3 prompt that would be sent to the LLM "
            "(useful for prompt debugging) and exit without calling it."
        ),
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

    # --- Stage 3 (optional)
    if args.show_prompt:
        from wise_investor.screening.stage3_prompts import render_prompt_for_inspection
        console.print()
        console.rule("[bold]Stage 3 prompt (preview only)[/bold]")
        console.print(render_prompt_for_inspection(funds, result))
        return 0

    if args.with_stage3:
        from wise_investor.screening.llm_screening import screen_ticker

        console.print()
        console.rule("[bold]Stage 3 — qualitative LLM screening[/bold]")
        if result.hierarchy_decision == "REJECT":
            console.print(
                "[yellow]Stage 2 already rejected; skipping Stage 3 LLM call.[/yellow]"
            )
            return 0

        stage3 = screen_ticker(funds, result)

        s3_color = "green" if stage3.hierarchy_decision == "ADVANCE_TO_STAGE_4" else "red"
        decision_text = (
            "[green]ADVANCE_TO_STAGE_4[/green]"
            if stage3.hierarchy_decision == "ADVANCE_TO_STAGE_4"
            else "[red]REJECT[/red]"
        )
        console.print(
            Panel.fit(
                f"[bold]{stage3.symbol}[/bold]\nStage 3 decision: {decision_text}",
                border_style=s3_color,
            )
        )
        if stage3.rejection_reason:
            console.print(f"[yellow]Rejection reason:[/yellow] {stage3.rejection_reason}")
        if stage3.llm_reported_decision and stage3.llm_reported_decision != stage3.hierarchy_decision:
            console.print(
                f"[dim]LLM reported {stage3.llm_reported_decision}; "
                "caller recomputed gate per §9.[/dim]"
            )

        s3_table = Table(title="Stage 3 axis outcomes", show_lines=True)
        s3_table.add_column("Axis", style="bold")
        s3_table.add_column("Verdict")
        s3_table.add_column("Qualifier")
        s3_table.add_column("Reasoning")
        for outcome in (stage3.moat, stage3.new_frontier, stage3.bottleneck):
            color = {"PASS": "green", "FAIL": "red", "INVALID": "magenta"}[
                outcome.verdict
            ]
            s3_table.add_row(
                outcome.axis,
                f"[{color}]{outcome.verdict}[/{color}]",
                outcome.qualifier or "—",
                outcome.reasoning,
            )
        console.print(s3_table)

    return 0


if __name__ == "__main__":
    sys.exit(main())
