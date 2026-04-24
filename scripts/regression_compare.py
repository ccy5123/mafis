"""Compare two crew reports → structured regression diff.

Usage:
    python scripts/regression_compare.py baseline.crew.md new.crew.md
    python scripts/regression_compare.py baseline.md new.md --output diff.md

Use it when you:
  - Tune a prompt and want to confirm it didn't regress quality
    metrics against yesterday's NVDA run.
  - Upgrade the model (Qwen 7B → Qwen 14B) and need a before/after.
  - Audit whether a discipline-matrix tweak actually changed the
    final verdict on historical reports.

The tool compares parsed signals (verdict, conviction, audit status,
all 6 quality metrics, edgar citation count, audit violation count)
rather than raw text. Narrative drift is expected — this exists to
catch STRUCTURAL regressions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.regression.compare import (  # noqa: E402
    compare_reports,
    render_diff_markdown,
)


console = Console()


def run(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline)
    new_path = Path(args.new)
    if not baseline_path.exists():
        console.print(f"[red]Baseline not found: {baseline_path}[/red]")
        return 1
    if not new_path.exists():
        console.print(f"[red]New report not found: {new_path}[/red]")
        return 1

    baseline_text = baseline_path.read_text(encoding="utf-8")
    new_text = new_path.read_text(encoding="utf-8")

    diff = compare_reports(baseline_text, new_text)
    md = render_diff_markdown(diff)

    console.rule(
        f"[bold]{baseline_path.name} → {new_path.name}[/bold]"
    )
    console.print(Markdown(md))

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        console.print(f"[green]Saved diff markdown to {args.output}[/green]")

    # Exit non-zero if there are regressions, so CI can catch them.
    if args.fail_on_regression and diff.regressions:
        console.print(
            f"[red]{len(diff.regressions)} regression(s) detected.[/red]"
        )
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Path to the baseline crew report markdown")
    parser.add_argument("new", help="Path to the new crew report markdown")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save the diff markdown",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit non-zero if any metric regressed (for CI use)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
