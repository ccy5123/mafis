"""Phase 1D quality-score CLI.

Usage:
    python scripts/score_report.py reports/NVDA_20260422_1827.crew.md
    python scripts/score_report.py <report.md> [--no-facts]

Runs every automated quality metric on the given report and prints a Rich
table. When a facts cache exists for the same symbol+date, the invention
audit metric is included automatically. Results are saved alongside the
report as <report>.score.json for regression tracking.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.quality.metrics import (  # noqa: E402
    MetricResult,
    score_report,
)


console = Console()


def _infer_facts_for_report(
    report_path: Path,
) -> tuple[dict[str, str] | None, str | None]:
    """Given `reports/<SYMBOL>_<YYYYMMDD>_<HHMM>.crew.md`, look for a
    same-day facts cache and the symbol's value chain brief.

    Returns (facts, value_chain_text) — either may be None if missing.
    """
    name = report_path.stem  # e.g. NVDA_20260422_1827.crew
    m = re.match(r"^([A-Z][A-Z0-9.]{0,9})_(\d{4})(\d{2})(\d{2})_", name)
    if not m:
        return None, None
    symbol, yyyy, mm, dd = m.groups()

    facts: dict[str, str] | None = None
    cache = REPO_ROOT / "data" / "facts_cache" / f"{symbol}_{yyyy}-{mm}-{dd}.json"
    if cache.exists():
        try:
            facts = json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            console.print(f"[yellow]Could not load facts cache {cache}: {e}[/yellow]")

    value_chain: str | None = None
    vc_path = REPO_ROOT / "docs" / "value_chains" / f"{symbol}.md"
    if vc_path.exists():
        try:
            value_chain = vc_path.read_text(encoding="utf-8")
        except Exception:
            pass

    return facts, value_chain


def _render_table(results: list[MetricResult], report_label: str) -> Table:
    t = Table(title=f"Quality score — {report_label}", show_lines=False)
    t.add_column("Metric")
    t.add_column("Value", justify="right")
    t.add_column("Unit")
    t.add_column("Threshold", justify="right")
    t.add_column("Pass", justify="center")
    t.add_column("Explanation", overflow="fold")
    for r in results:
        if r.passed is True:
            pass_cell = "[green]✓[/green]"
        elif r.passed is False:
            pass_cell = "[red]✗[/red]"
        else:
            pass_cell = "[yellow]—[/yellow]"
        threshold_str = (
            "—" if r.threshold is None else f"{r.threshold:g}"
        )
        t.add_row(
            r.name,
            f"{r.value:g}",
            r.unit,
            threshold_str,
            pass_cell,
            r.explanation,
        )
    return t


def run(report_path: Path, use_facts: bool = True) -> int:
    if not report_path.exists():
        console.print(f"[red]Report not found: {report_path}[/red]")
        return 1

    text = report_path.read_text(encoding="utf-8")
    facts: dict[str, str] | None = None
    value_chain: str | None = None
    if use_facts:
        facts, value_chain = _infer_facts_for_report(report_path)
    if facts is None and use_facts:
        console.print(
            "[yellow]No same-day facts cache found — invention_audit will be skipped.[/yellow]"
        )
    if value_chain is None and use_facts:
        console.print(
            "[yellow]No value chain brief found — curated numbers won't be in the audit pool.[/yellow]"
        )

    results = score_report(text, facts, value_chain_text=value_chain)
    console.print(_render_table(results, report_path.name))

    passed = sum(1 for r in results if r.passed is True)
    failed = sum(1 for r in results if r.passed is False)
    skipped = sum(1 for r in results if r.passed is None)
    summary = (
        f"[bold]Summary:[/bold] "
        f"[green]{passed} passed[/green] · "
        f"[red]{failed} failed[/red] · "
        f"[yellow]{skipped} n/a[/yellow]"
    )
    console.print(summary)

    # Persist scores alongside the report for later comparison.
    score_path = report_path.with_suffix(report_path.suffix + ".score.json")
    score_data = {
        "report": str(report_path.name),
        "metrics": [r.model_dump() for r in results],
        "summary": {"passed": passed, "failed": failed, "skipped": skipped},
    }
    score_path.write_text(json.dumps(score_data, indent=2, ensure_ascii=False))
    console.print(f"[dim]Scores saved: {score_path}[/dim]")
    return 0 if failed == 0 else 2


def main() -> int:
    args = sys.argv[1:]
    if not args:
        console.print(
            "Usage: python scripts/score_report.py <report.md> [--no-facts]"
        )
        return 1
    use_facts = "--no-facts" not in args
    paths = [Path(a) for a in args if not a.startswith("--")]
    rc = 0
    for p in paths:
        rc = run(p, use_facts=use_facts) or rc
        console.print()  # spacer between reports
    return rc


if __name__ == "__main__":
    sys.exit(main())
