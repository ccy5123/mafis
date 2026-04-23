"""Daily sweep across the 3-Tier ticker registry (design-v2.2 §6).

Usage:
  python scripts/daily_run.py                    # full sweep
  python scripts/daily_run.py --tiers 1          # Tier 1 only
  python scripts/daily_run.py --tiers 2 3        # Tier 2 + 3
  python scripts/daily_run.py --dry-run          # no network, no LLM

Behavior by tier:

  Tier 1: full 5-agent crew via scripts/run_crew.py path. Produces a
          reports/<SYMBOL>_<stamp>.crew.md plus meta and scores. Sequential
          (the GPU cannot hold two LLMs at once).

  Tier 2: pre-gather only — Python runs every Phase 1A calculation tool
          and the FRED macro snapshot, writes data/facts_cache/<SYMBOL>_
          <date>.json. No LLM spend. If the symbol is promoted to Tier 1
          the cache is already warm.

  Tier 3: registry-only. Reported in the summary so the operator knows
          what is being passively tracked. No data fetch.

A daily_run_<stamp>.log lands in reports/ describing the sweep: per-
ticker timing, output paths, and any failures.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.agents.analyst import ANALYST_BACKSTORY, ANALYST_GOAL  # noqa: E402
from wise_investor.agents.economist import make_economist_system_prompt  # noqa: E402
from wise_investor.agents.runner import (  # noqa: E402
    pre_gather_facts,
    run_crew_synthesis,
)
from wise_investor.agents.skeptic import make_skeptic_system_prompt  # noqa: E402
from wise_investor.agents.steward import make_steward_system_prompt  # noqa: E402
from wise_investor.agents.tasks import (  # noqa: E402
    CONTEXT_INSTRUCTIONS,
    REPORT_TEMPLATE,
    _load_value_chain,
    make_economist_user_prompt,
    make_skeptic_user_prompt,
    make_steward_user_prompt,
    make_valuer_user_prompt,
)
from wise_investor.agents.valuer import make_valuer_system_prompt  # noqa: E402
from wise_investor.config import settings  # noqa: E402
from wise_investor.notify.summary import (  # noqa: E402
    extract_verdict_summary,
    format_korean_summary,
)
from wise_investor.notify.telegram import TelegramNotifier  # noqa: E402
from wise_investor.ticker_registry import (  # noqa: E402
    RegistryError,
    Tier,
    load_registry,
)


console = Console()
REPORTS_DIR = REPO_ROOT / "reports"


def _build_analyst_system() -> str:
    return (
        "You are the Senior Equity Research Analyst.\n\n"
        f"Goal: {ANALYST_GOAL}\n\n"
        "--- Operating principles ---\n"
        f"{ANALYST_BACKSTORY}\n\n"
        "--- Output discipline ---\n"
        "You are producing the Analyst section of a combined research note. "
        "Return only that section in markdown, no preamble, no closing, no "
        "commentary on your process."
    )


def _build_analyst_task(symbol: str, value_chain: str) -> str:
    return (
        CONTEXT_INSTRUCTIONS.format(symbol=symbol, value_chain=value_chain)
        + "\n\n"
        + REPORT_TEMPLATE.format(symbol=symbol)
    )


# ---------------------------------------------------------------------------
# Per-tier dispatch
# ---------------------------------------------------------------------------


def run_tier_1(symbol: str, dry_run: bool = False) -> dict:
    """Full 5-agent crew. Returns a summary dict for the sweep log."""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS_DIR / f"{symbol}_{stamp}.crew.md"

    if dry_run:
        return {
            "symbol": symbol,
            "tier": "tier_1",
            "action": "full_crew (dry-run: not executed)",
            "output": str(report_path),
            "elapsed_sec": 0.0,
            "status": "dry_run",
        }

    REPORTS_DIR.mkdir(exist_ok=True)
    value_chain_text = _load_value_chain(symbol)

    t0 = time.perf_counter()
    facts = pre_gather_facts(symbol)
    result = run_crew_synthesis(
        symbol=symbol,
        value_chain_text=value_chain_text,
        facts=facts,
        economist_system=make_economist_system_prompt(),
        economist_user_prompt_builder=make_economist_user_prompt,
        analyst_system=_build_analyst_system(),
        analyst_task=_build_analyst_task(symbol, value_chain_text),
        valuer_system=make_valuer_system_prompt(),
        valuer_user_prompt_builder=make_valuer_user_prompt,
        skeptic_system=make_skeptic_system_prompt(),
        skeptic_user_prompt_builder=make_skeptic_user_prompt,
        steward_system=make_steward_system_prompt(),
        steward_user_prompt_builder=make_steward_user_prompt,
    )
    elapsed = time.perf_counter() - t0
    report_path.write_text(result.combined_markdown, encoding="utf-8")

    notifier = TelegramNotifier()
    pushed = False
    if notifier.configured:
        summary = extract_verdict_summary(symbol, result.combined_markdown)
        korean = format_korean_summary(summary, report_path=str(report_path))
        pushed = notifier.send(korean)

    return {
        "symbol": symbol,
        "tier": "tier_1",
        "action": "full_crew",
        "output": str(report_path),
        "elapsed_sec": round(elapsed, 1),
        "status": "ok",
        "telegram_pushed": pushed,
    }


def run_tier_2(symbol: str, dry_run: bool = False) -> dict:
    """Pre-gather only. Warms the facts cache without LLM spend."""
    if dry_run:
        return {
            "symbol": symbol,
            "tier": "tier_2",
            "action": "pre_gather (dry-run: not executed)",
            "elapsed_sec": 0.0,
            "status": "dry_run",
        }

    t0 = time.perf_counter()
    try:
        facts = pre_gather_facts(symbol)
    except Exception as e:
        return {
            "symbol": symbol,
            "tier": "tier_2",
            "action": "pre_gather",
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(time.perf_counter() - t0, 1),
        }
    elapsed = time.perf_counter() - t0
    return {
        "symbol": symbol,
        "tier": "tier_2",
        "action": "pre_gather",
        "facts_collected": len(facts),
        "elapsed_sec": round(elapsed, 1),
        "status": "ok",
    }


def run_tier_3(symbol: str, notes: str | None, dry_run: bool = False) -> dict:
    """Registry-only. No data fetch. Pure bookkeeping."""
    return {
        "symbol": symbol,
        "tier": "tier_3",
        "action": "registry_only",
        "notes": notes,
        "status": "tracked",
    }


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_sweep(
    tiers: set[str],
    dry_run: bool,
    max_tier_1: int | None = None,
) -> list[dict]:
    """Iterate the registry and run the appropriate action per ticker.

    `max_tier_1` caps the Tier 1 count for partial sweeps on a time budget.
    """
    registry = load_registry(strict=not dry_run)
    results: list[dict] = []

    if "tier_1" in tiers:
        entries = registry.by_tier("tier_1")
        if max_tier_1 is not None:
            entries = entries[:max_tier_1]
        for entry in entries:
            console.print(f"[cyan][Tier 1][/cyan] {entry.symbol}: running full crew …")
            try:
                r = run_tier_1(entry.symbol, dry_run=dry_run)
            except Exception as e:
                r = {
                    "symbol": entry.symbol,
                    "tier": "tier_1",
                    "action": "full_crew",
                    "status": "error",
                    "error": str(e),
                    "traceback": traceback.format_exc()[:400],
                }
            results.append(r)
            if r.get("status") == "ok":
                console.print(
                    f"  [green]ok[/green] {r['elapsed_sec']}s → {r.get('output', '?')}"
                )
            elif r.get("status") == "dry_run":
                console.print("  [dim]dry-run — skipped[/dim]")
            else:
                console.print(f"  [red]error[/red]: {r.get('error', '?')[:120]}")

    if "tier_2" in tiers:
        for entry in registry.by_tier("tier_2"):
            console.print(f"[yellow][Tier 2][/yellow] {entry.symbol}: pre-gather only …")
            r = run_tier_2(entry.symbol, dry_run=dry_run)
            results.append(r)
            if r.get("status") == "ok":
                console.print(
                    f"  [green]ok[/green] {r['elapsed_sec']}s ({r['facts_collected']} outputs)"
                )
            elif r.get("status") == "dry_run":
                console.print("  [dim]dry-run — skipped[/dim]")
            else:
                console.print(f"  [red]error[/red]: {r.get('error', '?')[:120]}")

    if "tier_3" in tiers:
        for entry in registry.by_tier("tier_3"):
            r = run_tier_3(entry.symbol, entry.notes, dry_run=dry_run)
            results.append(r)
            console.print(
                f"[dim][Tier 3] {entry.symbol}:[/dim] tracked only "
                + (f"({entry.notes})" if entry.notes else "")
            )

    return results


def _write_sweep_log(results: list[dict], stamp: str) -> Path:
    log_path = REPORTS_DIR / f"daily_run_{stamp}.log.json"
    REPORTS_DIR.mkdir(exist_ok=True)
    log_path.write_text(
        json.dumps(
            {"started": stamp, "results": results},
            indent=2,
            ensure_ascii=False,
        )
    )
    return log_path


def _render_summary(results: list[dict]) -> Table:
    t = Table(title="Daily sweep summary")
    t.add_column("Tier")
    t.add_column("Symbol")
    t.add_column("Action")
    t.add_column("Status")
    t.add_column("Elapsed (s)", justify="right")
    t.add_column("Detail")
    for r in results:
        status = r.get("status", "?")
        status_cell = (
            f"[green]{status}[/green]"
            if status in ("ok", "tracked")
            else f"[yellow]{status}[/yellow]"
            if status == "dry_run"
            else f"[red]{status}[/red]"
        )
        detail = r.get("output") or r.get("error") or r.get("notes") or ""
        t.add_row(
            r.get("tier", "?"),
            r.get("symbol", "?"),
            r.get("action", "?"),
            status_cell,
            str(r.get("elapsed_sec", "") or ""),
            str(detail)[:60],
        )
    return t


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=["1", "2", "3"],
        default=["1", "2", "3"],
        help="Which tiers to sweep (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan the sweep without executing network or LLM work.",
    )
    parser.add_argument(
        "--max-tier-1",
        type=int,
        default=None,
        help="Cap Tier 1 execution count (time-budget sweep).",
    )
    args = parser.parse_args()

    tier_set = {f"tier_{n}" for n in args.tiers}
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")

    console.rule(f"[bold]Daily sweep — tiers={sorted(tier_set)} dry_run={args.dry_run}[/bold]")

    try:
        results = run_sweep(
            tier_set, dry_run=args.dry_run, max_tier_1=args.max_tier_1
        )
    except RegistryError as e:
        console.print(f"[red]Registry error: {e}[/red]")
        return 1

    console.print()
    console.print(_render_summary(results))
    log_path = _write_sweep_log(results, stamp)
    console.print(f"\n[dim]Sweep log: {log_path}[/dim]")

    # -- One-line Telegram sweep summary if configured
    notifier = TelegramNotifier()
    if notifier.configured and not args.dry_run:
        ok = sum(1 for r in results if r.get("status") == "ok")
        errors = sum(1 for r in results if r.get("status") == "error")
        tracked = sum(1 for r in results if r.get("status") == "tracked")
        text = (
            "🔁 일일 sweep 완료\n\n"
            f"Tier 1 전체 분석: {sum(1 for r in results if r.get('tier') == 'tier_1')}건\n"
            f"Tier 2 pre-gather: {sum(1 for r in results if r.get('tier') == 'tier_2')}건\n"
            f"Tier 3 registry only: {tracked}건\n"
            f"오류: {errors}건"
        )
        notifier.send(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
