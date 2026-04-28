"""Constitution v2.0 Stage 4 crew runner — gated by Stage 2 + Stage 3.

This is the v2 counterpart to `scripts/run_crew.py`. The differences:

  1. Pre-gate: Stage 2 (quant prefilter) + Stage 3 (light LLM screen)
     run BEFORE the crew. Tickers that don't survive Stage 3 don't
     consume the full 6-agent debate budget.
  2. v2 prompts: Skeptic / Defender / Steward use the constitution-§19/
     §20/§21 prompts via the runner_adapter shim, which derives the
     AttackPlan from Stage 3's passed_axes.
  3. Steward audit: the Skeptic ↔ Defender ↔ Steward audit chain runs
     deterministically (agents.v2.audit) and the result is embedded in
     the Steward prompt — the LLM doesn't recompute it.

Side effects (paper trading, Telegram, translation) are intentionally
omitted in this first cut. They live in `run_crew.py`'s v1 path and
can be ported once v2 reports prove themselves on real tickers.

Usage:
    python scripts/run_crew_v2.py NVDA
    python scripts/run_crew_v2.py 005930.KS
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.agents.analyst import ANALYST_BACKSTORY, ANALYST_GOAL  # noqa: E402
from wise_investor.agents.economist import make_economist_system_prompt  # noqa: E402
from wise_investor.agents.runner import (  # noqa: E402
    pre_gather_facts,
    run_crew_synthesis,
)
from wise_investor.agents.tasks import (  # noqa: E402
    CONTEXT_INSTRUCTIONS,
    REPORT_TEMPLATE,
    _load_value_chain,
    make_economist_user_prompt,
    make_valuer_user_prompt,
)
from wise_investor.agents.v2.runner_adapter import build_v2_prompt_bundle  # noqa: E402
from wise_investor.agents.valuer import make_valuer_system_prompt  # noqa: E402
from wise_investor.config import settings  # noqa: E402
from wise_investor.screening.live_adapter import fetch_live_fundamentals  # noqa: E402
from wise_investor.screening.llm_screening import screen_ticker  # noqa: E402
from wise_investor.screening.prefilter import evaluate_ticker  # noqa: E402

console = Console()
REPORTS_DIR = REPO_ROOT / "reports"


def _build_analyst_system() -> str:
    """Reused verbatim from run_crew.py — Analyst prompt is unchanged
    in v2 (only Skeptic / Defender / Steward differ)."""
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
# Gating logic
# ---------------------------------------------------------------------------


def _run_gate(symbol: str) -> tuple[str, list[str], object | None]:
    """Run Stage 2 + Stage 3 and return (decision, passed_axes, stage3).

    `decision` is one of:
      - "STAGE4": Stage 3 advanced; passed_axes contains 2+ axes.
      - "REJECT_STAGE2": Stage 2 prefilter said REJECT.
      - "REJECT_STAGE3": Stage 3 LLM said REJECT.
      - "ERROR": fundamentals fetch or screening failed.

    `stage3` is the Stage3Result object when STAGE4, else None.
    """
    console.rule(f"[bold]Stage 2 + 3 gate[/bold] — {symbol}")

    try:
        funds = fetch_live_fundamentals(symbol)
    except Exception as e:
        console.print(f"[red]Fundamentals fetch failed: {e}[/red]")
        return ("ERROR", [], None)

    primary = funds.segments_history[-1] if funds.segments_history else None
    prefilter = evaluate_ticker(funds, primary)

    console.print(
        f"  Stage 2: [bold]{prefilter.hierarchy_decision}[/bold] "
        f"(moat={prefilter.moat.verdict}, "
        f"frontier={prefilter.new_frontier.verdict}, "
        f"bottleneck={prefilter.bottleneck.verdict})"
    )

    if prefilter.hierarchy_decision != "ADVANCE_TO_STAGE_3":
        return ("REJECT_STAGE2", [], None)

    try:
        stage3 = screen_ticker(funds, prefilter)
    except Exception as e:
        console.print(f"[red]Stage 3 LLM screening failed: {e}[/red]")
        return ("ERROR", [], None)

    console.print(
        f"  Stage 3: [bold]{stage3.hierarchy_decision}[/bold] "
        f"(moat={stage3.moat.verdict}, "
        f"frontier={stage3.new_frontier.verdict}, "
        f"bottleneck={stage3.bottleneck.verdict})"
    )

    if stage3.hierarchy_decision != "ADVANCE_TO_STAGE_4":
        return ("REJECT_STAGE3", [], stage3)

    passed_axes = [
        outcome.axis
        for outcome in (stage3.moat, stage3.new_frontier, stage3.bottleneck)
        if outcome.verdict == "PASS"
    ]
    return ("STAGE4", passed_axes, stage3)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run(symbol: str) -> int:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        console.print("[red]FINNHUB_API_KEY not set in .env[/red]")
        return 1

    symbol = symbol.upper()
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS_DIR / f"{symbol}_{stamp}.crew_v2.md"
    meta_path = REPORTS_DIR / f"{symbol}_{stamp}.crew_v2.meta.txt"

    decision, passed_axes, stage3 = _run_gate(symbol)
    if decision != "STAGE4":
        console.print()
        console.print(
            Panel.fit(
                f"[red]Gate result: {decision}[/red]\n"
                f"Stage 4 SKIPPED — ticker did not survive screening.",
                border_style="red",
            )
        )
        return 0  # not an error; the gate did its job

    console.rule(f"[bold]Stage 4 v2 crew[/bold] — {symbol}")
    console.print(f"Stage 3 passed axes: [cyan]{', '.join(passed_axes)}[/cyan]")
    console.print(f"Report → [dim]{report_path}[/dim]")

    def log(msg: str) -> None:
        console.print(msg)

    log("[pre-gather] running Phase 1A tools (or loading cache) …")
    t0 = time.perf_counter()
    facts = pre_gather_facts(symbol)
    pre_gather_elapsed = time.perf_counter() - t0
    log(f"[pre-gather] {len(facts)} tool outputs ready in {pre_gather_elapsed:.1f}s")

    value_chain_text = _load_value_chain(symbol)

    # v2 prompt bundle: derived from Stage 3 passed_axes at this point
    v2_bundle = build_v2_prompt_bundle(passed_axes)

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
        skeptic_system=v2_bundle["skeptic_system"],
        skeptic_user_prompt_builder=v2_bundle["skeptic_user_prompt_builder"],
        defender_system=v2_bundle["defender_system"],
        defender_user_prompt_builder=v2_bundle["defender_user_prompt_builder"],
        steward_system=v2_bundle["steward_system"],
        steward_user_prompt_builder=v2_bundle["steward_user_prompt_builder"],
        run_tag=f"{symbol}_{stamp}",
        log_fn=log,
    )
    result.pre_gather_elapsed = pre_gather_elapsed

    report_path.write_text(result.combined_markdown, encoding="utf-8")
    meta_path.write_text(
        f"symbol: {symbol}\n"
        f"started: {stamp}\n"
        f"constitution_version: 2.0\n"
        f"passed_axes: {','.join(passed_axes)}\n"
        f"economist_model: {result.economist_model}\n"
        f"analyst_model: {result.analyst_model}\n"
        f"valuer_model: {result.valuer_model}\n"
        f"skeptic_model: {result.skeptic_model}\n"
        f"defender_model: {result.defender_model}\n"
        f"steward_model: {result.steward_model}\n"
        f"pre_gather_sec: {result.pre_gather_elapsed:.1f}\n"
        f"economist_sec: {result.economist_elapsed:.1f}\n"
        f"analyst_sec: {result.analyst_elapsed:.1f}\n"
        f"valuer_sec: {result.valuer_elapsed:.1f}\n"
        f"skeptic_sec: {result.skeptic_elapsed:.1f}\n"
        f"defender_sec: {result.defender_elapsed:.1f}\n"
        f"steward_sec: {result.steward_elapsed:.1f}\n"
        f"total_sec: {result.total_elapsed + result.pre_gather_elapsed:.1f}\n",
        encoding="utf-8",
    )

    total_min = (result.total_elapsed + result.pre_gather_elapsed) / 60
    console.rule(f"[green]v2 crew done in {total_min:.1f} min[/green]")
    console.print(f"Report: [cyan]{report_path}[/cyan]")
    console.print(f"Meta:   [dim]{meta_path}[/dim]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbol",
        nargs="?",
        default="NVDA",
        help="Ticker to analyze (default: NVDA).",
    )
    args = parser.parse_args()
    return run(args.symbol)


if __name__ == "__main__":
    sys.exit(main())
