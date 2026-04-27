"""Phase 1B first Analyst run — native Ollama tool-calling path.

Replaces the earlier CrewAI-based runner after diagnose_tool_calling.py showed
that CrewAI's LiteLLM path does not surface Ollama's structured tool_calls.
The agent loop in wise_investor.agents.runner is a thin wrapper over
ollama.chat(tools=...) which we confirmed does work.

Usage:
  python scripts/run_analyst.py [TICKER]   # default: NVDA

Writes:
  reports/<TICKER>_<YYYYMMDD_HHMM>.md        — the markdown report
  reports/<TICKER>_<YYYYMMDD_HHMM>.meta.txt  — run metadata
  reports/<TICKER>_<YYYYMMDD_HHMM>.trace.jsonl — one JSON line per tool call
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.agents.analyst import ANALYST_BACKSTORY, ANALYST_GOAL  # noqa: E402
from wise_investor.agents.runner import pre_gather_facts, run_analyst_synthesis  # noqa: E402
from wise_investor.agents.tasks import (  # noqa: E402
    CONTEXT_INSTRUCTIONS,
    REPORT_TEMPLATE,
    _load_value_chain,
)
from wise_investor.config import settings  # noqa: E402


console = Console()
REPORTS_DIR = REPO_ROOT / "reports"


def build_system_prompt() -> str:
    return (
        "You are the Senior Equity Research Analyst.\n\n"
        f"Goal: {ANALYST_GOAL}\n\n"
        "--- Operating principles ---\n"
        f"{ANALYST_BACKSTORY}\n\n"
        "--- Tool-use policy ---\n"
        "You have six tools. You MUST call them to obtain numeric data before "
        "writing any numeric claim in the report. Never state a number that "
        "did not come from a tool call in this session. The calculation tools "
        "return both a computed value and FMP's reported value so you can cite "
        "the source and any divergence. After you have all the numbers you "
        "need, produce the final markdown report in your next message with no "
        "further tool calls."
    )


def build_user_prompt(symbol: str) -> str:
    symbol = symbol.upper()
    value_chain = _load_value_chain(symbol)
    return (
        CONTEXT_INSTRUCTIONS.format(symbol=symbol, value_chain=value_chain)
        + "\n\n"
        + REPORT_TEMPLATE.format(symbol=symbol)
    )


def run(symbol: str) -> int:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        console.print("[red]FMP_API_KEY not set in .env[/red]")
        return 1

    symbol = symbol.upper()
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS_DIR / f"{symbol}_{stamp}.md"
    meta_path = REPORTS_DIR / f"{symbol}_{stamp}.meta.txt"
    trace_path = REPORTS_DIR / f"{symbol}_{stamp}.trace.jsonl"

    from wise_investor.llm import get_agent_config, get_backend
    backend = get_backend()
    cfg = get_agent_config("analyst", backend=backend.name)
    s = cfg.sampling

    console.rule(f"[bold]Phase 1B — Native Analyst run for {symbol}[/bold]")
    console.print(f"Backend: [yellow]{backend.name}[/yellow]")
    console.print(f"Model: [cyan]{cfg.model}[/cyan]")
    console.print(
        f"Sampling: temp={s.temperature}, top_p={s.top_p}"
        + (f", seed={s.seed}" if s.seed is not None else "")
        + f"  (source: {cfg.source})"
    )
    console.print(f"Report → [dim]{report_path}[/dim]")

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(symbol)

    def log(msg: str) -> None:
        console.print(msg)

    # Phase 1B: Python pre-gathers all Phase 1A tool outputs, then the LLM
    # composes narrative only. See src/wise_investor/agents/runner.py for why
    # (diagnose_tool_pressure.py showed small local models skip tool calls
    # when asked for structured multi-section reports).
    log("[pre-gather] running all 6 Phase 1A tools …")
    facts = pre_gather_facts(symbol)
    log(f"[pre-gather] collected {len(facts)} tool outputs")

    result = run_analyst_synthesis(
        system_prompt=system_prompt,
        task_prompt=user_prompt,
        facts=facts,
        model=cfg.model,
        sampling=cfg.sampling,
        log_fn=log,
    )

    report_path.write_text(result.final_text, encoding="utf-8")
    meta_path.write_text(
        f"symbol: {symbol}\n"
        f"backend: {backend.name}\n"
        f"model: {cfg.model}\n"
        f"sampling.analyst: temperature={s.temperature}, top_p={s.top_p}, "
        f"seed={s.seed}, thinking={s.enable_thinking}, source={cfg.source}\n"
        f"started: {stamp}\n"
        f"elapsed_sec: {result.elapsed_sec:.1f}\n"
        f"iterations: {result.iterations}\n"
        f"tool_calls_made: {result.tool_calls_made}\n",
        encoding="utf-8",
    )
    with trace_path.open("w", encoding="utf-8") as f:
        for entry in result.tool_trace:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.rule(
        f"[green]Done — {result.tool_calls_made} tool calls in "
        f"{result.iterations} iterations, {result.elapsed_sec / 60:.1f} min[/green]"
    )
    console.print(f"Report: [cyan]{report_path}[/cyan]")
    console.print(f"Trace:  [dim]{trace_path}[/dim]")

    if result.tool_calls_made == 0:
        console.print(
            "[red]WARNING: zero tool calls — model hallucinated any numbers in the report.[/red]"
        )
        return 2
    return 0


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    sys.exit(run(ticker))
