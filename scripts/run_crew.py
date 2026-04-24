"""Phase 1C entry point: run the full Analyst -> Valuer -> Skeptic crew.

Usage:
  python scripts/run_crew.py [TICKER]   # default: NVDA

Writes:
  reports/<TICKER>_<YYYYMMDD_HHMM>.crew.md   — combined three-agent report
  reports/<TICKER>_<YYYYMMDD_HHMM>.crew.meta.txt — run metadata per agent

Expected runtime: 10-20 minutes on Qwen 2.5 7B (Analyst, Valuer) + Llama 3.1
8B-16k (Skeptic) / RTX 2060 6GB with one Qwen→Llama model swap.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.agents.analyst import ANALYST_BACKSTORY, ANALYST_GOAL  # noqa: E402
from wise_investor.agents.runner import (  # noqa: E402
    pre_gather_facts,
    run_crew_synthesis,
)
from wise_investor.agents.defender import make_defender_system_prompt  # noqa: E402
from wise_investor.agents.economist import make_economist_system_prompt  # noqa: E402
from wise_investor.agents.skeptic import make_skeptic_system_prompt  # noqa: E402
from wise_investor.agents.steward import make_steward_system_prompt  # noqa: E402
from wise_investor.agents.tasks import (  # noqa: E402
    CONTEXT_INSTRUCTIONS,
    REPORT_TEMPLATE,
    _load_value_chain,
    make_defender_user_prompt,
    make_economist_user_prompt,
    make_skeptic_user_prompt,
    make_steward_user_prompt,
    make_valuer_user_prompt,
)
from wise_investor.agents.valuer import make_valuer_system_prompt  # noqa: E402
from wise_investor.config import settings  # noqa: E402
from wise_investor.notify.summary import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    extract_verdict_summary,
    format_summary,
)
from wise_investor.notify.telegram import TelegramNotifier
from wise_investor.translation import translate_report  # noqa: E402  # noqa: E402


console = Console()
REPORTS_DIR = REPO_ROOT / "reports"


def build_analyst_system() -> str:
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


def build_analyst_task(symbol: str, value_chain: str) -> str:
    return (
        CONTEXT_INSTRUCTIONS.format(symbol=symbol, value_chain=value_chain)
        + "\n\n"
        + REPORT_TEMPLATE.format(symbol=symbol)
    )


def run(symbol: str) -> int:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        console.print("[red]FINNHUB_API_KEY not set in .env[/red]")
        return 1

    symbol = symbol.upper()
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    report_path = REPORTS_DIR / f"{symbol}_{stamp}.crew.md"
    meta_path = REPORTS_DIR / f"{symbol}_{stamp}.crew.meta.txt"

    console.rule(f"[bold]Phase 2 — Full 6-agent crew (debate) run for {symbol}[/bold]")
    console.print(f"Economist: [cyan]{settings.analyst_model}[/cyan] (shares Analyst model)")
    console.print(f"Analyst:   [cyan]{settings.analyst_model}[/cyan]")
    console.print(f"Valuer:    [cyan]{settings.valuer_model}[/cyan]")
    console.print(f"Skeptic:   [magenta]{settings.skeptic_model}[/magenta]")
    console.print(f"Defender:  [cyan]{settings.analyst_model}[/cyan] (shares Analyst model)")
    console.print(f"Steward:   [cyan]{settings.steward_model}[/cyan]")
    console.print(f"Temperature: {settings.llm_temperature}  Seed: {settings.llm_seed}")
    console.print(f"Report → [dim]{report_path}[/dim]")

    def log(msg: str) -> None:
        console.print(msg)

    # -- Pre-gather (Phase 1A tool outputs cached to disk)
    import time
    t0 = time.perf_counter()
    log("[pre-gather] running Phase 1A tools (or loading cache) …")
    facts = pre_gather_facts(symbol)
    pre_gather_elapsed = time.perf_counter() - t0
    log(f"[pre-gather] {len(facts)} tool outputs ready in {pre_gather_elapsed:.1f}s")

    # -- Load value chain brief once; reused by Valuer and Skeptic
    value_chain_text = _load_value_chain(symbol)

    # -- Run the full 5-agent pipeline
    result = run_crew_synthesis(
        symbol=symbol,
        value_chain_text=value_chain_text,
        facts=facts,
        economist_system=make_economist_system_prompt(),
        economist_user_prompt_builder=make_economist_user_prompt,
        analyst_system=build_analyst_system(),
        analyst_task=build_analyst_task(symbol, value_chain_text),
        valuer_system=make_valuer_system_prompt(),
        valuer_user_prompt_builder=make_valuer_user_prompt,
        skeptic_system=make_skeptic_system_prompt(),
        skeptic_user_prompt_builder=make_skeptic_user_prompt,
        defender_system=make_defender_system_prompt(),
        defender_user_prompt_builder=make_defender_user_prompt,
        steward_system=make_steward_system_prompt(),
        steward_user_prompt_builder=make_steward_user_prompt,
        log_fn=log,
    )
    # Attach the pre-gather time so the meta file is complete.
    result.pre_gather_elapsed = pre_gather_elapsed

    report_path.write_text(result.combined_markdown, encoding="utf-8")
    meta_path.write_text(
        f"symbol: {symbol}\n"
        f"started: {stamp}\n"
        f"economist_model: {result.economist_model}\n"
        f"analyst_model: {result.analyst_model}\n"
        f"valuer_model: {result.valuer_model}\n"
        f"skeptic_model: {result.skeptic_model}\n"
        f"defender_model: {result.defender_model}\n"
        f"steward_model: {result.steward_model}\n"
        f"temperature: {settings.llm_temperature}\n"
        f"seed: {settings.llm_seed}\n"
        f"pre_gather_sec: {result.pre_gather_elapsed:.1f}\n"
        f"economist_sec: {result.economist_elapsed:.1f}\n"
        f"analyst_sec: {result.analyst_elapsed:.1f}\n"
        f"valuer_sec: {result.valuer_elapsed:.1f}\n"
        f"skeptic_sec: {result.skeptic_elapsed:.1f}\n"
        f"defender_sec: {result.defender_elapsed:.1f}\n"
        f"steward_sec: {result.steward_elapsed:.1f}\n"
        f"total_sec: {result.total_elapsed + result.pre_gather_elapsed:.1f}\n"
        f"economist_chars: {len(result.economist_text)}\n"
        f"analyst_chars: {len(result.analyst_text)}\n"
        f"valuer_chars: {len(result.valuer_text)}\n"
        f"skeptic_chars: {len(result.skeptic_text)}\n"
        f"defender_chars: {len(result.defender_text)}\n"
        f"steward_chars: {len(result.steward_text)}\n",
        encoding="utf-8",
    )

    total_min = (result.total_elapsed + result.pre_gather_elapsed) / 60
    console.rule(f"[green]Crew done in {total_min:.1f} min[/green]")
    console.print(f"Report: [cyan]{report_path}[/cyan]")
    console.print(f"Meta:   [dim]{meta_path}[/dim]")

    # -- Paper-trading auto-record (Phase 4)
    # Parse the Steward verdict (post-audit) and persist to the ledger
    # so `scripts/paper_ledger.py summary` picks up this trade on the
    # next invocation. Entry price pulled from Finnhub if the ticker
    # is US-listed; Korean tickers skip the live-quote call here and
    # the ledger row stores price_at_verdict=None (caller can backfill).
    try:
        from wise_investor.data.dart_facts import is_korean_ticker
        from wise_investor.paper_trading.ledger import PaperTradeLedger
        from wise_investor.paper_trading.report_parser import parse_crew_report

        summary_for_ledger = parse_crew_report(
            result.combined_markdown, symbol_hint=symbol
        )
        if summary_for_ledger.verdict is not None:
            entry_price: float | None = None
            if not is_korean_ticker(symbol):
                try:
                    from wise_investor.data.finnhub import FinnhubClient
                    with FinnhubClient() as c:
                        entry_price = c.quote(symbol).price
                except Exception as e:
                    console.print(
                        f"[yellow]Entry price fetch failed: {e}[/yellow]"
                    )

            ledger = PaperTradeLedger()
            trade = ledger.record_trade(
                symbol=summary_for_ledger.symbol or symbol,
                verdict=summary_for_ledger.verdict,
                original_verdict=summary_for_ledger.original_verdict
                    or summary_for_ledger.verdict,
                conviction=summary_for_ledger.conviction,
                original_conviction=summary_for_ledger.original_conviction,
                audit_downgraded=summary_for_ledger.audit_downgraded,
                price_at_verdict=entry_price,
                report_path=str(report_path),
            )
            audit_flag = " (audit ↓)" if summary_for_ledger.audit_downgraded else ""
            price_str = (
                f"${entry_price:,.2f}" if entry_price is not None else "n/a"
            )
            console.print(
                f"[cyan]📒 Paper trade #{trade.id} recorded:[/cyan] "
                f"{trade.symbol} {trade.verdict} "
                f"C{trade.conviction or '?'}"
                f"{audit_flag} @ {price_str}"
            )
    except Exception as e:
        console.print(f"[yellow]Paper trade record skipped: {e}[/yellow]")

    # -- Push localized summary to Telegram if configured (no-op otherwise)
    notifier = TelegramNotifier()
    if notifier.configured:
        # Resolve target language. Invalid values (typo in .env) fall
        # back to Korean silently so a config mistake doesn't kill the
        # push. SUPPORTED_LANGUAGES is the locale pack in summary.py.
        user_lang = (settings.user_language or "ko").lower()
        if user_lang not in SUPPORTED_LANGUAGES:
            console.print(
                f"[yellow]Unsupported user_language={user_lang!r}; "
                "falling back to 'ko' for the Telegram summary.[/yellow]"
            )
            user_lang = "ko"

        summary = extract_verdict_summary(symbol, result.combined_markdown)
        # Narrative fragments (bull / rebuttal / position sizing) are
        # left in their English source form. We tried localizing them
        # via Qwen 2.5 7B but the model drifted into Chinese meta-
        # commentary on some snippets, producing worse output than
        # leaving it English. Labels (verdict, conviction, position
        # override) ARE localized because they come from the
        # deterministic LOCALE pack, not an LLM.
        summary_text = format_summary(summary, lang=user_lang)
        sent = notifier.send(summary_text)
        if sent:
            console.print(
                f"[cyan]📨 Telegram summary pushed ({user_lang})[/cyan]"
            )
        else:
            console.print("[yellow]Telegram push failed — see logs[/yellow]")

        # Produce the attached .md in the user's language. English is
        # a no-op so we just attach the original; other languages go
        # through the Ollama translator (Qwen 2.5 7B, temp=0, seed=42).
        # The translated file is saved alongside the English report so
        # the user can inspect both later if needed.
        attach_path = report_path
        caption_by_lang = {
            "ko": f"📄 {symbol} 전체 리포트",
            "en": f"📄 {symbol} full report",
            "ja": f"📄 {symbol} 全体レポート",
            "zh": f"📄 {symbol} 完整报告",
        }
        doc_caption = caption_by_lang.get(user_lang, caption_by_lang["ko"])

        if user_lang != "en":
            try:
                console.print(
                    f"[dim]Translating report → {user_lang} "
                    "(Ollama Qwen) …[/dim]"
                )
                import time as _time
                t_trans_start = _time.perf_counter()
                translated_md = translate_report(
                    result.combined_markdown, target_lang=user_lang
                )
                trans_elapsed = _time.perf_counter() - t_trans_start
                translated_path = REPORTS_DIR / (
                    f"{symbol}_{stamp}.crew.{user_lang}.md"
                )
                translated_path.write_text(translated_md, encoding="utf-8")
                attach_path = translated_path
                console.print(
                    f"[cyan]🌐 Translated report saved:[/cyan] "
                    f"{translated_path} ({trans_elapsed:.1f}s)"
                )
            except Exception as e:
                console.print(
                    f"[yellow]Report translation failed ({e}); "
                    "attaching English .md instead.[/yellow]"
                )

        doc_sent = notifier.send_document(str(attach_path), caption=doc_caption)
        if doc_sent:
            console.print("[cyan]📎 Telegram document pushed[/cyan]")
        else:
            console.print(
                "[yellow]Telegram document push failed — see logs[/yellow]"
            )
    return 0


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    sys.exit(run(ticker))
