"""Re-push an existing crew report through the new Telegram pipeline.

Skips the 20-minute crew run and uses whatever `reports/<NAME>.crew.md`
is already on disk. Applies the full notify pipeline: audit-aware
verdict extraction, multi-language summary, Ollama translation of
the attached .md. Handy for verifying push-side fixes (sizing
override, trailing-period preservation, multi-language rendering)
without burning GPU time.

Usage:
  python scripts/repush_report.py reports/NVDA_20260424_1137.crew.md
  python scripts/repush_report.py reports/NVDA_20260424_1137.crew.md --lang en
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402
from wise_investor.notify.summary import (  # noqa: E402
    SUPPORTED_LANGUAGES,
    extract_verdict_summary,
    format_summary,
)
from wise_investor.notify.telegram import TelegramNotifier  # noqa: E402
from wise_investor.translation import translate_report  # noqa: E402


console = Console()


def _parse_symbol_from_filename(path: Path) -> str:
    m = re.match(r"^([A-Z0-9.-]+)_", path.name)
    return m.group(1).upper() if m else "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_path", type=Path)
    parser.add_argument(
        "--lang",
        default=None,
        help=(
            "Override user_language for this push (ko/en/ja/zh). "
            "Default: settings.user_language from .env."
        ),
    )
    args = parser.parse_args()

    report_path: Path = args.report_path
    if not report_path.exists():
        console.print(f"[red]Not found:[/red] {report_path}")
        return 1

    symbol = _parse_symbol_from_filename(report_path)
    combined_markdown = report_path.read_text(encoding="utf-8")

    user_lang = (args.lang or settings.user_language or "ko").lower()
    if user_lang not in SUPPORTED_LANGUAGES:
        console.print(
            f"[yellow]Unsupported lang={user_lang!r}; using 'ko'.[/yellow]"
        )
        user_lang = "ko"

    console.rule(f"[bold]Re-push {symbol} ({user_lang}) — {report_path.name}[/bold]")

    notifier = TelegramNotifier()
    if not notifier.configured:
        console.print("[red]Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.[/red]")
        return 2

    # --- Summary push (narrative fragments kept in English source form)
    summary = extract_verdict_summary(symbol, combined_markdown)
    summary_text = format_summary(summary, lang=user_lang)
    console.print("[dim]--- Summary preview ---[/dim]")
    console.print(summary_text)
    console.print("[dim]-----------------------[/dim]")

    sent = notifier.send(summary_text)
    console.print(
        f"[cyan]📨 summary push[/cyan] → {'OK' if sent else 'FAILED'}"
    )

    # --- Translate + attach
    attach_path = report_path
    caption_by_lang = {
        "ko": f"📄 {symbol} 전체 리포트",
        "en": f"📄 {symbol} full report",
        "ja": f"📄 {symbol} 全体レポート",
        "zh": f"📄 {symbol} 完整报告",
    }
    doc_caption = caption_by_lang.get(user_lang, caption_by_lang["ko"])

    if user_lang != "en":
        console.print(
            f"[dim]Translating report → {user_lang} via Ollama "
            f"({settings.analyst_model}) …[/dim]"
        )
        t0 = time.perf_counter()
        translated_md = translate_report(combined_markdown, target_lang=user_lang)
        elapsed = time.perf_counter() - t0
        translated_path = report_path.with_suffix("").with_suffix(
            f".crew.{user_lang}.md"
        )
        # with_suffix double-call strips ".md"+".crew" incorrectly on some
        # filenames, so build the target path manually to be safe.
        translated_path = report_path.parent / (
            report_path.stem.replace(".crew", "") + f".crew.{user_lang}.md"
        )
        translated_path.write_text(translated_md, encoding="utf-8")
        attach_path = translated_path
        console.print(
            f"[cyan]🌐 translated saved[/cyan] → {translated_path} "
            f"({elapsed:.1f}s, {len(translated_md)} chars)"
        )

    doc_sent = notifier.send_document(str(attach_path), caption=doc_caption)
    console.print(
        f"[cyan]📎 document push[/cyan] → {'OK' if doc_sent else 'FAILED'}"
    )
    return 0 if sent and doc_sent else 3


if __name__ == "__main__":
    sys.exit(main())
