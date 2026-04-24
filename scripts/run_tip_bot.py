"""Telegram tip-bot daemon.

Long-polls Telegram for incoming messages, routes each one to the
dispatcher (which either stores it as a tip or invokes a crew run),
and loops until Ctrl+C.

Usage:
    python scripts/run_tip_bot.py

Requires:
    TELEGRAM_BOT_TOKEN in .env
    TELEGRAM_CHAT_ID   in .env (used as the allow-list for commands)

Persistent state:
    data/telegram_last_update.txt — last processed update_id so
    restarts don't reprocess messages already handled.
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path

from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402
from wise_investor.ingest.dispatcher import TipDispatcher  # noqa: E402
from wise_investor.ingest.telegram_receiver import TelegramReceiver  # noqa: E402
from wise_investor.ingest.tip_store import TipStore  # noqa: E402
from wise_investor.notify.telegram import TelegramNotifier  # noqa: E402


console = Console()


_shutdown = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _shutdown
    _shutdown = True
    console.print("\n[yellow]Shutdown requested — finishing current batch…[/yellow]")


def main() -> int:
    if not settings.telegram_bot_token or settings.telegram_bot_token in (
        "", "your_telegram_bot_token",
    ):
        console.print("[red]TELEGRAM_BOT_TOKEN not set in .env — cannot start bot.[/red]")
        return 1
    if not settings.telegram_chat_id:
        console.print(
            "[red]TELEGRAM_CHAT_ID not set in .env — the bot would accept "
            "commands from anyone. Refusing to start.[/red]"
        )
        return 1

    notifier = TelegramNotifier()
    receiver = TelegramReceiver()
    store = TipStore()
    dispatcher = TipDispatcher(
        tip_store=store,
        reply_fn=lambda text: _reply(notifier, text),
    )

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    console.rule("[bold]MAFIS Tip Bot[/bold]")
    console.print(f"Chat ID allow-list: [cyan]{settings.telegram_chat_id}[/cyan]")
    console.print(f"DB: [dim]{store.db_path}[/dim]")
    console.print(
        f"Offset file: [dim]{receiver.offset_path}[/dim] "
        f"(last id: {receiver._load_offset()})"
    )
    console.print("[green]Polling… Ctrl+C to stop.[/green]")

    consecutive_errors = 0
    while not _shutdown:
        try:
            messages = receiver.poll_updates()
        except Exception as e:
            consecutive_errors += 1
            console.print(f"[yellow]Poll error #{consecutive_errors}: {e}[/yellow]")
            if consecutive_errors >= 5:
                console.print(
                    "[red]5 consecutive poll errors — sleeping 60s before retry.[/red]"
                )
                time.sleep(60)
                consecutive_errors = 0
            else:
                time.sleep(5)
            continue

        consecutive_errors = 0

        for msg in messages:
            try:
                result = dispatcher.dispatch(msg)
            except Exception as e:
                console.print(f"[red]Dispatch error: {e}[/red]")
                continue

            preview = (msg.text or "")[:60].replace("\n", " ")
            console.print(
                f"[dim]{msg.date_unix}[/dim] "
                f"[cyan]{msg.sender_username or '?'}[/cyan]: "
                f"{preview!r} → [green]{result.action}[/green]"
            )

    console.print("[yellow]Bot stopped.[/yellow]")
    return 0


def _reply(notifier: TelegramNotifier, text: str) -> None:
    """Adapter: dispatcher calls reply_fn(text); we route to the
    configured chat_id via TelegramNotifier. Errors are logged but
    don't propagate (dispatcher shouldn't fail because a reply
    bounced).
    """
    sent = notifier.send(text)
    if not sent:
        console.print(f"[yellow]Reply send failed (len={len(text)}).[/yellow]")


if __name__ == "__main__":
    sys.exit(main())
