"""Telegram integration smoke test.

Usage:
    python scripts/probe_telegram.py
    python scripts/probe_telegram.py --message "custom text"

Reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from .env. If either is
missing, prints a clear diagnostic and exits non-zero. If both are
present, sends a test message and reports success or failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402

from wise_investor.config import settings  # noqa: E402
from wise_investor.notify.telegram import TelegramNotifier  # noqa: E402


console = Console()


DEFAULT_MESSAGE = (
    "✅ MAFIS Telegram integration test\n\n"
    "If you see this, your bot token and chat_id are configured correctly.\n"
    "The crew will push verdict summaries here when reports complete."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--message",
        default=DEFAULT_MESSAGE,
        help="Custom test message (default: integration-test banner)",
    )
    args = parser.parse_args()

    token_set = bool(settings.telegram_bot_token)
    chat_set = bool(settings.telegram_chat_id)

    console.rule("[bold]Telegram config[/bold]")
    console.print(
        f"TELEGRAM_BOT_TOKEN: "
        f"{'[green]set[/green]' if token_set else '[red]MISSING[/red]'}"
        + (
            f" (starts with {settings.telegram_bot_token[:8]}...)"
            if token_set
            else ""
        )
    )
    console.print(
        f"TELEGRAM_CHAT_ID:   "
        f"{'[green]set[/green]' if chat_set else '[red]MISSING[/red]'}"
        + (
            f" ({settings.telegram_chat_id})"
            if chat_set
            else ""
        )
    )

    if not (token_set and chat_set):
        console.print()
        console.print(
            "[red]Missing configuration.[/red] Set TELEGRAM_BOT_TOKEN and "
            "TELEGRAM_CHAT_ID in .env. See docs/usage/ko.md §Telegram "
            "for the 5-step setup."
        )
        return 1

    notifier = TelegramNotifier()
    if not notifier.configured:
        console.print("[red]Notifier reports not configured — check .env values[/red]")
        return 1

    console.print()
    console.print("Sending test message...")
    # parse_mode=None → plain text so the probe doesn't trip on
    # Markdown-like characters in the test message (underscores,
    # asterisks). Real production senders (crew summary, chain alerts)
    # format their own valid Markdown.
    sent = notifier.send(args.message, parse_mode=None)
    if sent:
        console.print("[green]✓ Message sent successfully.[/green]")
        console.print("Check your Telegram chat to confirm receipt.")
        return 0
    console.print(
        "[red]✗ Telegram API rejected the request.[/red] Possible causes:\n"
        "  - TELEGRAM_BOT_TOKEN is malformed or revoked\n"
        "  - TELEGRAM_CHAT_ID is wrong (use the number from getUpdates,\n"
        "    not the @username)\n"
        "  - You haven't sent ANY message to the bot yet (bot can't DM\n"
        "    you until you open a conversation first)"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
