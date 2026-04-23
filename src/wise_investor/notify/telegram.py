"""Thin Telegram Bot API client for push notifications.

Uses the https://api.telegram.org/bot<TOKEN>/sendMessage endpoint directly
(via httpx). No new dependency — python-telegram-bot is heavier than we
need for one-way notifications.

Graceful degradation: if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is empty,
calls to send() silently no-op and log at DEBUG. This lets scripts run
identically on machines without Telegram configured.
"""

from __future__ import annotations

import logging
from typing import Literal

import httpx

from wise_investor.config import settings


logger = logging.getLogger(__name__)


class TelegramError(RuntimeError):
    pass


class TelegramNotifier:
    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else settings.telegram_bot_token
        self.chat_id = chat_id if chat_id is not None else settings.telegram_chat_id
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        """True when both token and chat_id are set to non-placeholder values."""
        return bool(
            self.bot_token
            and self.chat_id
            and self.bot_token not in ("", "your_telegram_bot_token")
        )

    def send(
        self,
        text: str,
        parse_mode: Literal["MarkdownV2", "Markdown", "HTML"] | None = "Markdown",
        disable_web_page_preview: bool = True,
    ) -> bool:
        """Send `text` to the configured chat. Returns True on success, False
        on skip (not configured) or API failure. Never raises — failures are
        logged and execution continues.
        """
        if not self.configured:
            logger.debug(
                "Telegram not configured; skipping push (len=%d)", len(text)
            )
            return False

        url = f"{self.BASE_URL}/bot{self.bot_token}/sendMessage"
        payload: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if parse_mode is not None:
            payload["parse_mode"] = parse_mode
        try:
            r = httpx.post(url, json=payload, timeout=self.timeout)
            if r.status_code >= 400:
                logger.warning(
                    "Telegram API %d on sendMessage: %s", r.status_code, r.text[:200]
                )
                return False
            return True
        except Exception as e:
            logger.warning("Telegram send failed: %s", e)
            return False


def push_if_configured(text: str) -> bool:
    """Convenience: one-shot push using default env settings."""
    return TelegramNotifier().send(text)
