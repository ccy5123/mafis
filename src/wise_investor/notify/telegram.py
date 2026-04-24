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


# Telegram hard limit per sendMessage request. Values over this are
# rejected with HTTP 400 "message is too long". We chunk at a bit
# under the cap to leave room for a trailing continuation hint.
TELEGRAM_MAX_MESSAGE_LEN = 4096
_CHUNK_SAFE_LEN = 3900


def _chunk_text(text: str, limit: int = _CHUNK_SAFE_LEN) -> list[str]:
    """Split a long message into Telegram-sized chunks, preferring
    paragraph / line boundaries over mid-sentence breaks.
    """
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        # Prefer a double-newline split; fall back to single newline;
        # last resort is a hard cut at `limit`.
        cut = window.rfind("\n\n")
        if cut < limit * 0.5:
            cut = window.rfind("\n")
        if cut < limit * 0.5:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


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

        Automatically chunks messages over Telegram's 4096-char limit into
        multiple sendMessage calls, preserving paragraph / line boundaries.
        Returns True only if ALL chunks succeeded.
        """
        if not self.configured:
            logger.debug(
                "Telegram not configured; skipping push (len=%d)", len(text)
            )
            return False

        url = f"{self.BASE_URL}/bot{self.bot_token}/sendMessage"
        chunks = _chunk_text(text)
        if len(chunks) > 1:
            logger.info(
                "Telegram: splitting %d-char message into %d chunks",
                len(text),
                len(chunks),
            )

        for i, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                # Add a "(part i/N)" suffix so the reader knows more is coming.
                chunk = f"{chunk}\n\n(part {i}/{len(chunks)})"
            payload: dict[str, object] = {
                "chat_id": self.chat_id,
                "text": chunk,
                "disable_web_page_preview": disable_web_page_preview,
            }
            if parse_mode is not None:
                payload["parse_mode"] = parse_mode
            try:
                r = httpx.post(url, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    logger.warning(
                        "Telegram API %d on sendMessage (chunk %d/%d): %s",
                        r.status_code,
                        i,
                        len(chunks),
                        r.text[:200],
                    )
                    return False
            except Exception as e:
                logger.warning(
                    "Telegram send failed on chunk %d/%d: %s", i, len(chunks), e
                )
                return False
        return True


    def send_document(
        self,
        path: str,
        caption: str | None = None,
        parse_mode: Literal["MarkdownV2", "Markdown", "HTML"] | None = None,
    ) -> bool:
        """Upload a file via sendDocument so mobile clients can open it
        with a single tap (filesystem paths are unclickable on mobile).

        Returns True on HTTP 200, False on skip / failure. Never raises.

        Telegram's sendDocument accepts files up to 50 MB — well above
        any crew report (typical ~20 KB). We do NOT chunk the file;
        if it ever exceeds the limit the API returns 413 and we log
        the failure.
        """
        import pathlib

        if not self.configured:
            logger.debug(
                "Telegram not configured; skipping document push (%s)", path
            )
            return False

        file_path = pathlib.Path(path)
        if not file_path.exists():
            logger.warning("Telegram send_document: file not found %s", path)
            return False

        url = f"{self.BASE_URL}/bot{self.bot_token}/sendDocument"
        data: dict[str, str] = {"chat_id": str(self.chat_id)}
        if caption is not None:
            # Captions are limited to 1024 chars by Telegram.
            data["caption"] = caption[:1024]
            if parse_mode is not None:
                data["parse_mode"] = parse_mode

        try:
            with open(file_path, "rb") as f:
                files = {"document": (file_path.name, f, "text/markdown")}
                r = httpx.post(url, data=data, files=files, timeout=self.timeout)
        except Exception as e:
            logger.warning("Telegram send_document failed: %s", e)
            return False

        if r.status_code >= 400:
            logger.warning(
                "Telegram API %d on sendDocument: %s",
                r.status_code,
                r.text[:200],
            )
            return False
        return True


def push_if_configured(text: str) -> bool:
    """Convenience: one-shot push using default env settings."""
    return TelegramNotifier().send(text)
