"""Long-polling Telegram receiver for incoming tip messages.

Mirror of `notify.telegram.TelegramNotifier` but for the inbound
direction. Uses Telegram's `getUpdates` endpoint with long-polling
(default 25 s blocking wait) to pull new messages without burning
CPU on tight loops.

Offset persistence: Telegram delivers each update once the bot
acknowledges it by passing `offset = last_update_id + 1` on the
next call. We persist the last processed id to a file
(`data/telegram_last_update.txt`) so bot restarts don't reprocess
messages already handled (and don't lose queued messages either —
Telegram keeps undelivered updates for 24 hours).

Access control:
  - By default the receiver FILTERS to `settings.telegram_chat_id`
    so random strangers can't issue /analyze commands and burn our
    GPU budget.
  - Passing `allow_chat_ids=None` disables the filter; used only
    for debugging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import httpx

from wise_investor.config import PROJECT_ROOT, settings


logger = logging.getLogger(__name__)


_DEFAULT_OFFSET_PATH = PROJECT_ROOT / "data" / "telegram_last_update.txt"


@dataclass
class TelegramMessage:
    """Normalized shape of an incoming Telegram message."""

    update_id: int
    message_id: int
    chat_id: int
    text: str
    sender_username: str | None
    sender_first_name: str | None
    date_unix: int


class TelegramReceiver:
    """Long-poll client for the Telegram Bot API."""

    BASE_URL = "https://api.telegram.org"

    def __init__(
        self,
        bot_token: str | None = None,
        allow_chat_ids: set[int] | None = None,
        offset_path: Path | None = None,
        timeout: float = 30.0,
        long_poll_timeout: int = 25,
    ) -> None:
        self.bot_token = bot_token if bot_token is not None else settings.telegram_bot_token
        self.timeout = timeout
        self.long_poll_timeout = long_poll_timeout
        self.offset_path = offset_path if offset_path is not None else _DEFAULT_OFFSET_PATH

        # Default to only the configured chat_id for safety. `None` is
        # allowed explicitly — tests and debug sessions use it.
        if allow_chat_ids is None and settings.telegram_chat_id:
            try:
                allow_chat_ids = {int(settings.telegram_chat_id)}
            except ValueError:
                logger.warning(
                    "telegram_chat_id %r is not an int; disabling filter.",
                    settings.telegram_chat_id,
                )
                allow_chat_ids = None
        self.allow_chat_ids = allow_chat_ids

    @property
    def configured(self) -> bool:
        return bool(
            self.bot_token
            and self.bot_token not in ("", "your_telegram_bot_token")
        )

    # ---- Offset persistence -----------------------------------------

    def _load_offset(self) -> int | None:
        if not self.offset_path.exists():
            return None
        try:
            return int(self.offset_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError) as e:
            logger.warning("Offset file %s unreadable (%s); starting fresh.", self.offset_path, e)
            return None

    def _save_offset(self, update_id: int) -> None:
        try:
            self.offset_path.parent.mkdir(parents=True, exist_ok=True)
            self.offset_path.write_text(str(update_id), encoding="utf-8")
        except OSError as e:
            logger.warning("Could not persist offset %d to %s: %s", update_id, self.offset_path, e)

    # ---- Polling ----------------------------------------------------

    def poll_updates(self) -> list[TelegramMessage]:
        """Issue one getUpdates call and return new messages.

        Blocks up to `self.long_poll_timeout` seconds. Returns `[]` on
        timeout (no new messages) or when not configured. Never raises
        on network / API errors — failures are logged and processing
        continues on the next poll.
        """
        if not self.configured:
            logger.debug("Telegram receiver not configured; skipping poll.")
            return []

        offset = self._load_offset()
        params: dict[str, object] = {
            "timeout": self.long_poll_timeout,
            "allowed_updates": '["message"]',
        }
        if offset is not None:
            params["offset"] = offset + 1

        url = f"{self.BASE_URL}/bot{self.bot_token}/getUpdates"
        try:
            r = httpx.get(url, params=params, timeout=self.timeout + self.long_poll_timeout)
        except httpx.TimeoutException:
            # Long-poll hit its natural timeout with no messages.
            return []
        except Exception as e:
            logger.warning("Telegram getUpdates failed: %s", e)
            return []

        if r.status_code >= 400:
            logger.warning(
                "Telegram getUpdates returned %d: %s",
                r.status_code,
                r.text[:200],
            )
            return []

        try:
            payload = r.json()
        except ValueError:
            logger.warning("Telegram getUpdates returned non-JSON: %r", r.text[:200])
            return []

        if not payload.get("ok"):
            logger.warning("Telegram getUpdates payload not ok: %s", str(payload)[:200])
            return []

        results = payload.get("result") or []
        messages: list[TelegramMessage] = []
        max_update_id: int | None = None

        for update in results:
            try:
                update_id = int(update["update_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if max_update_id is None or update_id > max_update_id:
                max_update_id = update_id

            msg = _parse_update(update)
            if msg is None:
                continue
            if self.allow_chat_ids is not None and msg.chat_id not in self.allow_chat_ids:
                logger.info(
                    "Ignoring message from unapproved chat_id=%s (allowed=%s)",
                    msg.chat_id,
                    self.allow_chat_ids,
                )
                continue
            messages.append(msg)

        # Persist offset even if we filtered all messages — we've
        # ACK'd these updates either way, and Telegram should not
        # redeliver them.
        if max_update_id is not None:
            self._save_offset(max_update_id)

        return messages


def _parse_update(update: dict) -> TelegramMessage | None:
    """Extract our normalized TelegramMessage from a raw Update dict.

    Only processes `message` updates (we don't subscribe to edited_
    message, channel_post, callback_query, etc.). Returns None for
    non-text messages (photos, stickers) so the bot logs them as
    unsupported rather than crashing.
    """
    try:
        update_id = int(update["update_id"])
    except (KeyError, TypeError, ValueError):
        return None

    message = update.get("message")
    if not isinstance(message, dict):
        return None

    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        message_id = int(message["message_id"])
        chat = message["chat"]
        chat_id = int(chat["id"])
        date_unix = int(message.get("date") or 0)
    except (KeyError, TypeError, ValueError):
        return None

    sender = message.get("from") or {}
    sender_username = sender.get("username") if isinstance(sender, dict) else None
    sender_first_name = sender.get("first_name") if isinstance(sender, dict) else None

    return TelegramMessage(
        update_id=update_id,
        message_id=message_id,
        chat_id=chat_id,
        text=text,
        sender_username=sender_username if isinstance(sender_username, str) else None,
        sender_first_name=sender_first_name if isinstance(sender_first_name, str) else None,
        date_unix=date_unix,
    )


__all__ = ["TelegramMessage", "TelegramReceiver"]
