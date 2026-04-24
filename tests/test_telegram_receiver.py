"""Tests for the long-poll Telegram receiver.

All HTTP traffic is stubbed via httpx.MockTransport so tests run
offline and are deterministic.
"""

from __future__ import annotations

import json

import httpx
import pytest

from wise_investor.ingest.telegram_receiver import (
    TelegramMessage,
    TelegramReceiver,
    _parse_update,
)


# ---------------------------------------------------------------------------
# _parse_update — shape handling
# ---------------------------------------------------------------------------


def test_parse_update_extracts_text_message() -> None:
    update = {
        "update_id": 100,
        "message": {
            "message_id": 5,
            "from": {"id": 42, "username": "cyjoe", "first_name": "CY"},
            "chat": {"id": 42, "type": "private"},
            "date": 1713960000,
            "text": "엔비디아 좋대",
        },
    }
    msg = _parse_update(update)
    assert msg is not None
    assert msg.update_id == 100
    assert msg.message_id == 5
    assert msg.chat_id == 42
    assert msg.text == "엔비디아 좋대"
    assert msg.sender_username == "cyjoe"
    assert msg.sender_first_name == "CY"
    assert msg.date_unix == 1713960000


def test_parse_update_returns_none_for_non_message_update() -> None:
    """Updates without a `message` field (e.g. edited_message only) are skipped."""
    update = {
        "update_id": 100,
        "edited_message": {"message_id": 5, "text": "edited"},
    }
    assert _parse_update(update) is None


def test_parse_update_returns_none_for_non_text_message() -> None:
    """Photos, stickers, voice — messages without a `text` field are skipped."""
    update = {
        "update_id": 100,
        "message": {
            "message_id": 5,
            "chat": {"id": 42, "type": "private"},
            "date": 1713960000,
            "photo": [{"file_id": "abc"}],
        },
    }
    assert _parse_update(update) is None


def test_parse_update_handles_missing_sender_fields() -> None:
    update = {
        "update_id": 100,
        "message": {
            "message_id": 5,
            "chat": {"id": 42, "type": "private"},
            "date": 1713960000,
            "text": "hi",
            # no `from` field
        },
    }
    msg = _parse_update(update)
    assert msg is not None
    assert msg.sender_username is None
    assert msg.sender_first_name is None


def test_parse_update_rejects_blank_text() -> None:
    update = {
        "update_id": 100,
        "message": {
            "message_id": 5,
            "chat": {"id": 42, "type": "private"},
            "date": 1713960000,
            "text": "   \n",
        },
    }
    assert _parse_update(update) is None


# ---------------------------------------------------------------------------
# Offset persistence
# ---------------------------------------------------------------------------


def test_offset_file_not_read_when_missing(tmp_path) -> None:
    r = TelegramReceiver(
        bot_token="fake", allow_chat_ids={1}, offset_path=tmp_path / "offset.txt"
    )
    assert r._load_offset() is None


def test_offset_saved_and_reloaded(tmp_path) -> None:
    path = tmp_path / "offset.txt"
    r = TelegramReceiver(bot_token="fake", allow_chat_ids={1}, offset_path=path)
    r._save_offset(12345)
    assert r._load_offset() == 12345

    # A fresh receiver pointing at the same file should see it.
    r2 = TelegramReceiver(bot_token="fake", allow_chat_ids={1}, offset_path=path)
    assert r2._load_offset() == 12345


def test_offset_file_corrupt_is_ignored(tmp_path) -> None:
    path = tmp_path / "offset.txt"
    path.write_text("not an int", encoding="utf-8")
    r = TelegramReceiver(bot_token="fake", allow_chat_ids={1}, offset_path=path)
    assert r._load_offset() is None


# ---------------------------------------------------------------------------
# poll_updates — full HTTP roundtrip (mocked)
# ---------------------------------------------------------------------------


def _mock_response(updates: list[dict]) -> httpx.Response:
    return httpx.Response(
        200, content=json.dumps({"ok": True, "result": updates}).encode("utf-8")
    )


def _make_receiver(tmp_path, allow_chat_ids=None) -> TelegramReceiver:
    return TelegramReceiver(
        bot_token="fake",
        allow_chat_ids=allow_chat_ids if allow_chat_ids is not None else {42},
        offset_path=tmp_path / "offset.txt",
        timeout=1.0,
        long_poll_timeout=1,
    )


def test_poll_updates_not_configured_returns_empty(tmp_path) -> None:
    r = TelegramReceiver(
        bot_token="", allow_chat_ids={42}, offset_path=tmp_path / "offset.txt"
    )
    assert r.poll_updates() == []


def test_poll_updates_placeholder_token_returns_empty(tmp_path) -> None:
    r = TelegramReceiver(
        bot_token="your_telegram_bot_token",
        allow_chat_ids={42},
        offset_path=tmp_path / "offset.txt",
    )
    assert r.poll_updates() == []


def test_poll_updates_returns_parsed_messages(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)
    captured_params: dict = {}

    def _mock_get(url, params=None, timeout=None):
        captured_params.update(params or {})
        return _mock_response([
            {
                "update_id": 100,
                "message": {
                    "message_id": 1,
                    "from": {"id": 42, "username": "cyjoe"},
                    "chat": {"id": 42},
                    "date": 1000,
                    "text": "/analyze NVDA",
                },
            }
        ])

    monkeypatch.setattr("httpx.get", _mock_get)
    msgs = r.poll_updates()
    assert len(msgs) == 1
    assert msgs[0].text == "/analyze NVDA"
    # long-poll timeout shipped to Telegram.
    assert captured_params.get("timeout") == 1


def test_poll_updates_filters_unapproved_chat_ids(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path, allow_chat_ids={42})

    def _mock_get(url, params=None, timeout=None):
        return _mock_response([
            {
                "update_id": 100,
                "message": {
                    "message_id": 1,
                    "chat": {"id": 999},  # not allowed
                    "date": 1000,
                    "text": "should be ignored",
                },
            },
            {
                "update_id": 101,
                "message": {
                    "message_id": 2,
                    "chat": {"id": 42},  # allowed
                    "date": 1001,
                    "text": "should pass through",
                },
            },
        ])

    monkeypatch.setattr("httpx.get", _mock_get)
    msgs = r.poll_updates()
    assert [m.text for m in msgs] == ["should pass through"]
    # Offset advances past BOTH updates so Telegram doesn't redeliver
    # the ignored one on the next poll.
    assert r._load_offset() == 101


def test_poll_updates_accepts_any_chat_when_filter_disabled(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path, allow_chat_ids=set())  # empty = still filtering
    # Explicitly disable via set-to-None requires bypassing __init__ default;
    # construct directly.
    r.allow_chat_ids = None

    def _mock_get(url, params=None, timeout=None):
        return _mock_response([
            {
                "update_id": 100,
                "message": {
                    "message_id": 1,
                    "chat": {"id": 999},
                    "date": 1000,
                    "text": "anyone",
                },
            }
        ])

    monkeypatch.setattr("httpx.get", _mock_get)
    msgs = r.poll_updates()
    assert len(msgs) == 1


def test_poll_updates_sends_offset_after_first_batch(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)
    call_log: list[dict] = []

    def _mock_get(url, params=None, timeout=None):
        call_log.append(dict(params or {}))
        if len(call_log) == 1:
            return _mock_response([
                {
                    "update_id": 100,
                    "message": {
                        "message_id": 1,
                        "chat": {"id": 42},
                        "date": 1000,
                        "text": "first",
                    },
                }
            ])
        return _mock_response([])

    monkeypatch.setattr("httpx.get", _mock_get)
    r.poll_updates()  # first batch → saves offset=100
    r.poll_updates()  # second batch → should send offset=101

    assert "offset" not in call_log[0]  # no prior offset on cold start
    assert call_log[1].get("offset") == 101


def test_poll_updates_swallows_http_error(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)

    def _mock_get(url, params=None, timeout=None):
        return httpx.Response(500, content=b"server error")

    monkeypatch.setattr("httpx.get", _mock_get)
    assert r.poll_updates() == []  # no raise


def test_poll_updates_swallows_network_exception(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)

    def _mock_get(url, params=None, timeout=None):
        raise httpx.ConnectError("mock outage")

    monkeypatch.setattr("httpx.get", _mock_get)
    assert r.poll_updates() == []


def test_poll_updates_swallows_timeout(tmp_path, monkeypatch) -> None:
    """Long-poll timeout (nothing to read) must return [] silently."""
    r = _make_receiver(tmp_path)

    def _mock_get(url, params=None, timeout=None):
        raise httpx.TimeoutException("long-poll timeout")

    monkeypatch.setattr("httpx.get", _mock_get)
    assert r.poll_updates() == []


def test_poll_updates_swallows_non_json(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)

    def _mock_get(url, params=None, timeout=None):
        return httpx.Response(200, content=b"not json at all")

    monkeypatch.setattr("httpx.get", _mock_get)
    assert r.poll_updates() == []


def test_poll_updates_swallows_non_ok_payload(tmp_path, monkeypatch) -> None:
    r = _make_receiver(tmp_path)

    def _mock_get(url, params=None, timeout=None):
        return httpx.Response(
            200, content=json.dumps({"ok": False, "description": "x"}).encode()
        )

    monkeypatch.setattr("httpx.get", _mock_get)
    assert r.poll_updates() == []
