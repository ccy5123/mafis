"""Tests for the TipDispatcher — the router between Telegram messages
and side effects (tip storage, crew spawning, replies).
"""

from __future__ import annotations

import pytest

from wise_investor.ingest.dispatcher import (
    DispatchResult,
    TipDispatcher,
    _format_tips_list,
)
from wise_investor.ingest.telegram_receiver import TelegramMessage
from wise_investor.ingest.tip_store import TipStore


def _make_msg(text: str, username: str | None = "cyjoe") -> TelegramMessage:
    return TelegramMessage(
        update_id=1,
        message_id=1,
        chat_id=42,
        text=text,
        sender_username=username,
        sender_first_name="CY",
        date_unix=1713960000,
    )


@pytest.fixture
def store(tmp_path) -> TipStore:
    return TipStore(db_path=tmp_path / "tips.sqlite")


@pytest.fixture
def replies() -> list[str]:
    return []


@pytest.fixture
def dispatcher(store, replies):
    """Dispatcher with deterministic extractor (no LLM) and stub crew."""
    spawned: list[str] = []

    def _extractor(text: str) -> list[str]:
        text_lower = text.lower()
        out = []
        if "nvda" in text_lower or "엔비디아" in text_lower or "nvidia" in text_lower:
            out.append("NVDA")
        if "tsmc" in text_lower or "tsm" in text_lower:
            out.append("TSM")
        if "삼성전자" in text_lower or "005930" in text_lower:
            out.append("005930")
        return out

    def _crew_invoker(ticker: str) -> bool:
        spawned.append(ticker)
        return True

    d = TipDispatcher(
        tip_store=store,
        reply_fn=replies.append,
        extractor=_extractor,
        crew_invoker=_crew_invoker,
    )
    d._spawned = spawned  # type: ignore[attr-defined]  # test accessor
    return d


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_help_command_replies_with_help_text(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/help"))
    assert result.action == "help_sent"
    assert len(replies) == 1
    assert "/analyze" in replies[0]
    assert "/tips" in replies[0]


def test_start_command_replies_with_help_text(dispatcher, replies) -> None:
    """Telegram /start convention — same handler as /help."""
    result = dispatcher.dispatch(_make_msg("/start"))
    assert result.action == "help_sent"
    assert "/analyze" in replies[0]


def test_analyze_command_spawns_crew(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/analyze NVDA"))
    assert result.action == "crew_spawned"
    assert result.invoked_ticker == "NVDA"
    assert dispatcher._spawned == ["NVDA"]
    assert "NVDA" in replies[0]
    assert "크루 분석 시작" in replies[0] or "시작" in replies[0]


def test_analyze_command_uppercases_ticker(dispatcher) -> None:
    result = dispatcher.dispatch(_make_msg("/analyze nvda"))
    assert result.invoked_ticker == "NVDA"
    assert dispatcher._spawned == ["NVDA"]


def test_analyze_without_ticker_replies_with_usage(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/analyze"))
    assert result.action == "invalid_args"
    assert dispatcher._spawned == []
    assert "사용법" in replies[0] or "티커" in replies[0]


def test_analyze_reports_failure_when_spawn_fails(store, replies) -> None:
    def _failing_invoker(ticker: str) -> bool:
        return False

    d = TipDispatcher(
        tip_store=store,
        reply_fn=replies.append,
        extractor=lambda t: [],
        crew_invoker=_failing_invoker,
    )
    result = d.dispatch(_make_msg("/analyze NVDA"))
    assert "실패" in replies[0] or "❌" in replies[0]
    assert result.invoked_ticker == "NVDA"


def test_tips_empty_store_replies_with_empty_message(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/tips"))
    assert result.action == "tips_listed"
    assert "저장된 팁이 아직 없습니다" in replies[0]


def test_tips_returns_stored_tips_summary(dispatcher, store, replies) -> None:
    store.record_tip("엔비디아 실적 좋대", detected_tickers=["NVDA"], sender="cyjoe")
    store.record_tip("TSMC 수요 폭발", detected_tickers=["TSM"])

    result = dispatcher.dispatch(_make_msg("/tips"))
    assert result.action == "tips_listed"
    # Both tips appear in the summary.
    assert "엔비디아 실적" in replies[0]
    assert "TSMC 수요" in replies[0]


def test_tips_filter_by_ticker(dispatcher, store, replies) -> None:
    store.record_tip("엔비디아 실적 좋대", detected_tickers=["NVDA"])
    store.record_tip("TSMC 수요 폭발", detected_tickers=["TSM"])

    result = dispatcher.dispatch(_make_msg("/tips NVDA"))
    assert "엔비디아" in replies[0]
    assert "TSMC" not in replies[0]
    assert "(NVDA," in replies[0]


def test_tips_filter_with_empty_ticker_result(dispatcher, store, replies) -> None:
    store.record_tip("엔비디아", detected_tickers=["NVDA"])
    result = dispatcher.dispatch(_make_msg("/tips TSLA"))
    assert result.action == "tips_listed"
    assert "TSLA" in replies[0] and "없습니다" in replies[0]


def test_status_command_replies_with_placeholder(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/status"))
    assert result.action == "status_sent"
    # Phase 1 placeholder — must ship something non-empty.
    assert replies[0] != ""


def test_unknown_command_replies_with_hint(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("/xyz NVDA"))
    assert result.action == "unknown_command"
    assert "알 수 없는" in replies[0] or "?" in replies[0]


# ---------------------------------------------------------------------------
# Free-text tips
# ---------------------------------------------------------------------------


def test_tip_with_detected_ticker_is_stored_and_acknowledged(dispatcher, store, replies) -> None:
    result = dispatcher.dispatch(_make_msg("엔비디아 실적 좋대"))
    assert result.action == "tip_stored"
    assert result.extracted_tickers == ["NVDA"]
    assert result.stored_tip_id is not None

    stored = store.list_tips()
    assert len(stored) == 1
    assert stored[0].raw_text == "엔비디아 실적 좋대"
    assert stored[0].detected_tickers == ["NVDA"]
    assert stored[0].sender == "cyjoe"

    # Acknowledgement includes the detected ticker.
    assert "NVDA" in replies[0]
    assert "저장" in replies[0]


def test_tip_without_detected_ticker_still_stored_but_flagged(
    dispatcher, store, replies
) -> None:
    result = dispatcher.dispatch(_make_msg("오늘 날씨 좋다"))
    assert result.action == "tip_stored"
    assert result.extracted_tickers == []

    stored = store.list_tips()
    assert len(stored) == 1  # still persisted for manual review

    # Reply warns that ticker extraction failed.
    assert "⚠️" in replies[0] or "추출" in replies[0]


def test_tip_uses_first_name_when_username_missing(dispatcher, store) -> None:
    dispatcher.dispatch(_make_msg("NVDA 좋대", username=None))
    stored = store.list_tips()
    assert stored[0].sender == "CY"  # first_name fallback


def test_empty_message_is_noop(dispatcher, replies) -> None:
    result = dispatcher.dispatch(_make_msg("   "))
    assert result.action == "noop"
    assert replies == []


def test_tip_with_multiple_tickers(dispatcher, store, replies) -> None:
    result = dispatcher.dispatch(_make_msg("NVDA 오르면 TSMC 도 같이 오를듯"))
    assert result.action == "tip_stored"
    assert set(result.extracted_tickers or []) == {"NVDA", "TSM"}
    assert "NVDA" in replies[0]
    assert "TSM" in replies[0]


def test_tip_handles_store_failure_gracefully(store, replies, monkeypatch) -> None:
    """If SQLite insert raises, reply with the error rather than crashing."""
    def _broken_insert(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "record_tip", _broken_insert)

    d = TipDispatcher(
        tip_store=store,
        reply_fn=replies.append,
        extractor=lambda t: ["NVDA"],
        crew_invoker=lambda t: True,
    )
    result = d.dispatch(_make_msg("엔비디아"))
    assert result.action == "tip_stored"
    assert "실패" in replies[0] or "❌" in replies[0]


def test_tip_handles_extractor_failure_gracefully(store, replies) -> None:
    def _broken_extractor(text: str) -> list[str]:
        raise RuntimeError("extractor blew up")

    d = TipDispatcher(
        tip_store=store,
        reply_fn=replies.append,
        extractor=_broken_extractor,
        crew_invoker=lambda t: True,
    )
    result = d.dispatch(_make_msg("엔비디아 좋대"))
    # Tip is still persisted (with empty ticker list) despite the
    # extractor failure.
    assert result.action == "tip_stored"
    assert result.extracted_tickers == []
    assert len(store.list_tips()) == 1


# ---------------------------------------------------------------------------
# Formatting helper
# ---------------------------------------------------------------------------


def test_format_tips_list_truncates_long_bodies(store) -> None:
    store.record_tip(
        "A" * 500, detected_tickers=["NVDA"], received_at="2026-04-24T11:00:00"
    )
    tips = store.list_tips()
    out = _format_tips_list(tips, ticker_filter=None)
    # Truncated with ellipsis, well under telegram's 4096 char cap.
    assert "…" in out
    assert len(out) < 500


def test_format_tips_list_header_shows_ticker_filter(store) -> None:
    store.record_tip("NVDA", detected_tickers=["NVDA"])
    out = _format_tips_list(store.list_tips(), ticker_filter="NVDA")
    assert "(NVDA," in out


def test_format_tips_list_no_sender_section_when_missing(store) -> None:
    """Tips without a sender shouldn't have the '(by @None)' string."""
    store.record_tip("NVDA", detected_tickers=["NVDA"])
    out = _format_tips_list(store.list_tips(), ticker_filter=None)
    assert "@None" not in out
    assert "(by " not in out
