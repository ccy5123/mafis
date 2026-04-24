"""Tests for the tip-bot intent parser."""

from __future__ import annotations

from wise_investor.ingest.intent_parser import (
    HELP_TEXT_KO,
    KNOWN_COMMANDS,
    parse_intent,
)


# ---------------------------------------------------------------------------
# Empty / whitespace
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty_intent() -> None:
    out = parse_intent("")
    assert out.kind == "empty"


def test_whitespace_returns_empty_intent() -> None:
    assert parse_intent("   ").kind == "empty"
    assert parse_intent("\n\n").kind == "empty"


def test_none_text_returns_empty_intent() -> None:
    out = parse_intent(None)  # type: ignore[arg-type]
    assert out.kind == "empty"


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def test_bare_help_command() -> None:
    out = parse_intent("/help")
    assert out.kind == "command"
    assert out.command == "help"
    assert out.args == []
    assert out.is_known_command is True


def test_analyze_with_ticker() -> None:
    out = parse_intent("/analyze NVDA")
    assert out.kind == "command"
    assert out.command == "analyze"
    assert out.args == ["NVDA"]
    assert out.is_known_command is True


def test_analyze_case_insensitive_command_name() -> None:
    """Users sometimes send '/Analyze' or '/ANALYZE' — normalize."""
    out = parse_intent("/ANALYZE NVDA")
    assert out.command == "analyze"
    assert out.args == ["NVDA"]


def test_tips_with_ticker_filter() -> None:
    out = parse_intent("/tips NVDA")
    assert out.command == "tips"
    assert out.args == ["NVDA"]


def test_tips_without_args() -> None:
    out = parse_intent("/tips")
    assert out.command == "tips"
    assert out.args == []


def test_status_command() -> None:
    out = parse_intent("/status")
    assert out.command == "status"
    assert out.args == []


def test_start_is_recognized_as_known_command() -> None:
    """Telegram's UI sends /start on first contact — must be handled."""
    out = parse_intent("/start")
    assert out.command == "start"
    assert out.is_known_command is True


def test_command_with_bot_username_suffix() -> None:
    """In group chats, Telegram appends @botname to commands."""
    out = parse_intent("/analyze@MAFIS_bot NVDA")
    assert out.kind == "command"
    assert out.command == "analyze"
    assert out.args == ["NVDA"]


def test_command_with_multiple_args() -> None:
    out = parse_intent("/analyze NVDA ignore_this")
    assert out.command == "analyze"
    assert out.args == ["NVDA", "ignore_this"]


def test_command_with_extra_whitespace() -> None:
    out = parse_intent("   /analyze    NVDA   ")
    assert out.command == "analyze"
    assert out.args == ["NVDA"]


def test_unknown_command_still_classified_as_command() -> None:
    """A slash prefix means it's a command attempt, even if the
    specific name is unknown — the dispatcher decides what to do
    (reply with help hint).
    """
    out = parse_intent("/xyz NVDA")
    assert out.kind == "command"
    assert out.command == "xyz"
    assert out.is_known_command is False


# ---------------------------------------------------------------------------
# Free-text tips
# ---------------------------------------------------------------------------


def test_korean_free_text_is_tip() -> None:
    out = parse_intent("엔비디아 실적 좋다던데")
    assert out.kind == "tip"
    assert out.raw_text == "엔비디아 실적 좋다던데"


def test_english_free_text_is_tip() -> None:
    out = parse_intent("NVDA looks strong this quarter")
    assert out.kind == "tip"
    assert out.raw_text == "NVDA looks strong this quarter"


def test_mid_sentence_slash_is_not_a_command() -> None:
    """A slash in the middle of a sentence is not a command."""
    out = parse_intent("NVDA 랑 AMD/TSM 둘 다 궁금")
    assert out.kind == "tip"


def test_multiline_tip_preserves_linebreaks_in_raw_text() -> None:
    msg = "NVDA 실적 좋다\n\nTSMC 주문도 늘었다"
    out = parse_intent(msg)
    assert out.kind == "tip"
    assert "\n" in out.raw_text


def test_leading_whitespace_tip_is_stripped() -> None:
    out = parse_intent("   엔비디아 좋대   ")
    assert out.kind == "tip"
    assert out.raw_text == "엔비디아 좋대"


# ---------------------------------------------------------------------------
# Known-command vocabulary
# ---------------------------------------------------------------------------


def test_known_commands_includes_all_documented_verbs() -> None:
    assert "analyze" in KNOWN_COMMANDS
    assert "tips" in KNOWN_COMMANDS
    assert "status" in KNOWN_COMMANDS
    assert "help" in KNOWN_COMMANDS
    assert "start" in KNOWN_COMMANDS


def test_help_text_lists_each_user_facing_command() -> None:
    """If we add a user-facing command to KNOWN_COMMANDS we must also
    mention it in the help message — otherwise users won't discover
    it. /start is a Telegram convention (sent on first bot contact),
    aliased to /help by the dispatcher, so we don't re-advertise it.
    """
    internal = {"start"}
    for cmd in KNOWN_COMMANDS - internal:
        assert f"/{cmd}" in HELP_TEXT_KO, f"{cmd} missing from HELP_TEXT_KO"
