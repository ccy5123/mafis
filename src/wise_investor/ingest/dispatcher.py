"""Dispatch incoming Telegram messages to tip storage or crew actions.

One level above the parser / extractor / store: this is the glue
that takes a `TelegramMessage`, classifies it, and performs the
user-visible side effect — either storing a tip and acknowledging,
or kicking off a crew analysis.

Keeps the mechanics injectable so tests can exercise the full
branch matrix without Telegram / Ollama / subprocess.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from wise_investor.config import PROJECT_ROOT
from wise_investor.ingest.intent_parser import (
    HELP_TEXT_KO,
    UNKNOWN_COMMAND_TEXT_KO,
    Intent,
    parse_intent,
)
from wise_investor.ingest.telegram_receiver import TelegramMessage
from wise_investor.ingest.ticker_extractor import (
    default_llm_fallback,
    extract_tickers,
)
from wise_investor.ingest.tip_store import Tip, TipStore


logger = logging.getLogger(__name__)


# Callable shapes — declared here so tests can inject fakes.
ExtractorFn = Callable[[str], list[str]]
ReplyFn = Callable[[str], None]
CrewInvokerFn = Callable[[str], bool]


@dataclass
class DispatchResult:
    """Side-effect summary, mostly for tests + logging."""

    action: str           # "tip_stored", "crew_spawned", "tips_listed",
                          # "help_sent", "status_sent", "unknown_command",
                          # "no_ticker_extracted", "invalid_args", "noop"
    reply_text: str | None = None
    stored_tip_id: int | None = None
    invoked_ticker: str | None = None
    extracted_tickers: list[str] | None = None


def _default_crew_invoker(ticker: str) -> bool:
    """Spawn `scripts/run_crew.py TICKER` as a detached subprocess.

    Returns True when the process started successfully (not when the
    crew finishes — the crew takes 10-20 min; we fire and forget so
    the bot stays responsive).
    """
    script = PROJECT_ROOT / "scripts" / "run_crew.py"
    if not script.exists():
        logger.error("Crew script not found at %s", script)
        return False
    try:
        subprocess.Popen(
            [sys.executable, str(script), ticker],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
        )
        return True
    except OSError as e:
        logger.warning("Failed to spawn crew for %s: %s", ticker, e)
        return False


class TipDispatcher:
    """Routes an inbound message to the correct handler."""

    def __init__(
        self,
        tip_store: TipStore,
        reply_fn: ReplyFn,
        extractor: ExtractorFn | None = None,
        crew_invoker: CrewInvokerFn | None = None,
    ) -> None:
        self.store = tip_store
        self.reply = reply_fn
        # Default extractor uses the bundled alias map + Qwen fallback.
        # Tests pass a deterministic stub.
        self.extractor = extractor or (
            lambda text: extract_tickers(text, llm_fallback=default_llm_fallback)
        )
        self.crew_invoker = crew_invoker or _default_crew_invoker

    def dispatch(self, msg: TelegramMessage) -> DispatchResult:
        """Classify `msg` and apply its side effect. Always replies
        to the sender (except for `empty` intents).
        """
        intent = parse_intent(msg.text)

        if intent.kind == "empty":
            return DispatchResult(action="noop")

        if intent.kind == "command":
            return self._handle_command(intent, msg)

        # intent.kind == "tip"
        return self._handle_tip(intent, msg)

    # ---- Command handlers -------------------------------------------

    def _handle_command(
        self, intent: Intent, msg: TelegramMessage
    ) -> DispatchResult:
        assert intent.command is not None

        if intent.command in ("help", "start"):
            self.reply(HELP_TEXT_KO)
            return DispatchResult(action="help_sent", reply_text=HELP_TEXT_KO)

        if intent.command == "analyze":
            return self._handle_analyze(intent)

        if intent.command == "tips":
            return self._handle_tips(intent)

        if intent.command == "status":
            # Phase 1 placeholder — we don't track running crews yet.
            text = "📡 /status 는 Phase 2 에서 제공됩니다. 지금은 로그를 확인하세요."
            self.reply(text)
            return DispatchResult(action="status_sent", reply_text=text)

        # Known-list miss — unknown command.
        self.reply(UNKNOWN_COMMAND_TEXT_KO)
        return DispatchResult(
            action="unknown_command",
            reply_text=UNKNOWN_COMMAND_TEXT_KO,
        )

    def _handle_analyze(self, intent: Intent) -> DispatchResult:
        if not intent.args:
            text = "사용법: `/analyze NVDA`\n티커를 함께 입력해 주세요."
            self.reply(text)
            return DispatchResult(action="invalid_args", reply_text=text)

        ticker = intent.args[0].upper()
        ok = self.crew_invoker(ticker)
        if ok:
            text = (
                f"🚀 {ticker} 크루 분석 시작 — 완료까지 10-20분 소요 예상. "
                "결과는 별도 메세지로 도착합니다."
            )
            self.reply(text)
            return DispatchResult(
                action="crew_spawned",
                reply_text=text,
                invoked_ticker=ticker,
            )

        text = f"❌ {ticker} 크루 실행에 실패했습니다. 봇 호스트 로그를 확인해 주세요."
        self.reply(text)
        return DispatchResult(
            action="crew_spawned", reply_text=text, invoked_ticker=ticker
        )

    def _handle_tips(self, intent: Intent) -> DispatchResult:
        ticker = intent.args[0].upper() if intent.args else None
        tips = self.store.list_tips(ticker=ticker, limit=10)
        if not tips:
            text = (
                f"📭 {ticker} 관련 저장된 팁이 없습니다."
                if ticker
                else "📭 저장된 팁이 아직 없습니다."
            )
            self.reply(text)
            return DispatchResult(action="tips_listed", reply_text=text)

        text = _format_tips_list(tips, ticker)
        self.reply(text)
        return DispatchResult(action="tips_listed", reply_text=text)

    # ---- Free-text tip handler --------------------------------------

    def _handle_tip(
        self, intent: Intent, msg: TelegramMessage
    ) -> DispatchResult:
        text = intent.raw_text
        try:
            tickers = self.extractor(text)
        except Exception as e:
            logger.warning("Ticker extraction failed: %s", e)
            tickers = []

        tip: Tip | None = None
        try:
            tip = self.store.record_tip(
                raw_text=text,
                detected_tickers=tickers,
                lang="ko",
                sender=msg.sender_username or msg.sender_first_name,
            )
        except Exception as e:
            logger.warning("Tip store insert failed: %s", e)
            reply = f"❌ 팁 저장에 실패했습니다: {e}"
            self.reply(reply)
            return DispatchResult(action="tip_stored", reply_text=reply)

        if tickers:
            ticker_display = ", ".join(tickers)
            reply = (
                f"✅ 팁 저장됨 #{tip.id}\n"
                f"감지된 종목: {ticker_display}"
            )
        else:
            reply = (
                f"⚠️ 팁 저장됨 #{tip.id}\n"
                "메세지에서 종목을 추출하지 못했습니다. "
                "`data/korean_ticker_aliases.json`에 별칭을 추가해 보세요."
            )

        self.reply(reply)
        return DispatchResult(
            action="tip_stored",
            reply_text=reply,
            stored_tip_id=tip.id,
            extracted_tickers=tickers,
        )


def _format_tips_list(tips: list[Tip], ticker_filter: str | None) -> str:
    """Render up to 10 tips as a compact Telegram message body."""
    header = (
        f"📋 최근 팁 ({ticker_filter}, {len(tips)}개)"
        if ticker_filter
        else f"📋 최근 팁 ({len(tips)}개)"
    )
    lines = [header, ""]
    for i, t in enumerate(tips, start=1):
        sender = f" (by @{t.sender})" if t.sender else ""
        tickers_str = ", ".join(t.detected_tickers) if t.detected_tickers else "미추출"
        # Trim long tip bodies — Telegram caps messages at 4096, and
        # the summary is more readable when tips fit on 2-3 lines.
        body = t.raw_text.replace("\n", " ")
        if len(body) > 140:
            body = body[:137].rstrip() + "…"
        lines.append(
            f"{i}. [{tickers_str}] {t.received_at[:16]} — \"{body}\"{sender}"
        )
    return "\n".join(lines)


__all__ = [
    "DispatchResult",
    "TipDispatcher",
]
