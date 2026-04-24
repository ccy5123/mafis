"""Classify an incoming Telegram message into a command or a tip.

The tip bot is a narrow interface: the user either issues a slash
command (/analyze, /tips, /status, /help, /start) or forwards a
free-text message that we treat as a stock tip.

Why a separate module:
  - Command parsing is deterministic and cheap; keep it out of the
    LLM-adjacent code paths so tests can exercise it without any
    Ollama dependency.
  - Telegram prefixes bot-specific commands with the bot's username
    in group chats ("/analyze@my_bot NVDA") — we normalize that here
    so the dispatcher sees a clean command + args list.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal


IntentKind = Literal["command", "tip", "empty"]


# Canonical command vocabulary. Extending this requires updating the
# dispatcher too; keep the list small.
KNOWN_COMMANDS: frozenset[str] = frozenset(
    {"analyze", "tips", "status", "help", "start"}
)


@dataclass
class Intent:
    """Structured classification of an inbound message."""

    kind: IntentKind
    command: str | None = None         # normalized lowercase, without leading '/'
    args: list[str] = field(default_factory=list)
    raw_text: str = ""
    is_known_command: bool = False     # True only if command is in KNOWN_COMMANDS


_COMMAND_RE = re.compile(
    r"^\s*/(?P<cmd>[A-Za-z][A-Za-z0-9_]*)(?:@[A-Za-z0-9_]+)?"
    r"(?:\s+(?P<rest>.+))?\s*$",
    re.DOTALL,
)


def parse_intent(text: str) -> Intent:
    """Classify `text` as command / tip / empty.

    Empty-or-whitespace text → `Intent(kind="empty")`.
    Starts with '/' → `Intent(kind="command", command=<lower>, args=[...])`.
      Any '@botname' suffix on the command is stripped.
      `is_known_command` is True only for vocabulary in KNOWN_COMMANDS —
      the dispatcher uses this to reply with a "unknown command" hint
      rather than silently ignoring typos.
    Otherwise → `Intent(kind="tip", raw_text=<stripped>)`.
    """
    if text is None or not text.strip():
        return Intent(kind="empty", raw_text=(text or ""))

    m = _COMMAND_RE.match(text)
    if m:
        cmd = m.group("cmd").lower()
        rest = (m.group("rest") or "").strip()
        # Split args on whitespace but preserve quoted spans. For now
        # the vocabulary only takes a single optional TICKER arg, so a
        # simple whitespace split is sufficient.
        args = rest.split() if rest else []
        return Intent(
            kind="command",
            command=cmd,
            args=args,
            raw_text=text.strip(),
            is_known_command=cmd in KNOWN_COMMANDS,
        )

    return Intent(kind="tip", raw_text=text.strip())


# ---------------------------------------------------------------------------
# User-facing help text (Korean by default — bot is Korean-oriented)
# ---------------------------------------------------------------------------


HELP_TEXT_KO = (
    "🤖 *MAFIS 투자 리서치 봇*\n"
    "\n"
    "*명령어*\n"
    "• `/analyze <TICKER>` — 해당 종목 크루 분석 즉시 실행\n"
    "• `/tips` — 최근 저장된 팁 목록\n"
    "• `/tips <TICKER>` — 특정 종목 관련 최근 팁\n"
    "• `/status` — 실행 중 크루 상태\n"
    "• `/help` — 이 메세지\n"
    "\n"
    "*자유 텍스트*\n"
    "명령어 없이 아무 메세지나 보내면 팁으로 저장됨. "
    "메세지에 언급된 종목을 자동 추출 (한국어 이름 지원). "
    "다음 크루 실행 시 Analyst 컨텍스트에 주입.\n"
    "\n"
    "예시:\n"
    "• `엔비디아 실적 좋다던데` → NVDA 팁 저장\n"
    "• `삼성전자 반등할까?` → 005930 팁 저장\n"
    "• `/analyze NVDA` → 크루 실행"
)


UNKNOWN_COMMAND_TEXT_KO = (
    "❓ 알 수 없는 명령어입니다.\n"
    "`/help`로 사용 가능한 명령어를 확인하세요."
)


__all__ = [
    "HELP_TEXT_KO",
    "Intent",
    "IntentKind",
    "KNOWN_COMMANDS",
    "UNKNOWN_COMMAND_TEXT_KO",
    "parse_intent",
]
