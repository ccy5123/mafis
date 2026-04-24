"""Ticker-alias vocabulary helpers.

Provides the canonical map of (ticker → Korean/English aliases) plus
normalization helpers used by the classifier. The actual extraction
work — deciding whether a mention is in an investment context — lives
in `classifier.py`; this module is intentionally small and LLM-free.

Design notes:
  - Alias map seeded inline, user-extensible via
    `data/korean_ticker_aliases.json` (file entries merge with defaults;
    entries override on key conflict).
  - `_normalize_ticker` enforces a shape: 1-5 uppercase ASCII letters
    (US tickers) or 6 digits (KRX stock codes). Reject anything else
    so LLM garbage (1234567, "hello world") never reaches the store.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from wise_investor.config import PROJECT_ROOT


logger = logging.getLogger(__name__)


_DEFAULT_ALIAS_PATH = PROJECT_ROOT / "data" / "korean_ticker_aliases.json"


# Seed list — covers names the user is most likely to forward from a
# Korean stock group chat. Extend via `data/korean_ticker_aliases.json`
# (same shape: {ticker: [alias, alias, ...]}). File entries MERGE with
# these defaults (file takes precedence on key conflict).
_DEFAULT_ALIASES: dict[str, list[str]] = {
    # ----- US equities -----
    "NVDA":  ["nvda", "nvidia", "엔비디아"],
    "AMD":   ["amd", "에이엠디"],
    "TSM":   ["tsm", "tsmc", "티에스엠씨"],
    "AAPL":  ["aapl", "apple", "애플"],
    "MSFT":  ["msft", "microsoft", "마이크로소프트"],
    "GOOGL": ["googl", "goog", "google", "구글", "알파벳"],
    "META":  ["meta", "메타", "페이스북"],
    "TSLA":  ["tsla", "tesla", "테슬라"],
    "AMZN":  ["amzn", "amazon", "아마존"],
    "AVGO":  ["avgo", "broadcom", "브로드컴"],
    "GEV":   ["gev", "지이버노바", "ge버노바"],
    "INTC":  ["intc", "intel", "인텔"],
    "QCOM":  ["qcom", "qualcomm", "퀄컴"],
    "MU":    ["mu", "micron", "마이크론"],
    "ARM":   ["arm", "arm홀딩스"],
    "PLTR":  ["pltr", "팔란티어"],
    "ORCL":  ["orcl", "oracle", "오라클"],
    # ----- KRX (6-digit stock codes) -----
    "005930": ["005930", "삼성전자"],
    "000660": ["000660", "하이닉스", "sk하이닉스"],
    "035420": ["035420", "naver", "네이버"],
    "035720": ["035720", "kakao", "카카오"],
    "005380": ["005380", "현대차", "현대자동차"],
    "051910": ["051910", "lg화학"],
    "373220": ["373220", "lg에너지솔루션", "엘지엔솔"],
    "207940": ["207940", "삼성바이오로직스"],
}


def load_aliases(path: Path | None = None) -> dict[str, list[str]]:
    """Load the ticker alias map, merging file overrides onto the defaults."""
    merged: dict[str, list[str]] = {
        k: list(v) for k, v in _DEFAULT_ALIASES.items()
    }
    override_path = path if path is not None else _DEFAULT_ALIAS_PATH
    if override_path.exists():
        try:
            data = json.loads(override_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for ticker, aliases in data.items():
                    if not isinstance(aliases, list):
                        continue
                    merged[str(ticker).upper()] = [str(a) for a in aliases]
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "Ticker alias override %s unreadable (%s); using defaults.",
                override_path,
                e,
            )
    return merged


def build_inverse_index(aliases: dict[str, list[str]]) -> dict[str, str]:
    """alias_lower → canonical ticker. The ticker itself is always an
    alias of itself so bare-ticker messages ("NVDA 살까?") match.
    """
    out: dict[str, str] = {}
    for ticker, alias_list in aliases.items():
        canonical = ticker.strip().upper()
        for a in [canonical] + list(alias_list or []):
            key = a.strip().lower()
            if key:
                out[key] = canonical
    return out


def _normalize_ticker(raw: str) -> str | None:
    """Accept only sanely-shaped tickers to reject LLM garbage.

    Valid:
      - 1-5 uppercase ASCII letters (US tickers — BRK.B and similar
        dotted classes are rare enough to skip for now).
      - 6 digits (KRX stock codes).
    """
    if not raw:
        return None
    s = raw.strip().upper()
    if 1 <= len(s) <= 5 and s.isalpha() and s.isascii():
        return s
    if len(s) == 6 and s.isdigit():
        return s
    return None


__all__ = [
    "build_inverse_index",
    "load_aliases",
]
