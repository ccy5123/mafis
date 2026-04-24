"""Extract stock tickers from free-text tips (Korean / English).

Called by the tip_bot when a user forwards a message from their
stock group chat. The extractor runs two passes:

  1. Static alias map — fast, deterministic, covers the names the
     user cares about. Seeded with ~25 well-known US + KRX names and
     user-extensible via `data/korean_ticker_aliases.json`.

  2. Qwen 2.5 7B fallback (optional) — only invoked when the static
     map returns zero hits AND the caller provides an `llm_fallback`.
     Guards the LLM output to valid ticker shapes (1-5 uppercase
     letters for US, 6 digits for KRX).

Design notes:
  - Hangul aliases are substring-matched (Korean has no spaces between
    most words). ASCII aliases require \\b word boundaries so "amd"
    doesn't match "amdk" or "namd".
  - De-duplicates in first-seen order so the persisted list is stable
    and the downstream crew injection respects the order of mention.
  - Case-insensitive: the map's inverse index is lower-cased once at
    load time.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

from wise_investor.config import PROJECT_ROOT, settings


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


_ASCII_ALIAS_RE = re.compile(r"^[A-Za-z0-9]+$")


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


def extract_tickers(
    text: str,
    alias_index: dict[str, str] | None = None,
    llm_fallback: Callable[[str], list[str]] | None = None,
) -> list[str]:
    """Return tickers mentioned in `text`, de-duplicated, first-seen order.

    Strategy:
      1. Static alias map — substring for Hangul, \\b-bounded regex for
         ASCII. O(|aliases| × |text|), trivial at our scale.
      2. If no hits AND `llm_fallback` is supplied, invoke it; filter
         results to valid ticker shapes.

    Pass `alias_index` to override the default inverse map (used by
    tests to inject a tiny, deterministic mapping).
    """
    if not text or not text.strip():
        return []

    if alias_index is None:
        alias_index = build_inverse_index(load_aliases())

    text_lower = text.lower()
    found: list[str] = []

    # Order matters for first-seen stability: record (first_position,
    # ticker) for every alias that matches, then sort and dedupe.
    mentions: list[tuple[int, str]] = []
    for alias, ticker in alias_index.items():
        position: int | None = None
        if _ASCII_ALIAS_RE.match(alias):
            m = re.search(r"\b" + re.escape(alias) + r"\b", text_lower)
            if m is not None:
                position = m.start()
        else:
            idx = text_lower.find(alias)
            if idx >= 0:
                position = idx
        if position is not None:
            mentions.append((position, ticker))

    mentions.sort(key=lambda p: p[0])
    for _, ticker in mentions:
        if ticker not in found:
            found.append(ticker)

    if not found and llm_fallback is not None:
        try:
            fallback = llm_fallback(text)
        except Exception as e:
            logger.warning("LLM ticker fallback failed: %s", e)
            fallback = []
        for raw in fallback:
            canonical = _normalize_ticker(raw)
            if canonical and canonical not in found:
                found.append(canonical)

    return found


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


def default_llm_fallback(text: str) -> list[str]:
    """Production Ollama call: ask Qwen to extract tickers as JSON array.

    Separated from `_default_llm_call` in the translation package
    because the prompt and parsing are different. Still uses the
    Analyst model at temp=0, seed=42 for determinism.
    """
    import ollama

    system = (
        "You are a stock ticker extractor. Given a Korean or English "
        "message, return a JSON array of stock tickers mentioned. "
        "Use standard US tickers for US stocks (e.g. NVDA, AAPL, TSLA). "
        "Use 6-digit KRX codes for Korean stocks (e.g. 005930 for "
        "Samsung Electronics). If no ticker is mentioned, return []. "
        "Output ONLY the JSON array, nothing else — no explanation, "
        "no preamble, no markdown fences."
    )

    fewshot = [
        {"role": "user", "content": "엔비디아 실적 좋대"},
        {"role": "assistant", "content": '["NVDA"]'},
        {"role": "user", "content": "TSMC랑 AMD 둘 다 오르는데"},
        {"role": "assistant", "content": '["TSM","AMD"]'},
        {"role": "user", "content": "삼성전자 반등할까?"},
        {"role": "assistant", "content": '["005930"]'},
        {"role": "user", "content": "오늘 날씨 좋네"},
        {"role": "assistant", "content": "[]"},
    ]

    resp = ollama.chat(
        model=settings.analyst_model,
        messages=[
            {"role": "system", "content": system},
            *fewshot,
            {"role": "user", "content": text},
        ],
        options={
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
    )
    raw = (resp["message"]["content"] or "").strip()

    # Strip markdown code fences if the LLM sneaks them in.
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except json.JSONDecodeError:
        logger.warning("LLM fallback returned non-JSON: %r", raw[:80])
    return []


__all__ = [
    "build_inverse_index",
    "default_llm_fallback",
    "extract_tickers",
    "load_aliases",
]
