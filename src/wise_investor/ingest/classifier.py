"""LLM-first classification of incoming tip messages.

Replaces the pure-keyword ticker extractor. User messages are short
natural-language snippets whose investment relevance depends on
context, not just presence of a known name:

  "애플 파이 레시피 공유"   → none    (not about AAPL)
  "메타버스 안 됐네"         → none    (generic tech term, not META)
  "메타, 메타버스 사업 접음"  → ticker  (META corporate action)
  "연준 금리 동결 예상"       → macro   (not about a ticker)
  "중국 경기 둔화"            → geopolitics
  "유가 100달러"              → commodity

The classifier returns a `TipClassification(category, tickers, topics)`
which the dispatcher persists and the Phase 2 injectors use to route
tips to the right agent (ticker → Analyst, macro/fx/commodity/geo →
Economist, sector → Analyst for sector-relevant symbols).

LLM contract:
  - System prompt + few-shot exchanges in the target language.
  - Output JSON object only; shape validated after parse.
  - Sampling follows the active backend's resolved `agents.classifier`
    config (model-family recommendation by default; users can pin
    deterministic mode in agent_models.yaml).
  - On any parse/LLM failure we return category='unknown' so the
    store still persists the tip for human review (no data loss).

Degraded mode:
  - If no `llm_call` is supplied (tests, Ollama down), falls back
    to pure alias-map matching with category='ticker' if any match,
    else 'unknown'. This mode cannot distinguish investment context
    from casual mention — it's a last resort.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Callable

from wise_investor.ingest.ticker_extractor import (
    _normalize_ticker,
    build_inverse_index,
    load_aliases,
)
from wise_investor.ingest.tip_store import CATEGORIES


logger = logging.getLogger(__name__)


@dataclass
class TipClassification:
    """Structured output of `classify_tip`."""

    category: str                     # one of CATEGORIES
    tickers: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)


# Categories the LLM is allowed to assign. `unknown` is reserved for
# classifier failures; the LLM should never return it.
_LLM_CATEGORIES: frozenset[str] = frozenset(
    {"ticker", "macro", "fx", "sector", "geopolitics", "commodity", "none"}
)


def _build_alias_hint(alias_map: dict[str, list[str]], max_entries: int = 40) -> str:
    """Render the alias map as a bullet list for the LLM prompt.

    Kept short so it doesn't blow out context — the LLM only needs
    the vocabulary, not every alias variant.
    """
    lines: list[str] = []
    for i, (ticker, aliases) in enumerate(alias_map.items()):
        if i >= max_entries:
            break
        short_aliases = ", ".join(aliases[:5]) if aliases else ticker.lower()
        lines.append(f"- {ticker}: {short_aliases}")
    return "\n".join(lines)


def _build_system_prompt(alias_map: dict[str, list[str]]) -> str:
    hint = _build_alias_hint(alias_map)
    return (
        "You classify short Korean/English messages for an investment "
        "research bot. For each message, decide the investment CATEGORY "
        "and extract relevant tickers or topic slugs.\n"
        "\n"
        "Categories (choose exactly one):\n"
        "- ticker: specific publicly listed stock discussed in an "
        "investment context (price, earnings, business, corporate action).\n"
        "- macro: monetary policy, inflation, GDP, employment, the Fed, "
        "the Bank of Korea, recessions.\n"
        "- fx: currency / FX moves.\n"
        "- sector: an industry sector as a whole (semiconductor, biotech, "
        "energy, defense) without a specific ticker focus.\n"
        "- geopolitics: war, sanctions, trade tensions, country-level risk.\n"
        "- commodity: oil, gold, copper, natural gas, wheat, etc.\n"
        "- none: the message is not about investments at all "
        "(food, weather, metaphorical tech terms like 'metaverse' used "
        "generically, casual product references, service instructions).\n"
        "\n"
        "Ticker vocabulary (Korean/English aliases → canonical ticker):\n"
        f"{hint}\n"
        "Also accept tickers not in this list if they clearly refer to "
        "a real stock (1-5 uppercase letters for US, 6 digits for KRX).\n"
        "\n"
        "Output contract: ONLY a JSON object, no prose, no markdown fences. "
        "Shape: {\"category\": \"...\", \"tickers\": [\"...\"], "
        "\"topics\": [\"...\"]}.\n"
        "- For category=ticker: populate `tickers` with canonical "
        "symbols, leave `topics` empty.\n"
        "- For macro/fx/sector/geopolitics/commodity: leave `tickers` "
        "empty, populate `topics` with 1-3 English lowercase slugs "
        "(interest_rates, fed, cpi, inflation, employment, gdp, "
        "recession, krw_usd, jpy, cny, semiconductor, ai, biotech, "
        "energy, auto, defense, china, taiwan, russia_ukraine, "
        "middle_east, sanctions, trade_war, oil, gold, copper, "
        "natural_gas — or a concise new slug if none fit).\n"
        "- For none: both arrays empty.\n"
        "\n"
        "Anti-false-positive rules:\n"
        "- Casual product mentions (\"애플 파이\", \"메타버스 게임 해봤어\") → none.\n"
        "- Generic technology terms (\"메타버스 안 됐네\" without company "
        "context) → none.\n"
        "- Service / UI instructions (\"네이버 검색해봐\", \"카카오톡 봐\") → none.\n"
        "- Company names in proper investment context (earnings, price, "
        "business change, corporate action) → ticker."
    )


_FEWSHOTS: list[tuple[str, str]] = [
    ("엔비디아 실적 좋다", '{"category":"ticker","tickers":["NVDA"],"topics":[]}'),
    ("TSMC랑 AMD 둘 다 오를 듯", '{"category":"ticker","tickers":["TSM","AMD"],"topics":[]}'),
    ("삼성전자 반등할까?", '{"category":"ticker","tickers":["005930"],"topics":[]}'),
    ("메타가 메타버스 사업 접는다", '{"category":"ticker","tickers":["META"],"topics":[]}'),
    ("연준 다음주 금리 동결 예상", '{"category":"macro","tickers":[],"topics":["interest_rates","fed"]}'),
    ("CPI 3.2% 나왔네", '{"category":"macro","tickers":[],"topics":["inflation","cpi"]}'),
    ("환율 1500원 돌파할 수도", '{"category":"fx","tickers":[],"topics":["krw_usd"]}'),
    ("반도체 사이클 반등 중", '{"category":"sector","tickers":[],"topics":["semiconductor"]}'),
    ("중국 경기 둔화 심각", '{"category":"geopolitics","tickers":[],"topics":["china"]}'),
    ("유가 배럴당 100달러 돌파", '{"category":"commodity","tickers":[],"topics":["oil"]}'),
    ("애플 파이 맛있네", '{"category":"none","tickers":[],"topics":[]}'),
    ("메타버스 안 됐네", '{"category":"none","tickers":[],"topics":[]}'),
    ("네이버 검색해봐", '{"category":"none","tickers":[],"topics":[]}'),
    ("오늘 비 많이 온다", '{"category":"none","tickers":[],"topics":[]}'),
]


def classify_tip(
    text: str,
    llm_call: Callable[[str, str, list[tuple[str, str]]], str] | None = None,
    alias_map: dict[str, list[str]] | None = None,
) -> TipClassification:
    """Classify `text` into a TipClassification.

    `llm_call(system, user, fewshots) -> str` is injectable so tests
    run without Ollama. Production path uses `default_llm_call`
    which ships the few-shot transcript as chat-history messages.

    Degraded mode (`llm_call is None`): pure alias-map match. If any
    ticker is found, category='ticker'; else category='unknown' so
    the tip persists for manual review.
    """
    if not text or not text.strip():
        return TipClassification(category="none")

    if alias_map is None:
        alias_map = load_aliases()

    if llm_call is None:
        return _degraded_keyword_classify(text, alias_map)

    system = _build_system_prompt(alias_map)
    try:
        raw = llm_call(system, text.strip(), _FEWSHOTS)
    except Exception as e:
        logger.warning("classify_tip LLM call failed: %s", e)
        return TipClassification(category="unknown")

    parsed = _parse_classification(raw)
    if parsed is None:
        logger.warning(
            "classify_tip could not parse LLM output: %r", (raw or "")[:120]
        )
        return TipClassification(category="unknown")
    return parsed


def _parse_classification(raw: str) -> TipClassification | None:
    """Parse a JSON object response into TipClassification.

    Tolerates markdown code fences (```json ... ```). Rejects output
    that doesn't match the contract (unknown category, wrong types).
    """
    if not raw:
        return None
    cleaned = raw.strip()

    # Strip markdown fences if they leak through.
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) >= 2:
            cleaned = parts[1].strip()
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    category = obj.get("category")
    if not isinstance(category, str):
        return None
    category_norm = category.strip().lower()
    if category_norm not in _LLM_CATEGORIES:
        return None

    raw_tickers = obj.get("tickers") or []
    tickers: list[str] = []
    if isinstance(raw_tickers, list):
        for t in raw_tickers:
            norm = _normalize_ticker(str(t)) if t is not None else None
            if norm and norm not in tickers:
                tickers.append(norm)

    raw_topics = obj.get("topics") or []
    topics: list[str] = []
    if isinstance(raw_topics, list):
        for t in raw_topics:
            if not isinstance(t, str):
                continue
            slug = t.strip().lower()
            # Topic slugs are English snake_case; reject anything with
            # whitespace or non-ASCII so downstream code can treat them
            # as opaque keys.
            if slug and slug.isascii() and " " not in slug and slug not in topics:
                topics.append(slug)

    # Consistency: ticker category requires at least one ticker;
    # non-ticker categories ignore tickers.
    if category_norm == "ticker" and not tickers:
        # LLM said ticker but returned no valid ones — fall back to
        # 'unknown' so we don't silently misclassify.
        return TipClassification(category="unknown")
    if category_norm != "ticker":
        tickers = []
    if category_norm in ("ticker", "none"):
        topics = []

    return TipClassification(
        category=category_norm,
        tickers=tickers,
        topics=topics,
    )


def _degraded_keyword_classify(
    text: str, alias_map: dict[str, list[str]]
) -> TipClassification:
    """No-LLM fallback: alias-map match only.

    Cannot distinguish investment context from casual mentions, but
    keeps the bot usable when Ollama is down. Returns 'ticker' on
    any match, 'unknown' otherwise — NEVER 'none' (we don't want to
    silently drop tips that might be investment-relevant).
    """
    import re

    index = build_inverse_index(alias_map)
    text_lower = text.lower()
    tickers: list[str] = []
    mentions: list[tuple[int, str]] = []
    _ascii_re = re.compile(r"^[A-Za-z0-9]+$")
    for alias, ticker in index.items():
        if _ascii_re.match(alias):
            m = re.search(r"\b" + re.escape(alias) + r"\b", text_lower)
            if m is not None:
                mentions.append((m.start(), ticker))
        else:
            idx = text_lower.find(alias)
            if idx >= 0:
                mentions.append((idx, ticker))
    mentions.sort(key=lambda p: p[0])
    for _, ticker in mentions:
        if ticker not in tickers:
            tickers.append(ticker)

    if tickers:
        return TipClassification(category="ticker", tickers=tickers)
    return TipClassification(category="unknown")


# ---------------------------------------------------------------------------
# Production LLM call (Ollama Qwen 2.5 7B)
# ---------------------------------------------------------------------------


def default_llm_call(
    system: str, user: str, fewshots: list[tuple[str, str]]
) -> str:
    """Production classifier call. Ships few-shots as chat history so
    Qwen sees a consistent JSON output contract across turns. Model
    + sampling come from `agents.classifier` in agent_models.yaml,
    falling back to the Analyst entry when unspecified.
    """
    from wise_investor.llm import get_agent_config, get_backend

    backend = get_backend()
    cfg = get_agent_config("classifier", backend=backend.name)

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for u, a in fewshots:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})

    response = backend.chat(
        messages=messages,
        model=cfg.model,
        sampling=cfg.sampling,
    )
    return response.content


__all__ = [
    "TipClassification",
    "classify_tip",
    "default_llm_call",
]
