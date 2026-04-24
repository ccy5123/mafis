"""Tests for the LLM-first tip classifier."""

from __future__ import annotations

import json

import pytest

from wise_investor.ingest.classifier import (
    TipClassification,
    _build_alias_hint,
    _parse_classification,
    classify_tip,
)


_TINY_MAP = {
    "NVDA": ["nvda", "nvidia", "엔비디아"],
    "TSM": ["tsm", "tsmc"],
    "AMD": ["amd", "에이엠디"],
    "META": ["meta", "메타"],
    "005930": ["005930", "삼성전자"],
}


# ---------------------------------------------------------------------------
# Empty / trivial
# ---------------------------------------------------------------------------


def test_empty_returns_none_category() -> None:
    def _stub(system, user, fewshots):
        raise AssertionError("LLM must not be called on empty text")

    assert classify_tip("", llm_call=_stub).category == "none"
    assert classify_tip("   ", llm_call=_stub).category == "none"


# ---------------------------------------------------------------------------
# LLM-first path with stubbed output
# ---------------------------------------------------------------------------


def _llm_returning(payload: str):
    def _fn(system, user, fewshots):
        return payload
    return _fn


def test_ticker_category_roundtrip() -> None:
    llm = _llm_returning('{"category":"ticker","tickers":["NVDA"],"topics":[]}')
    out = classify_tip("엔비디아 실적", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "ticker"
    assert out.tickers == ["NVDA"]
    assert out.topics == []


def test_macro_category_roundtrip() -> None:
    llm = _llm_returning(
        '{"category":"macro","tickers":[],"topics":["interest_rates","fed"]}'
    )
    out = classify_tip("연준 금리 동결", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "macro"
    assert out.topics == ["interest_rates", "fed"]
    assert out.tickers == []


def test_fx_category_roundtrip() -> None:
    llm = _llm_returning('{"category":"fx","tickers":[],"topics":["krw_usd"]}')
    out = classify_tip("환율 1500", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "fx"
    assert out.topics == ["krw_usd"]


def test_sector_category_roundtrip() -> None:
    llm = _llm_returning(
        '{"category":"sector","tickers":[],"topics":["semiconductor"]}'
    )
    out = classify_tip("반도체 사이클", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "sector"
    assert out.topics == ["semiconductor"]


def test_geopolitics_category_roundtrip() -> None:
    llm = _llm_returning(
        '{"category":"geopolitics","tickers":[],"topics":["china"]}'
    )
    out = classify_tip("중국 경기", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "geopolitics"
    assert out.topics == ["china"]


def test_commodity_category_roundtrip() -> None:
    llm = _llm_returning('{"category":"commodity","tickers":[],"topics":["oil"]}')
    out = classify_tip("유가 급등", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "commodity"
    assert out.topics == ["oil"]


def test_none_category_for_non_investment_text() -> None:
    llm = _llm_returning('{"category":"none","tickers":[],"topics":[]}')
    out = classify_tip("애플 파이 맛있네", llm_call=llm, alias_map=_TINY_MAP)
    assert out.category == "none"
    assert out.tickers == []
    assert out.topics == []


# ---------------------------------------------------------------------------
# Malformed / adversarial LLM output
# ---------------------------------------------------------------------------


def test_non_json_output_yields_unknown() -> None:
    out = classify_tip(
        "엔비디아", llm_call=_llm_returning("this is not json at all")
    )
    assert out.category == "unknown"


def test_markdown_fences_are_stripped() -> None:
    out = classify_tip(
        "엔비디아",
        llm_call=_llm_returning(
            '```json\n{"category":"ticker","tickers":["NVDA"],"topics":[]}\n```'
        ),
        alias_map=_TINY_MAP,
    )
    assert out.category == "ticker"
    assert out.tickers == ["NVDA"]


def test_ticker_category_without_tickers_falls_back_to_unknown() -> None:
    """LLM said 'ticker' but returned no tickers → reject. Otherwise
    we'd persist as 'ticker' and Phase 2 injection would blow up
    trying to find a symbol match.
    """
    out = classify_tip(
        "엔비디아",
        llm_call=_llm_returning('{"category":"ticker","tickers":[],"topics":[]}'),
    )
    assert out.category == "unknown"


def test_non_ticker_category_clears_stray_tickers() -> None:
    """LLM should leave tickers empty for macro, but if it doesn't we
    drop them on normalization.
    """
    out = classify_tip(
        "금리",
        llm_call=_llm_returning(
            '{"category":"macro","tickers":["NVDA"],"topics":["interest_rates"]}'
        ),
        alias_map=_TINY_MAP,
    )
    assert out.category == "macro"
    assert out.tickers == []
    assert out.topics == ["interest_rates"]


def test_ticker_normalization_rejects_garbage() -> None:
    """LLM returned junk tickers alongside good ones — the shape
    validator drops the junk.
    """
    out = classify_tip(
        "섞인 티커",
        llm_call=_llm_returning(
            '{"category":"ticker","tickers":["NVDA","hello world","TSM","123"],'
            '"topics":[]}'
        ),
        alias_map=_TINY_MAP,
    )
    assert out.category == "ticker"
    assert out.tickers == ["NVDA", "TSM"]


def test_topic_slugs_must_be_ascii_snake_case() -> None:
    """Topics with whitespace or non-ASCII are dropped (they'd be
    awful index keys for Phase 2).
    """
    out = classify_tip(
        "금리",
        llm_call=_llm_returning(
            '{"category":"macro","tickers":[],'
            '"topics":["interest_rates","한국어","has space","fed"]}'
        ),
    )
    assert out.category == "macro"
    assert out.topics == ["interest_rates", "fed"]


def test_unknown_category_string_rejected() -> None:
    out = classify_tip(
        "text",
        llm_call=_llm_returning(
            '{"category":"frobnicate","tickers":[],"topics":[]}'
        ),
    )
    assert out.category == "unknown"


def test_wrong_shape_rejected() -> None:
    # Top-level array instead of object.
    out = classify_tip(
        "text",
        llm_call=_llm_returning('["NVDA"]'),
    )
    assert out.category == "unknown"


def test_llm_exception_is_swallowed() -> None:
    def _flaky(system, user, fewshots):
        raise RuntimeError("mock Ollama down")

    out = classify_tip("엔비디아", llm_call=_flaky)
    assert out.category == "unknown"


def test_llm_empty_output_yields_unknown() -> None:
    out = classify_tip("엔비디아", llm_call=_llm_returning(""))
    assert out.category == "unknown"


# ---------------------------------------------------------------------------
# Degraded mode (no llm_call) — keyword fallback
# ---------------------------------------------------------------------------


def test_degraded_mode_matches_alias_to_ticker() -> None:
    out = classify_tip("엔비디아 실적", alias_map=_TINY_MAP)  # no llm_call
    assert out.category == "ticker"
    assert out.tickers == ["NVDA"]


def test_degraded_mode_multiple_tickers_first_seen_order() -> None:
    out = classify_tip("TSMC 랑 NVDA 둘 다 오를듯", alias_map=_TINY_MAP)
    assert out.category == "ticker"
    assert out.tickers == ["TSM", "NVDA"]


def test_degraded_mode_no_match_yields_unknown_not_none() -> None:
    """When no LLM and no ticker match, we say 'unknown' rather than
    'none' — without context we can't be sure it's irrelevant.
    """
    out = classify_tip("오늘 날씨 좋네", alias_map=_TINY_MAP)
    assert out.category == "unknown"


def test_degraded_mode_ascii_word_boundary() -> None:
    """Even in degraded mode we avoid 'amd' matching 'namdong'."""
    out = classify_tip("namdongnet 어쩌고", alias_map=_TINY_MAP)
    assert out.category == "unknown"


# ---------------------------------------------------------------------------
# Prompt construction smoke tests
# ---------------------------------------------------------------------------


def test_alias_hint_includes_known_tickers() -> None:
    hint = _build_alias_hint(_TINY_MAP)
    assert "NVDA:" in hint
    assert "엔비디아" in hint
    assert "005930:" in hint


def test_alias_hint_caps_entries(tmp_path) -> None:
    big = {f"X{i:04d}": [f"alias{i}"] for i in range(100)}
    hint = _build_alias_hint(big, max_entries=5)
    # Only 5 lines rendered despite 100 entries.
    assert hint.count("\n") == 4


# ---------------------------------------------------------------------------
# _parse_classification directly
# ---------------------------------------------------------------------------


def test_parse_classification_rejects_non_object() -> None:
    assert _parse_classification("null") is None
    assert _parse_classification("123") is None
    assert _parse_classification('"string"') is None


def test_parse_classification_handles_extra_keys() -> None:
    """Extra keys in the JSON shouldn't fail parsing — we ignore them."""
    out = _parse_classification(
        json.dumps(
            {
                "category": "macro",
                "tickers": [],
                "topics": ["fed"],
                "rationale": "because",
                "confidence": 0.9,
            }
        )
    )
    assert out is not None
    assert out.category == "macro"
    assert out.topics == ["fed"]