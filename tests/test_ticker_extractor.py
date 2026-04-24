"""Tests for the ticker extractor used by the tip bot."""

from __future__ import annotations

import json

import pytest

from wise_investor.ingest.ticker_extractor import (
    _normalize_ticker,
    build_inverse_index,
    extract_tickers,
    load_aliases,
)


# Deterministic minimal alias map for most tests — keeps test output
# independent of the default seed list evolving.
_TINY_MAP = {
    "NVDA": ["nvda", "nvidia", "엔비디아"],
    "TSM": ["tsm", "tsmc", "티에스엠씨"],
    "AMD": ["amd", "에이엠디"],
    "005930": ["005930", "삼성전자"],
}


@pytest.fixture
def index() -> dict[str, str]:
    return build_inverse_index(_TINY_MAP)


# ---------------------------------------------------------------------------
# build_inverse_index
# ---------------------------------------------------------------------------


def test_build_inverse_index_lowercases_aliases() -> None:
    idx = build_inverse_index({"NVDA": ["NVDA", "NVIDIA", "엔비디아"]})
    assert idx["nvda"] == "NVDA"
    assert idx["nvidia"] == "NVDA"
    assert idx["엔비디아"] == "NVDA"


def test_build_inverse_index_ticker_is_its_own_alias() -> None:
    """Bare tickers like 'NVDA' should match even if the alias list
    doesn't explicitly repeat the ticker.
    """
    idx = build_inverse_index({"NVDA": []})
    assert idx["nvda"] == "NVDA"


def test_build_inverse_index_skips_blank_aliases() -> None:
    idx = build_inverse_index({"NVDA": ["", "  ", "nvda"]})
    assert "" not in idx
    assert "  " not in idx
    assert idx["nvda"] == "NVDA"


# ---------------------------------------------------------------------------
# Static-map extraction — Korean & English paths
# ---------------------------------------------------------------------------


def test_extract_bare_us_ticker(index) -> None:
    assert extract_tickers("요즘 NVDA 어때?", alias_index=index) == ["NVDA"]


def test_extract_english_company_name(index) -> None:
    assert extract_tickers("nvidia 실적 좋대", alias_index=index) == ["NVDA"]


def test_extract_korean_alias(index) -> None:
    assert extract_tickers("엔비디아 살까?", alias_index=index) == ["NVDA"]


def test_extract_krx_korean_name(index) -> None:
    assert extract_tickers("삼성전자 반등할까", alias_index=index) == ["005930"]


def test_extract_krx_numeric_code(index) -> None:
    assert extract_tickers("005930 주가 움직이네", alias_index=index) == ["005930"]


def test_extract_multiple_tickers_preserves_order(index) -> None:
    text = "TSMC 랑 NVDA 둘 다 오르네"
    assert extract_tickers(text, alias_index=index) == ["TSM", "NVDA"]


def test_extract_returns_no_duplicates(index) -> None:
    text = "엔비디아가 오르면 NVDA 주가도 당연히 오르고 nvidia 다들 산대"
    assert extract_tickers(text, alias_index=index) == ["NVDA"]


def test_extract_is_case_insensitive(index) -> None:
    assert extract_tickers("Nvda 좋대", alias_index=index) == ["NVDA"]
    assert extract_tickers("nVdA 좋대", alias_index=index) == ["NVDA"]


def test_extract_empty_text_returns_empty(index) -> None:
    assert extract_tickers("", alias_index=index) == []
    assert extract_tickers("   \n  ", alias_index=index) == []


def test_extract_no_tickers_returns_empty(index) -> None:
    assert extract_tickers("오늘 날씨 좋다", alias_index=index) == []


# ---------------------------------------------------------------------------
# ASCII word-boundary — avoid false positives
# ---------------------------------------------------------------------------


def test_ascii_alias_requires_word_boundary(index) -> None:
    """'amd' should NOT match 'namdoe' or 'amdongnet' — word-bounded
    regex prevents the substring false-positive that a naive search
    would hit.
    """
    assert extract_tickers("namdong은 별개지", alias_index=index) == []
    assert extract_tickers("amdongnet 어쩌고", alias_index=index) == []
    # But a legitimate mention does match.
    assert extract_tickers("AMD 올랐어", alias_index=index) == ["AMD"]


def test_hangul_alias_substring_match_is_fine(index) -> None:
    """Korean text has no spaces between words, so substring
    matching is appropriate for Hangul aliases. '엔비디아' should
    match inside '엔비디아주식도'.
    """
    assert extract_tickers("엔비디아주식도 사자", alias_index=index) == ["NVDA"]


# ---------------------------------------------------------------------------
# LLM fallback path
# ---------------------------------------------------------------------------


def test_llm_fallback_invoked_only_when_static_map_misses(index) -> None:
    called: list[str] = []

    def _stub(text: str) -> list[str]:
        called.append(text)
        return ["NVDA"]

    # Static map already resolves this — LLM must NOT be called.
    out = extract_tickers("NVDA 좋다", alias_index=index, llm_fallback=_stub)
    assert out == ["NVDA"]
    assert called == []


def test_llm_fallback_invoked_when_static_map_empty(index) -> None:
    """Unknown Korean name triggers the LLM fallback."""
    called: list[str] = []

    def _stub(text: str) -> list[str]:
        called.append(text)
        return ["MSFT"]

    out = extract_tickers(
        "마이크로소프트 뉴스", alias_index=index, llm_fallback=_stub
    )
    assert out == ["MSFT"]
    assert len(called) == 1


def test_llm_fallback_output_is_normalized(index) -> None:
    """LLM might return lowercase or weird shapes — normalizer filters."""
    def _stub(text: str) -> list[str]:
        # Mix of valid + garbage — garbage must be dropped.
        return ["nvda", "hello world", "123", "005930", "THISIS2LONG"]

    out = extract_tickers("unknown", alias_index=index, llm_fallback=_stub)
    assert "NVDA" in out
    assert "005930" in out
    assert "HELLO WORLD" not in out
    assert "123" not in out
    assert "THISIS2LONG" not in out


def test_llm_fallback_error_is_swallowed(index) -> None:
    def _flaky(text: str) -> list[str]:
        raise RuntimeError("mock Ollama outage")

    out = extract_tickers("unknown name", alias_index=index, llm_fallback=_flaky)
    assert out == []


def test_normalize_ticker_shapes() -> None:
    assert _normalize_ticker("NVDA") == "NVDA"
    assert _normalize_ticker("nvda") == "NVDA"
    assert _normalize_ticker("005930") == "005930"
    assert _normalize_ticker("AAPL") == "AAPL"
    # Rejections:
    assert _normalize_ticker("") is None
    assert _normalize_ticker("TOO LONG TICKER") is None
    assert _normalize_ticker("12345") is None      # 5 digits, not 6
    assert _normalize_ticker("1234567") is None    # 7 digits
    assert _normalize_ticker("NVDA1") is None      # letters + digits
    assert _normalize_ticker("한글티커") is None   # non-ASCII


# ---------------------------------------------------------------------------
# load_aliases — file override
# ---------------------------------------------------------------------------


def test_load_aliases_uses_defaults_when_file_missing(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.json"
    aliases = load_aliases(path=missing)
    # Should contain at least the seed list.
    assert "NVDA" in aliases
    assert "005930" in aliases


def test_load_aliases_merges_overrides_from_file(tmp_path) -> None:
    """User-supplied overrides REPLACE the default aliases for that
    ticker (and add new tickers). Existing tickers not in the
    override file keep their defaults.
    """
    override = tmp_path / "aliases.json"
    override.write_text(
        json.dumps(
            {
                "NVDA": ["custom_nvda_alias"],       # replaces default
                "XYZ":  ["xyz_alias", "xyz_corp"],   # brand new entry
            }
        ),
        encoding="utf-8",
    )
    aliases = load_aliases(path=override)
    # Override takes effect for NVDA.
    assert aliases["NVDA"] == ["custom_nvda_alias"]
    # New ticker added.
    assert aliases["XYZ"] == ["xyz_alias", "xyz_corp"]
    # Unrelated default ticker untouched.
    assert "TSM" in aliases


def test_load_aliases_survives_malformed_json(tmp_path) -> None:
    """A broken override file should NOT crash the bot — fall back
    to defaults with a warning log.
    """
    override = tmp_path / "aliases.json"
    override.write_text("{this is not valid json", encoding="utf-8")
    aliases = load_aliases(path=override)
    # Defaults still present.
    assert "NVDA" in aliases


def test_load_aliases_ignores_non_list_values(tmp_path) -> None:
    """A user might accidentally write a string instead of a list.
    The loader must not crash — it ignores the bad entry and keeps
    the default for that ticker.
    """
    override = tmp_path / "aliases.json"
    override.write_text(
        json.dumps({"NVDA": "not a list"}),
        encoding="utf-8",
    )
    aliases = load_aliases(path=override)
    assert "NVDA" in aliases
    # Default value retained since override was malformed.
    assert isinstance(aliases["NVDA"], list)


# ---------------------------------------------------------------------------
# Integration with default (seed) aliases
# ---------------------------------------------------------------------------


def test_default_seeds_cover_common_korean_names() -> None:
    """Smoke test the seed list — if these break, users will file
    bugs the moment they start forwarding tips.
    """
    out = extract_tickers("엔비디아 좋대")
    assert "NVDA" in out
    out = extract_tickers("삼성전자 반등할까")
    assert "005930" in out
    out = extract_tickers("테슬라 5% 상승")
    assert "TSLA" in out
