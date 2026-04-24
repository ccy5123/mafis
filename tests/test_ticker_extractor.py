"""Tests for the ticker-alias vocabulary helpers.

Extraction itself has moved to `classifier.py` (LLM-first, context-aware).
This file covers only the alias-map loader, inverse-index builder, and
shape normalizer.
"""

from __future__ import annotations

import json

import pytest

from wise_investor.ingest.ticker_extractor import (
    _normalize_ticker,
    build_inverse_index,
    load_aliases,
)


# Deterministic minimal alias map — keeps test output stable as the
# default seed list grows.
_TINY_MAP = {
    "NVDA": ["nvda", "nvidia", "엔비디아"],
    "TSM": ["tsm", "tsmc", "티에스엠씨"],
    "AMD": ["amd", "에이엠디"],
    "005930": ["005930", "삼성전자"],
}


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
# _normalize_ticker
# ---------------------------------------------------------------------------


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


def test_load_aliases_seeds_cover_common_korean_names() -> None:
    """The classifier uses these defaults to build its vocab hint;
    if these regress, the LLM loses Korean-name grounding.
    """
    aliases = load_aliases()
    assert "엔비디아" in aliases["NVDA"]
    assert "삼성전자" in aliases["005930"]
    assert "테슬라" in aliases["TSLA"]
