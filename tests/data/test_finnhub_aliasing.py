"""Tests for the financials-reported ticker alias map (#4)."""

from __future__ import annotations

from wise_investor.data.finnhub import _financials_symbol


def test_goog_aliases_to_googl() -> None:
    """Calibration finding (#4, 2026-04): post-2015 Alphabet reorg
    moved 10-K filings to GOOGL. Querying GOOG returns 4 stale
    pre-2015 entries; GOOGL returns the current 11 entries."""
    assert _financials_symbol("GOOG") == "GOOGL"
    assert _financials_symbol("goog") == "GOOGL"  # case-insensitive


def test_default_passthrough() -> None:
    """Most tickers map to themselves — the alias is the exception."""
    assert _financials_symbol("NVDA") == "NVDA"
    assert _financials_symbol("MSFT") == "MSFT"
    assert _financials_symbol("BRK-B") == "BRK-B"


def test_googl_already_correct_passthrough() -> None:
    """GOOGL itself shouldn't be mangled — the alias only fires on
    GOOG (the Class C symbol)."""
    assert _financials_symbol("GOOGL") == "GOOGL"


def test_lowercase_normalized_to_upper() -> None:
    assert _financials_symbol("nvda") == "NVDA"
