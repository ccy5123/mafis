"""Tests for the CrewAI tool wrappers in agents/tools.py.

Most behavior is covered by the underlying calculation tool tests. This file
verifies:
- The @tool decorator produced valid CrewAI tool objects.
- Each tool has a non-empty description (visible to the LLM).
- The dynamic field-list substitution in verify_number's description worked.
- Network: one end-to-end call per tool confirms the output format is a
  non-empty string the LLM can consume.
"""

from __future__ import annotations

import pytest

from wise_investor.agents.tools import (
    ALL_TOOLS,
    tool_calculate_ev_ebitda,
    tool_calculate_per,
    tool_cross_validate_quote,
    tool_get_peer_multiples,
    tool_reverse_dcf,
    tool_verify_number,
)
from wise_investor.config import settings


# ---------------------------------------------------------------------------
# Offline: metadata
# ---------------------------------------------------------------------------


def test_all_tools_registered() -> None:
    assert len(ALL_TOOLS) == 6


def test_each_tool_has_name_and_description() -> None:
    for t in ALL_TOOLS:
        assert t.name
        assert t.description
        assert len(t.description) > 20, f"{t.name}: description too short"


def test_verify_number_description_lists_supported_fields() -> None:
    desc = tool_verify_number.description
    # These should appear after the {fields} substitution at import time.
    assert "revenue" in desc
    assert "per" in desc
    assert "implied_growth_rate" in desc


def test_tool_names_are_unique() -> None:
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# Network: one call per tool
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def _require_fmp() -> None:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        pytest.skip("FMP_API_KEY not set")


@pytest.mark.network
def test_network_cross_validate_tool_on_aapl(_require_fmp) -> None:
    out = tool_cross_validate_quote.run(symbol="AAPL")
    assert "Cross-validation for AAPL" in out
    assert "price" in out
    assert "market_cap" in out


@pytest.mark.network
def test_network_calculate_per_tool_on_aapl(_require_fmp) -> None:
    out = tool_calculate_per.run(symbol="AAPL")
    assert "PER for AAPL" in out
    assert "Computed PER" in out
    assert "Source:" in out


@pytest.mark.network
def test_network_calculate_ev_ebitda_tool_on_aapl(_require_fmp) -> None:
    out = tool_calculate_ev_ebitda.run(symbol="AAPL")
    assert "EV/EBITDA for AAPL" in out
    assert "Enterprise Value" in out


@pytest.mark.network
def test_network_peer_multiples_tool_on_aapl(_require_fmp) -> None:
    out = tool_get_peer_multiples.run(symbol="AAPL", max_peers=3)
    assert "Peer multiples table" in out
    assert "AAPL" in out


@pytest.mark.network
def test_network_reverse_dcf_tool_on_aapl(_require_fmp) -> None:
    out = tool_reverse_dcf.run(symbol="AAPL")
    assert "Reverse DCF for AAPL" in out
    assert "Implied annual FCF growth" in out
    assert "discount rate" in out


@pytest.mark.network
def test_network_verify_number_tool_on_aapl(_require_fmp) -> None:
    # Deliberately wrong revenue; tool should report MISMATCH.
    out = tool_verify_number.run(claim=1.0, field="revenue", symbol="AAPL")
    assert "Verification for AAPL.revenue" in out
    assert "MISMATCH" in out
