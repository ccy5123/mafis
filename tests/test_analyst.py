"""Sanity checks for the Analyst agent construction.

Actual Llama-powered execution is deferred to the scripts/run_analyst.py smoke —
instantiating a live agent inside pytest would pull the 4.7GB model into VRAM
and slow the suite considerably.
"""

from __future__ import annotations

from wise_investor.agents.analyst import (
    ANALYST_BACKSTORY,
    ANALYST_GOAL,
    make_analyst,
    make_analyst_llm,
)
from wise_investor.agents.tools import ALL_TOOLS
from wise_investor.config import settings


def test_analyst_llm_uses_configured_ollama_model() -> None:
    # CrewAI strips the "ollama/" provider prefix when storing the model name;
    # the prefix is still used internally by LiteLLM for provider routing.
    llm = make_analyst_llm()
    assert llm.model == settings.analyst_model
    assert llm.temperature == settings.llm_temperature


def test_analyst_llm_points_at_local_ollama() -> None:
    # CrewAI appends "/v1" to base_url to talk to Ollama's OpenAI-compatible API.
    llm = make_analyst_llm()
    assert llm.base_url.rstrip("/").startswith(settings.ollama_host.rstrip("/"))


def test_analyst_goal_names_value_chain_and_five_to_ten_years() -> None:
    assert "value chain" in ANALYST_GOAL.lower()
    assert "five" in ANALYST_GOAL.lower() and "ten" in ANALYST_GOAL.lower()


def test_analyst_backstory_enforces_core_principles() -> None:
    text = ANALYST_BACKSTORY.lower()
    assert "never invent numbers" in text
    assert "source" in text
    assert "valuer" in text  # boundary with Valuer agent
    assert "translation" in text  # English-only internal artifact rule


def test_make_analyst_attaches_all_tools() -> None:
    a = make_analyst()
    assert len(a.tools) == len(ALL_TOOLS)
    tool_names = {t.name for t in a.tools}
    for tl in ALL_TOOLS:
        assert tl.name in tool_names


def test_make_analyst_disables_delegation() -> None:
    # Phase 1B has a single agent; delegation would route to phantom teammates.
    a = make_analyst()
    assert a.allow_delegation is False
