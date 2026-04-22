"""Sanity checks for the Analyst task factory.

Verifies value chain injection and the rigid seven-section contract without
spinning up a live LLM.
"""

from __future__ import annotations

import pytest

from wise_investor.agents.analyst import make_analyst
from wise_investor.agents.tasks import make_analyst_task


def test_task_injects_value_chain_for_nvda() -> None:
    agent = make_analyst()
    task = make_analyst_task("NVDA", agent=agent)
    # The NVDA value chain doc is part of the repo; its contents must end up
    # inside the task description so the LLM sees it without RAG. Wrapped in
    # XML tags per Anthropic prompt-engineering guidance for source documents.
    assert "TSMC" in task.description
    assert "Vulnerable links" in task.description or "Vulnerable Links" in task.description
    assert "<value_chain_brief>" in task.description
    assert "</value_chain_brief>" in task.description


def test_task_description_enforces_seven_sections() -> None:
    agent = make_analyst()
    task = make_analyst_task("NVDA", agent=agent)
    for heading in [
        "## 1. Business Summary",
        "## 2. Value Chain Context",
        "## 3. Financial Health",
        "## 4. Competitive Position / Moat",
        "## 5. Valuation Context",
        "## 6. Data Gaps and Warnings",
        "## 7. Questions for Skeptic",
    ]:
        assert heading in task.description, f"Missing mandatory section: {heading}"


def test_task_forbids_buy_sell_recommendation() -> None:
    agent = make_analyst()
    task = make_analyst_task("NVDA", agent=agent)
    # Phase 1B: Steward does Buy/Hold/Pass; Analyst must not.
    assert "Do NOT issue a buy/sell/hold recommendation" in task.description


def test_task_uppercases_symbol() -> None:
    agent = make_analyst()
    task = make_analyst_task("nvda", agent=agent)
    assert "NVDA" in task.description
    assert "nvda" not in task.description.lower().split("nvda")[0]  # clean uppercase


def test_task_raises_when_value_chain_missing() -> None:
    agent = make_analyst()
    with pytest.raises(FileNotFoundError, match="value chain document"):
        make_analyst_task("ZZZZ", agent=agent)
