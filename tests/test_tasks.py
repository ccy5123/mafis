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


# ---------------------------------------------------------------------------
# Edgar citation mandates (Phase 3D-strengthened prompts)
# ---------------------------------------------------------------------------


def test_analyst_template_mandates_edgar_business_citation() -> None:
    agent = make_analyst()
    task = make_analyst_task("NVDA", agent=agent)
    # Business Summary section must require at least one edgar.business_segments citation.
    assert "10-K GROUNDING (MANDATORY)" in task.description
    assert "edgar.business_segments" in task.description


def test_analyst_template_mandates_edgar_moat_citation() -> None:
    agent = make_analyst()
    task = make_analyst_task("NVDA", agent=agent)
    assert "edgar.moat_signals" in task.description


def test_skeptic_template_mandates_edgar_risk_citation() -> None:
    from wise_investor.agents.tasks import SKEPTIC_REPORT_TEMPLATE

    # At least one rebuttal must cite edgar.risk_factors.
    assert "10-K GROUNDING (MANDATORY)" in SKEPTIC_REPORT_TEMPLATE
    assert "edgar.risk_factors" in SKEPTIC_REPORT_TEMPLATE


def test_economist_template_references_geo_snapshot() -> None:
    from wise_investor.agents.tasks import ECONOMIST_REPORT_TEMPLATE

    # Economist must cite news from geo.snapshot when relevant.
    assert "geo.snapshot" in ECONOMIST_REPORT_TEMPLATE
    assert "Google News" in ECONOMIST_REPORT_TEMPLATE
    assert "GDELT" in ECONOMIST_REPORT_TEMPLATE
