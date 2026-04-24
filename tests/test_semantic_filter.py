"""Tests for the Stage 3 semantic filter (Qwen-backed relevance scoring)."""

from __future__ import annotations

from wise_investor.filters.pre_filter import FilterHit
from wise_investor.filters.semantic import (
    SemanticDecision,
    _parse_decision,
    filter_hits_semantically,
    materials_only,
)


def _hit(title: str = "NVIDIA reports Q1 earnings beat", symbol: str = "NVDA") -> FilterHit:
    return FilterHit(
        symbol=symbol,
        stage="keyword",
        matched_term="NVIDIA",
        news_title=title,
        news_source="Reuters",
        news_published="2026-04-24",
    )


# ---------------------------------------------------------------------------
# _parse_decision
# ---------------------------------------------------------------------------


def test_parse_decision_yes_token_material() -> None:
    is_material, reason = _parse_decision("YES\nEarnings beat is material.")
    assert is_material is True
    assert "material" in reason.lower()


def test_parse_decision_no_token_immaterial() -> None:
    is_material, reason = _parse_decision("NO\nRoutine PR, no new facts.")
    assert is_material is False
    assert "pr" in reason.lower() or "routine" in reason.lower()


def test_parse_decision_empty_response_is_no() -> None:
    is_material, reason = _parse_decision("")
    assert is_material is False


def test_parse_decision_ambiguous_text_is_no() -> None:
    is_material, _ = _parse_decision("It depends")
    assert is_material is False


def test_parse_decision_no_overrides_yes_when_both_in_verdict_line() -> None:
    """Pathological LLM output like 'YES and NO' on one line → NO."""
    is_material, _ = _parse_decision("YES but also NO for various reasons")
    assert is_material is False


def test_parse_decision_accepts_material_synonym() -> None:
    is_material, _ = _parse_decision("MATERIAL\nSupply chain shock.")
    assert is_material is True


# ---------------------------------------------------------------------------
# filter_hits_semantically
# ---------------------------------------------------------------------------


def test_filter_hits_semantically_empty_list_short_circuits() -> None:
    # Inject an llm_call that would raise if invoked.
    def _boom(*a, **k):
        raise AssertionError("LLM should not be called on empty hits")

    out = filter_hits_semantically([], llm_call=_boom)
    assert out == []


def test_filter_hits_semantically_yes_kept() -> None:
    def _llm(system, user):
        return "YES\nEarnings beat."

    hits = [_hit()]
    decisions = filter_hits_semantically(hits, llm_call=_llm)
    assert len(decisions) == 1
    assert decisions[0].is_material is True
    assert decisions[0].reason == "Earnings beat."


def test_filter_hits_semantically_no_dropped() -> None:
    def _llm(system, user):
        return "NO\nCharity donation, not thesis-material."

    hits = [_hit(title="NVIDIA donates to STEM scholarship")]
    decisions = filter_hits_semantically(hits, llm_call=_llm)
    assert decisions[0].is_material is False


def test_filter_hits_semantically_respects_max_hits_cap() -> None:
    """Beyond max_hits, the LLM is NOT called; hits are auto-NO."""
    calls = {"n": 0}

    def _llm(system, user):
        calls["n"] += 1
        return "YES\nMaterial."

    hits = [_hit(title=f"Headline {i}") for i in range(5)]
    decisions = filter_hits_semantically(hits, llm_call=_llm, max_hits=2)
    # Only 2 LLM calls; the remaining 3 are auto-NO.
    assert calls["n"] == 2
    kept = [d for d in decisions if d.is_material]
    assert len(kept) == 2
    dropped = [d for d in decisions if not d.is_material]
    assert len(dropped) == 3
    # Truncation reason for the capped hits.
    assert all("max_hits" in d.reason for d in dropped)


def test_filter_hits_semantically_llm_error_becomes_no() -> None:
    def _boom(system, user):
        raise RuntimeError("ollama down")

    decisions = filter_hits_semantically([_hit()], llm_call=_boom)
    assert decisions[0].is_material is False
    assert "llm error" in decisions[0].reason


def test_materials_only_extracts_kept_hits() -> None:
    h1 = _hit(title="Material headline")
    h2 = _hit(title="Immaterial headline")
    decisions = [
        SemanticDecision(hit=h1, is_material=True, reason="", raw_response=""),
        SemanticDecision(hit=h2, is_material=False, reason="", raw_response=""),
    ]
    kept = materials_only(decisions)
    assert kept == [h1]


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_system_prompt_covers_material_and_immaterial_categories() -> None:
    """Guardrails: the system prompt must enumerate what counts as
    material (earnings, supply shock, etc.) AND what's immaterial
    (routine PR, ETF inclusion). Regression guard.
    """
    from wise_investor.filters.semantic import _SYSTEM_PROMPT

    assert "earnings" in _SYSTEM_PROMPT.lower()
    assert "supply chain" in _SYSTEM_PROMPT.lower()
    assert "regulatory" in _SYSTEM_PROMPT.lower()
    assert "pr" in _SYSTEM_PROMPT.lower() or "routine" in _SYSTEM_PROMPT.lower()
