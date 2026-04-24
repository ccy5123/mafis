"""Model-family sampling default resolution."""

from __future__ import annotations

import pytest

from wise_investor.llm.utils.sampling import (
    get_recommended_sampling,
    known_families,
)


# ---------------------------------------------------------------------------
# Family matching — each supported family must resolve
# ---------------------------------------------------------------------------


def test_qwen3_default_is_non_thinking_variant() -> None:
    """Historical MAFIS never used thinking mode, so Qwen3 tags with
    no explicit override should land on the non-thinking variant.
    """
    s = get_recommended_sampling("qwen3:14b")
    assert s.temperature == 0.7
    assert s.top_p == 0.8
    assert s.enable_thinking is False


def test_qwen3_thinking_variant_when_requested() -> None:
    s = get_recommended_sampling("qwen3:14b", enable_thinking=True)
    assert s.temperature == 0.6
    assert s.top_p == 0.95
    assert s.min_p == 0.0
    assert s.enable_thinking is True


def test_qwen3_non_thinking_explicit() -> None:
    s = get_recommended_sampling("qwen3:14b", enable_thinking=False)
    assert s.temperature == 0.7
    assert s.enable_thinking is False


def test_deepseek_r1_enables_thinking_by_default() -> None:
    """DeepSeek-R1's recommended path IS thinking mode; infrastructure
    should reflect that.
    """
    s = get_recommended_sampling("deepseek-r1-distill-qwen-7b")
    assert s.temperature == 0.6
    assert s.top_p == 0.95
    assert s.enable_thinking is True


def test_deepseek_r1_override_disables_thinking() -> None:
    s = get_recommended_sampling(
        "deepseek-r1-distill-qwen-7b", enable_thinking=False
    )
    assert s.enable_thinking is False


def test_qwen25_defaults_match_published_recipe() -> None:
    s = get_recommended_sampling("qwen2.5:7b-16k")
    assert s.temperature == 0.7
    assert s.top_p == 0.8


def test_llama3_family_matches_ollama_tag() -> None:
    s = get_recommended_sampling("llama3.1:8b")
    assert s.temperature == 0.7
    assert s.top_p == 0.9


def test_llama_3_dot_family_matches_openai_style_tag() -> None:
    """Matches e.g. 'meta/Llama-3.1-8B-Instruct' from OpenAI-compat servers."""
    s = get_recommended_sampling("meta/Llama-3.1-8B-Instruct")
    assert s.temperature == 0.7
    assert s.top_p == 0.9


def test_glm4_matches() -> None:
    s = get_recommended_sampling("glm-4-9b-chat")
    assert s.temperature == 0.8
    assert s.top_p == 0.8


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------


def test_unknown_model_returns_generic_defaults() -> None:
    """Unknown tags shouldn't raise — return a mid-temperature fallback."""
    s = get_recommended_sampling("brand_new_model_42")
    assert 0.0 < s.temperature < 1.5
    assert 0.0 < s.top_p <= 1.0
    assert s.enable_thinking is False


def test_empty_model_name_returns_generic_defaults() -> None:
    s = get_recommended_sampling("")
    assert s.temperature > 0


# ---------------------------------------------------------------------------
# Policy — no family resolves to the legacy (temp=0, seed=42) config
# ---------------------------------------------------------------------------


def test_no_family_default_is_fully_deterministic() -> None:
    """After the policy change, recommended sampling MUST NOT pin
    temperature=0 for any supported family. That's the whole point.
    """
    for family in known_families():
        s = get_recommended_sampling(f"{family}:test")
        assert s.temperature > 0, f"{family} returned deterministic default"
        assert s.seed is None, f"{family} returned deterministic seed"


def test_known_families_are_documented_set() -> None:
    """Guardrail: if someone adds a family, they must also add it to
    the docs. This test enumerates the published set so changes are
    visible in diff review.
    """
    assert set(known_families()) == {
        "qwen3",
        "deepseek-r1",
        "qwen2.5",
        "llama-3",
        "llama3",
        "glm-4",
    }
