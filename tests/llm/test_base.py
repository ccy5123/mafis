"""Tests for the abstract LLM layer dataclasses."""

from __future__ import annotations

from wise_investor.llm.base import LLMResponse, SamplingConfig


def test_sampling_config_defaults_are_generic_mid_temperature() -> None:
    """Defaults should NOT be deterministic (temperature=0) anymore —
    the new policy hands that decision to the per-model recommendation.
    """
    s = SamplingConfig()
    assert s.temperature > 0
    assert s.top_p > 0
    assert s.max_tokens is None       # unlimited unless caller sets
    assert s.enable_thinking is False
    assert s.seed is None             # non-deterministic by default


def test_sampling_config_as_dict_roundtrips_for_meta_txt() -> None:
    """meta.txt recording relies on .as_dict() returning every field
    so reviewers see "I did set top_k to None" vs "top_k not mentioned".
    """
    s = SamplingConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        max_tokens=2048,
        enable_thinking=True,
        seed=42,
    )
    d = s.as_dict()
    assert d == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "max_tokens": 2048,
        "enable_thinking": True,
        "seed": 42,
    }


def test_llm_response_thinking_defaults_to_none() -> None:
    """Non-thinking models should produce thinking=None so callers
    can do `if resp.thinking:` without worrying about empty strings.
    """
    r = LLMResponse(
        content="hello",
        model="qwen2.5:7b",
        backend="ollama",
        sampling_config={},
    )
    assert r.thinking is None
    assert r.extra == {}


def test_llm_response_preserves_extra_dict() -> None:
    r = LLMResponse(
        content="hi",
        model="m",
        backend="b",
        sampling_config={},
        extra={"tool_calls": [{"id": 1}], "eval_count": 42},
    )
    assert r.extra["tool_calls"] == [{"id": 1}]
    assert r.extra["eval_count"] == 42
