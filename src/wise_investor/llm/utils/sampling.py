"""Model-family recommended sampling defaults.

Each entry encodes the sampling settings the model's authors publish
as "best for downstream tasks". Matching is substring-based on the
model identifier the user has configured (`qwen2.5:7b-16k`,
`mlx-community/Qwen3-14B-Instruct-4bit`, `llama3.1:8b`, etc.), so a
tag like `qwen3:14b-thinking-q4_K_M` still maps to the Qwen3 entry.

Policy:
  - Pure keyword match, longest-match wins.
  - Unknown models fall back to a generic mid-temperature default
    rather than raising — users can always override via
    `config/agent_models.yaml`.
  - Qwen3 has two variants because its thinking vs non-thinking modes
    have different published recommendations.

References (captured 2026-04, pinned in `docs/llm_backends.md`):
  - Qwen3: https://github.com/QwenLM/Qwen3 README "Best Practices"
  - DeepSeek-R1: DeepSeek-AI/DeepSeek-R1 inference guidelines
  - Llama 3.1 / 3.2: Meta Llama recipes
  - Qwen 2.5: Qwen/Qwen2.5 README
  - GLM-4: THUDM/GLM-4 inference defaults
"""

from __future__ import annotations

from wise_investor.llm.base import SamplingConfig


# Ordered longest-prefix-first so "qwen3" matches before "qwen2.5"-ish
# typos and "deepseek-r1" matches before "deepseek".
_MODEL_FAMILY_DEFAULTS: list[tuple[str, dict]] = [
    (
        "qwen3",
        {
            "thinking": {
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.0,
                "enable_thinking": True,
            },
            "non_thinking": {
                "temperature": 0.7,
                "top_p": 0.8,
                "enable_thinking": False,
            },
        },
    ),
    (
        "deepseek-r1",
        {
            "temperature": 0.6,
            "top_p": 0.95,
            "enable_thinking": True,
        },
    ),
    (
        "qwen2.5",
        {
            "temperature": 0.7,
            "top_p": 0.8,
            "enable_thinking": False,
        },
    ),
    (
        "llama-3",  # matches llama-3.1, llama-3.2, etc. in OpenAI-style tags
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "enable_thinking": False,
        },
    ),
    (
        "llama3",  # matches llama3.1, llama3.2 as served by Ollama
        {
            "temperature": 0.7,
            "top_p": 0.9,
            "enable_thinking": False,
        },
    ),
    (
        "glm-4",
        {
            "temperature": 0.8,
            "top_p": 0.8,
            "enable_thinking": False,
        },
    ),
]


# Generic fallback when no family matches. Conservative mid-temp so
# unknown models at least produce non-garbage text.
_GENERIC_DEFAULT = {
    "temperature": 0.7,
    "top_p": 0.9,
    "enable_thinking": False,
}


def _match_family(model_name: str) -> tuple[str, dict] | None:
    """Return (family_key, family_defaults) for `model_name`, or None.

    Matching is substring and case-insensitive so callers pass the
    model tag exactly as they configured it. Longer family keys match
    first because the list is ordered — `qwen3` beats any ambiguity
    against `qwen2.5`.
    """
    lowered = model_name.lower()
    for family, defaults in _MODEL_FAMILY_DEFAULTS:
        if family in lowered:
            return family, defaults
    return None


def get_recommended_sampling(
    model_name: str,
    enable_thinking: bool | None = None,
) -> SamplingConfig:
    """Build a `SamplingConfig` populated with the model's recommended
    defaults.

    Args:
        model_name: the backend-specific model identifier.
        enable_thinking: override for families (like Qwen3) that have
            separate thinking / non-thinking recommendations. Pass
            None to let the family's own default decide.

    Caller is free to override any field afterwards via `dataclasses.replace`.
    """
    match = _match_family(model_name)
    if match is None:
        return SamplingConfig(**_GENERIC_DEFAULT)

    family, defaults = match

    # Qwen3 splits into thinking / non_thinking sub-configs.
    if "thinking" in defaults and "non_thinking" in defaults:
        if enable_thinking is True:
            variant = defaults["thinking"]
        elif enable_thinking is False:
            variant = defaults["non_thinking"]
        else:
            # Caller didn't specify — default to non-thinking for
            # backward compatibility (MAFIS historically didn't use
            # thinking mode).
            variant = defaults["non_thinking"]
        return SamplingConfig(**variant)

    # Flat family defaults (deepseek-r1, llama3, qwen2.5, glm-4).
    config = SamplingConfig(**defaults)
    # Allow explicit enable_thinking override even for families that
    # don't split (e.g. forcing DeepSeek-R1 to hide thinking).
    if enable_thinking is not None:
        config.enable_thinking = enable_thinking
    return config


def known_families() -> list[str]:
    """Family keys in their declaration order. Useful for tests and docs."""
    return [family for family, _ in _MODEL_FAMILY_DEFAULTS]


__all__ = ["get_recommended_sampling", "known_families"]
