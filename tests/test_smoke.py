"""Phase 0 smoke tests — verify the scaffold imports cleanly."""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    import wise_investor

    assert wise_investor.__version__ == "0.1.0"


def test_agent_sampling_follows_model_recommendation() -> None:
    """Phase 5 policy shift: MAFIS no longer guarantees byte-identical
    output across runs. Each agent gets the model-author-published
    recommended sampling config (e.g. Qwen 2.5: temp=0.7 / top_p=0.8;
    Qwen3 thinking: temp=0.6 / top_p=0.95 / min_p=0). Users who need
    deterministic output for backtests can override via
    `config/agent_models.yaml` per-agent — see docs/llm_backends.md.

    This test asserts the new contract: every crew agent resolves to
    a non-zero temperature with the model-family recommendation
    applied.
    """
    from wise_investor.llm.config import get_agent_config

    for agent in ("economist", "analyst", "valuer", "skeptic", "defender", "steward"):
        cfg = get_agent_config(agent, backend="ollama")
        assert cfg.model, f"{agent} did not resolve a model"
        assert cfg.sampling.temperature > 0, (
            f"{agent} sampling temperature must follow the model "
            "recommendation, not the legacy temperature=0 contract"
        )
        assert cfg.sampling.top_p is not None or cfg.sampling.top_k is not None


def test_model_assignment_matches_design() -> None:
    """Bull side uses one model, Bear side uses a distinct local model (design-v2.2 §7.4).

    Phase 1C restores this invariant: Analyst/Valuer on Qwen 2.5 7B-16k,
    Skeptic on Llama 3.1 8B-16k. Genuinely different training corpora and
    architectures for adversarial diversity.
    """
    from wise_investor.config import settings

    bull_models = {settings.analyst_model, settings.valuer_model}
    bear_model = settings.skeptic_model

    assert bear_model not in bull_models, (
        "Skeptic must use a model different from Analyst/Valuer "
        "(local diversity, design-v2.2 §7.4)"
    )


def test_subpackages_importable() -> None:
    import wise_investor.agents  # noqa: F401
    import wise_investor.data  # noqa: F401
    import wise_investor.rag  # noqa: F401
    import wise_investor.tools  # noqa: F401
