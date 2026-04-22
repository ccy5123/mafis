"""Phase 0 smoke tests — verify the scaffold imports cleanly."""

from __future__ import annotations

import pytest


def test_package_imports() -> None:
    import wise_investor

    assert wise_investor.__version__ == "0.1.0"


def test_config_defaults_enforce_reproducibility() -> None:
    """design-v2.2 re-review Critical #1: temperature=0, fixed seed."""
    from wise_investor.config import settings

    assert settings.llm_temperature == 0.0, "temperature must be 0 for reproducibility"
    assert settings.llm_seed == 42, "seed must be fixed for reproducibility"


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
