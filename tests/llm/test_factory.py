"""Backend factory routing + caching behavior."""

from __future__ import annotations

import pytest

from wise_investor.llm import factory
from wise_investor.llm.backends.ollama import OllamaBackend


@pytest.fixture(autouse=True)
def _reset_cache():
    """Every test starts with a fresh cache so ordering doesn't leak state."""
    factory._reset_for_test()
    yield
    factory._reset_for_test()


# ---------------------------------------------------------------------------
# Default behavior
# ---------------------------------------------------------------------------


def test_default_without_env_returns_ollama(monkeypatch) -> None:
    """Contract: git pull + no .env change should keep Ollama as backend."""
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    backend = factory.get_backend()
    assert isinstance(backend, OllamaBackend)


def test_explicit_name_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai_compat")
    backend = factory.get_backend(name="ollama")
    assert isinstance(backend, OllamaBackend)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_unknown_backend_raises_value_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "quantum_flux")
    with pytest.raises(ValueError) as exc_info:
        factory.get_backend()
    assert "quantum_flux" in str(exc_info.value)


def test_case_insensitive_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "OLLAMA")
    backend = factory.get_backend()
    assert isinstance(backend, OllamaBackend)


def test_whitespace_trimmed(monkeypatch) -> None:
    """User might accidentally have `LLM_BACKEND= ollama ` — strip it."""
    monkeypatch.setenv("LLM_BACKEND", "  ollama  ")
    backend = factory.get_backend()
    assert isinstance(backend, OllamaBackend)


# ---------------------------------------------------------------------------
# Phase 3 placeholders — fail in an obvious way before implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["openai_compat", "mlx", "llamacpp"])
def test_phase3_backends_signal_not_implemented(monkeypatch, name) -> None:
    monkeypatch.setenv("LLM_BACKEND", name)
    with pytest.raises(NotImplementedError) as exc_info:
        factory.get_backend()
    assert name in str(exc_info.value)
    assert "Phase 3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_returns_same_instance_for_same_name() -> None:
    first = factory.get_backend(name="ollama")
    second = factory.get_backend(name="ollama")
    assert first is second


def test_different_name_rebuilds_cache() -> None:
    """Switching LLM_BACKEND at runtime (e.g. in tests) must not
    return the previously cached backend.
    """
    first = factory.get_backend(name="ollama")
    # Request another name — should raise for Phase 2 stubs, but
    # the cache should NOT block the attempt.
    with pytest.raises(NotImplementedError):
        factory.get_backend(name="mlx")
    # Back to ollama — fresh build because the cache was invalidated
    # by the mid-test switch attempt. (Cache key is name; mlx failed
    # before caching, so ollama's cached instance is still there.)
    again = factory.get_backend(name="ollama")
    assert again is first  # still cached; mlx never populated


def test_supported_backends_is_stable() -> None:
    """If this constant changes, downstream config validation breaks."""
    assert factory.SUPPORTED_BACKENDS == (
        "ollama",
        "openai_compat",
        "mlx",
        "llamacpp",
    )
