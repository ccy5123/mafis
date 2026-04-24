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
# Phase 3 backends — wired up; they build when their deps + config permit
# ---------------------------------------------------------------------------


def test_openai_compat_fails_without_base_url(monkeypatch) -> None:
    """openai_compat requires OPENAI_COMPAT_BASE_URL. Without it the
    factory raises ValueError with a clear message — a typo in .env
    should surface immediately, not silently fall back to Ollama.
    """
    monkeypatch.setenv("LLM_BACKEND", "openai_compat")
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ValueError) as exc_info:
        factory.get_backend()
    assert "base" in str(exc_info.value).lower()


def test_openai_compat_builds_when_base_url_set(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
    backend = factory.get_backend()
    assert backend.name == "openai_compat"


def test_mlx_builds_when_dep_available_or_raises_clearly(monkeypatch) -> None:
    """MLX construction on non-Apple platforms raises RuntimeError
    mentioning Apple Silicon — the right behavior for the factory.
    """
    monkeypatch.setenv("LLM_BACKEND", "mlx")
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError) as exc_info:
            factory.get_backend()
        assert (
            "apple silicon" in str(exc_info.value).lower()
            or "mlx-lm" in str(exc_info.value)
        )
    else:
        backend = factory.get_backend()
        assert backend.name == "mlx"


def test_llamacpp_requires_model_path(monkeypatch) -> None:
    """llamacpp without a GGUF path raises — either RuntimeError
    (lib missing) or ValueError (path missing)."""
    monkeypatch.setenv("LLM_BACKEND", "llamacpp")
    monkeypatch.delenv("LLAMACPP_MODEL_PATH", raising=False)
    with pytest.raises((RuntimeError, ValueError)):
        factory.get_backend()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_returns_same_instance_for_same_name() -> None:
    first = factory.get_backend(name="ollama")
    second = factory.get_backend(name="ollama")
    assert first is second


def test_different_name_rebuilds_cache(monkeypatch) -> None:
    """Switching LLM_BACKEND at runtime must not return the previously
    cached backend. mlx construction fails on non-Apple (RuntimeError);
    that failure must NOT evict ollama's cached instance.
    """
    first = factory.get_backend(name="ollama")
    # Request mlx — expect it to fail on Windows/Linux dev machines.
    try:
        import mlx_lm  # noqa: F401

        mlx_available = True
    except ImportError:
        mlx_available = False

    if mlx_available:
        mlx_backend = factory.get_backend(name="mlx")
        assert mlx_backend.name == "mlx"
        # After building mlx, the cache now holds mlx.
        again = factory.get_backend(name="ollama")
        assert again.name == "ollama"  # rebuilt
    else:
        with pytest.raises(RuntimeError):
            factory.get_backend(name="mlx")
        # ollama cache survived the failed mlx attempt.
        again = factory.get_backend(name="ollama")
        assert again is first


def test_supported_backends_is_stable() -> None:
    """If this constant changes, downstream config validation breaks."""
    assert factory.SUPPORTED_BACKENDS == (
        "ollama",
        "openai_compat",
        "mlx",
        "llamacpp",
    )
