"""Entry point that hands out the correct `LLMBackend` instance.

Routing:
  - `LLM_BACKEND=ollama` (default): always returns `OllamaBackend`.
  - `LLM_BACKEND=openai_compat`: any OpenAI-wire-compatible server
    (vLLM, LM Studio, Ollama's own `/v1/` endpoint, LocalAI).
  - `LLM_BACKEND=mlx`: Apple Silicon only; import failure is a
    "not available" signal rather than a hard crash (other platforms
    can still import the factory module).
  - `LLM_BACKEND=llamacpp`: llama-cpp-python with a local GGUF.

Unknown values raise — typos in `.env` should surface immediately,
not silently fall back to Ollama, because the resulting behavior
mismatch would be hard to debug.

The factory caches the selected backend as a module-level singleton
so repeated calls don't re-probe availability; tests reset via
`_reset_for_test()`.
"""

from __future__ import annotations

import logging
import os
import threading

from wise_investor.llm.base import LLMBackend


logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS: tuple[str, ...] = (
    "ollama",
    "openai_compat",
    "mlx",
    "llamacpp",
)


_DEFAULT_BACKEND = "ollama"


_cache_lock = threading.Lock()
_cached_backend: LLMBackend | None = None
_cached_name: str | None = None


def get_backend(name: str | None = None) -> LLMBackend:
    """Return an `LLMBackend` for `name` (or `$LLM_BACKEND`, default ollama).

    Caching: the first call caches the resolved backend. Subsequent
    calls with the SAME name return the cached instance; calls with a
    different name build a fresh one (but replace the cache).
    """
    requested = (name or os.getenv("LLM_BACKEND") or _DEFAULT_BACKEND).strip().lower()
    if requested not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown LLM_BACKEND={requested!r}. "
            f"Supported: {', '.join(SUPPORTED_BACKENDS)}."
        )

    global _cached_backend, _cached_name
    with _cache_lock:
        if _cached_backend is not None and _cached_name == requested:
            return _cached_backend
        backend = _build_backend(requested)
        _cached_backend = backend
        _cached_name = requested
        return backend


def _build_backend(name: str) -> LLMBackend:
    """Instantiate the named backend. Imports are lazy so missing
    optional dependencies only fail for the backend that needs them.
    """
    if name == "ollama":
        from wise_investor.llm.backends.ollama import OllamaBackend

        return OllamaBackend()

    if name == "openai_compat":
        from wise_investor.llm.backends.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend()

    if name == "mlx":
        from wise_investor.llm.backends.mlx import MLXBackend

        return MLXBackend()

    if name == "llamacpp":
        from wise_investor.llm.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend()

    raise ValueError(f"Unreachable: {name!r} passed SUPPORTED_BACKENDS check")


def _reset_for_test() -> None:
    """Clear the module-level cache. Internal use by the test suite
    to cleanly switch backends across tests without module reload.
    """
    global _cached_backend, _cached_name
    with _cache_lock:
        _cached_backend = None
        _cached_name = None


__all__ = ["SUPPORTED_BACKENDS", "get_backend"]
