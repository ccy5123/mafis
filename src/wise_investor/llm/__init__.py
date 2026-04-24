"""Pluggable LLM backend layer.

Users pick a backend (Ollama, MLX, llama.cpp, OpenAI-compat) via the
LLM_BACKEND env var; agents receive a `LLMBackend` instance and an
`SamplingConfig` without knowing which runtime actually serves the model.

This package is currently a stand-alone scaffold — existing agents still
call `ollama.chat(...)` directly. Phase 5 migrates them onto this API.
"""

from wise_investor.llm.base import LLMBackend, LLMResponse, SamplingConfig
from wise_investor.llm.factory import get_backend

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "SamplingConfig",
    "get_backend",
]
