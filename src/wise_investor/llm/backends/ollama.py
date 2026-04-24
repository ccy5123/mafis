"""Ollama backend — default on all platforms.

Wraps the existing `ollama` Python client with the `LLMBackend`
contract. The wrapper is a thin translation layer, not a behavior
change — Phase 2 keeps production identical to pre-abstraction MAFIS.
The Phase 5 migration is what actually routes agents through this
backend (and that's where model-recommended sampling replaces the
hardcoded `temperature=0, seed=42`).

Design notes:
  - `options` is the Ollama-specific sampling dict. We populate only
    the keys we actually need — passing `num_predict` when
    `max_tokens is None` would truncate generation, which differs
    from current behavior (Ollama default is -1 = unlimited).
  - `tools` / `keep_alive` pass through as kwargs so the runner's
    tool-loop and model-swap logic work unchanged when migrated.
  - `is_available()` uses a small `/api/tags` GET so we can light up
    UI hints without triggering a model load.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from wise_investor.llm.base import LLMBackend, LLMResponse, SamplingConfig
from wise_investor.llm.utils.thinking import strip_thinking


logger = logging.getLogger(__name__)


# Default endpoint — overridden by OLLAMA_HOST / constructor arg.
_DEFAULT_HOST = "http://localhost:11434"


class OllamaBackend(LLMBackend):
    """Local Ollama server. Zero setup on the user side beyond
    `ollama pull <model>`.
    """

    def __init__(self, host: str | None = None, timeout: float = 600.0) -> None:
        # Import lazily so the module stays import-clean when the
        # `ollama` extra isn't installed (which shouldn't happen for
        # the default backend, but keeps Phase 2 isolation tidy).
        try:
            import ollama  # noqa: F401
        except ImportError as e:  # pragma: no cover — default dep
            raise RuntimeError(
                "The `ollama` package is required for OllamaBackend. "
                "Install via `pip install ollama`."
            ) from e

        if host is None:
            # Prefer settings.ollama_host so users can override via env;
            # fall back to the documented localhost default.
            try:
                from wise_investor.config import settings

                host = settings.ollama_host or _DEFAULT_HOST
            except Exception:
                host = _DEFAULT_HOST

        self.host = host.rstrip("/")
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    def is_available(self) -> bool:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=min(self.timeout, 5.0))
            return r.status_code < 400
        except Exception as e:
            logger.debug("Ollama availability check failed: %s", e)
            return False

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.host}/api/tags", timeout=min(self.timeout, 5.0))
            if r.status_code >= 400:
                return []
            payload = r.json()
        except Exception as e:
            logger.warning("Ollama /api/tags failed: %s", e)
            return []

        models = payload.get("models") or []
        return [
            m["name"]
            for m in models
            if isinstance(m, dict) and isinstance(m.get("name"), str)
        ]

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling: SamplingConfig,
        **kwargs: Any,
    ) -> LLMResponse:
        import ollama

        options = _sampling_to_ollama_options(sampling)

        ollama_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
        }

        # Pass-through for tool-loop + model-swap use cases (runner.py).
        # Explicit keys only — we don't forward every kwarg to avoid
        # silently changing ollama.chat semantics.
        for key in ("tools", "keep_alive", "format", "stream"):
            if key in kwargs:
                ollama_kwargs[key] = kwargs[key]

        resp = ollama.chat(**ollama_kwargs)
        message = resp.get("message") or {}
        raw_content = message.get("content") or ""

        cleaned, thinking = strip_thinking(raw_content)

        extra: dict[str, Any] = {}
        # Preserve tool_calls when present — runner.py needs them.
        if message.get("tool_calls"):
            extra["tool_calls"] = message["tool_calls"]
        for key in ("eval_count", "prompt_eval_count", "total_duration"):
            if key in resp:
                extra[key] = resp[key]

        return LLMResponse(
            content=cleaned,
            model=model,
            backend=self.name,
            sampling_config=sampling.as_dict(),
            thinking=thinking,
            extra=extra,
        )

    def make_crewai_llm(self, model: str, sampling: SamplingConfig) -> Any:
        """Build a `crewai.LLM` routed through LiteLLM's Ollama provider.

        CrewAI agents (Analyst in the current codebase) use this path
        rather than `chat()` so their tool-calling loop stays CrewAI-
        native. The seed is only forwarded when the user has explicitly
        opted into deterministic mode — otherwise we let LiteLLM use
        its native default.
        """
        from crewai import LLM

        kwargs: dict[str, Any] = {
            "model": f"ollama/{model}",
            "base_url": self.host,
            "temperature": sampling.temperature,
        }
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        return LLM(**kwargs)


def _sampling_to_ollama_options(sampling: SamplingConfig) -> dict[str, Any]:
    """Translate `SamplingConfig` to Ollama's `options` dict.

    Only keys the user actually set are emitted so we don't override
    Ollama defaults for unspecified fields. `max_tokens=None` in
    particular MUST NOT become `num_predict=None` — we just omit it
    so Ollama uses its -1 (unlimited) default.
    """
    options: dict[str, Any] = {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
    }
    if sampling.top_k is not None:
        options["top_k"] = sampling.top_k
    if sampling.min_p is not None:
        options["min_p"] = sampling.min_p
    if sampling.max_tokens is not None:
        options["num_predict"] = sampling.max_tokens
    if sampling.seed is not None:
        options["seed"] = sampling.seed
    return options


__all__ = ["OllamaBackend"]
