"""OpenAI wire-compatible HTTP backend.

Covers a wide family of servers that all speak the same `/v1/chat/
completions` shape:
  - vLLM (`python -m vllm.entrypoints.openai.api_server …`)
  - LM Studio
  - Ollama's own `/v1/` endpoint (alternative to the native API)
  - LocalAI, Jan, Text Generation WebUI (OpenAI extension), mlx-lm's
    server (`mlx_lm.server …`), SGLang, etc.

Configuration via env:
  OPENAI_COMPAT_BASE_URL  required at runtime (e.g. http://localhost:8000/v1)
  OPENAI_COMPAT_API_KEY   optional — most local servers don't require it,
                          but LiteLLM downstream wants *something* so we
                          default to the string "local" to keep the CrewAI
                          path happy.

Design notes:
  - Uses httpx.Client per call so connection reuse is trivial, and
    each backend instance owns its own timeout / base URL.
  - We do not assume the server supports streaming, seeds, or
    log-probs — the contract is just chat completion with sampling.
  - Thinking-mode output (servers serving Qwen3 / R1) comes back in
    the same `content` field; strip via the shared util so the agent
    layer never sees `<think>` tags.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from wise_investor.llm.base import LLMBackend, LLMResponse, SamplingConfig
from wise_investor.llm.utils.thinking import strip_thinking


logger = logging.getLogger(__name__)


_DEFAULT_API_KEY = "local"


class OpenAICompatBackend(LLMBackend):
    """HTTP backend for any OpenAI-compatible server."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 600.0,
    ) -> None:
        base_url = base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or ""
        if not base_url:
            raise ValueError(
                "OpenAICompatBackend requires a base URL. Set "
                "OPENAI_COMPAT_BASE_URL in .env or pass base_url="
                "<http://host:port/v1>."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else (os.getenv("OPENAI_COMPAT_API_KEY") or _DEFAULT_API_KEY)
        )
        self.timeout = timeout

    @property
    def name(self) -> str:
        return "openai_compat"

    # ---- availability + discovery ------------------------------------

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def is_available(self) -> bool:
        try:
            r = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=min(self.timeout, 5.0),
            )
            return r.status_code < 400
        except Exception as e:
            logger.debug("OpenAI-compat availability check failed: %s", e)
            return False

    def list_models(self) -> list[str]:
        try:
            r = httpx.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=min(self.timeout, 5.0),
            )
            if r.status_code >= 400:
                return []
            payload = r.json()
        except Exception as e:
            logger.warning("OpenAI-compat /v1/models failed: %s", e)
            return []

        # /v1/models response shape: {"data": [{"id": "...", ...}], ...}
        data = payload.get("data") or []
        return [
            m["id"]
            for m in data
            if isinstance(m, dict) and isinstance(m.get("id"), str)
        ]

    # ---- chat --------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling: SamplingConfig,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        if sampling.top_k is not None:
            payload["top_k"] = sampling.top_k
        if sampling.max_tokens is not None:
            payload["max_tokens"] = sampling.max_tokens
        if sampling.seed is not None:
            payload["seed"] = sampling.seed

        # Pass-through for caller-specific extensions.
        for key in ("tools", "tool_choice", "response_format", "stop"):
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                timeout=self.timeout,
            )
        except Exception as e:
            raise RuntimeError(
                f"OpenAI-compat POST /chat/completions failed: {e}"
            ) from e

        if r.status_code >= 400:
            raise RuntimeError(
                f"OpenAI-compat /chat/completions returned "
                f"{r.status_code}: {r.text[:200]}"
            )

        try:
            body = r.json()
        except ValueError as e:
            raise RuntimeError(
                f"OpenAI-compat response was not JSON: {r.text[:200]}"
            ) from e

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"OpenAI-compat response had no choices: {body}"
            )
        message = choices[0].get("message") or {}
        raw_content = message.get("content") or ""

        cleaned, thinking = strip_thinking(raw_content)

        extra: dict[str, Any] = {}
        if message.get("tool_calls"):
            extra["tool_calls"] = message["tool_calls"]
        if body.get("usage"):
            extra["usage"] = body["usage"]

        return LLMResponse(
            content=cleaned,
            model=model,
            backend=self.name,
            sampling_config=sampling.as_dict(),
            thinking=thinking,
            extra=extra,
        )

    # ---- CrewAI bridge -----------------------------------------------

    def make_crewai_llm(self, model: str, sampling: SamplingConfig) -> Any:
        """CrewAI natively supports OpenAI-compat local servers via
        the `hosted_vllm` provider tag. That route covers vLLM, LM
        Studio, mlx_lm.server, and Ollama's `/v1` endpoint without
        requiring LiteLLM as an extra.
        """
        from crewai import LLM

        kwargs: dict[str, Any] = {
            "model": f"hosted_vllm/{model}",
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": sampling.temperature,
        }
        if sampling.seed is not None:
            kwargs["seed"] = sampling.seed
        if sampling.top_p is not None:
            kwargs["top_p"] = sampling.top_p
        return LLM(**kwargs)


__all__ = ["OpenAICompatBackend"]
