"""llama.cpp (llama-cpp-python) backend.

In-process GGUF inference. Works on every OS; needs CUDA / Metal
build flags to use GPU. Users install via:
  pip install "mafis[llamacpp]"                    # CPU-only
  CMAKE_ARGS="-DLLAMA_CUDA=on" pip install …       # NVIDIA GPU
  CMAKE_ARGS="-DLLAMA_METAL=on" pip install …      # Apple Silicon

Environment:
  LLAMACPP_MODEL_PATH    required at runtime — path to a .gguf file.
  LLAMACPP_N_GPU_LAYERS  optional, default -1 (all layers on GPU if
                         built with GPU support; 0 for CPU-only).
  LLAMACPP_N_CTX         optional, default 8192.

No CrewAI bridge — llama.cpp has no LiteLLM provider for the in-
process route. Users who want llama.cpp behind a CrewAI path run
`llama-cpp-python[server]` and point `openai_compat` at `/v1`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from wise_investor.llm.base import LLMBackend, LLMResponse, SamplingConfig
from wise_investor.llm.utils.thinking import strip_thinking


logger = logging.getLogger(__name__)


class LlamaCppBackend(LLMBackend):
    """GGUF inference via llama-cpp-python."""

    def __init__(
        self,
        model_path: str | None = None,
        n_gpu_layers: int | None = None,
        n_ctx: int | None = None,
    ) -> None:
        try:
            import llama_cpp  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "llamacpp backend requires llama-cpp-python. Install via "
                "`pip install mafis[llamacpp]` (CPU-only by default; see "
                "docs/llm_backends.md for GPU build flags). Original "
                f"error: {e}"
            ) from e

        resolved_path = (
            model_path
            if model_path is not None
            else os.getenv("LLAMACPP_MODEL_PATH")
        )
        if not resolved_path:
            raise ValueError(
                "LlamaCppBackend requires a GGUF model path. Set "
                "LLAMACPP_MODEL_PATH in .env or pass model_path."
            )
        self.model_path = Path(resolved_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GGUF file not found: {self.model_path}. Download a "
                "GGUF quantization (e.g. from TheBloke on HuggingFace) "
                "and set LLAMACPP_MODEL_PATH to point at it."
            )

        # GPU offload — -1 means "all layers the build supports"; 0
        # forces CPU. Unbuilt-for-GPU wheels ignore this flag.
        if n_gpu_layers is None:
            env_val = os.getenv("LLAMACPP_N_GPU_LAYERS")
            n_gpu_layers = int(env_val) if env_val is not None else -1
        self.n_gpu_layers = n_gpu_layers

        if n_ctx is None:
            env_val = os.getenv("LLAMACPP_N_CTX")
            n_ctx = int(env_val) if env_val is not None else 8192
        self.n_ctx = n_ctx

        # Model is lazily instantiated on the first chat() call so
        # just constructing the backend (e.g. for is_available checks)
        # doesn't pay the 2-5s load cost.
        self._llm: Any = None

    @property
    def name(self) -> str:
        return "llamacpp"

    def is_available(self) -> bool:
        # Successful construction implies the lib imported and the
        # GGUF file exists. Actual generation readiness requires a
        # load; we don't trigger that here to keep the probe cheap.
        return self.model_path.exists()

    def list_models(self) -> list[str]:
        """Single model per backend instance (the configured GGUF).
        Return its filename as a stable identifier.
        """
        return [self.model_path.name]

    def _ensure_loaded(self) -> Any:
        if self._llm is not None:
            return self._llm
        from llama_cpp import Llama

        logger.info(
            "llama.cpp loading %s (n_gpu_layers=%d, n_ctx=%d)",
            self.model_path.name,
            self.n_gpu_layers,
            self.n_ctx,
        )
        self._llm = Llama(
            model_path=str(self.model_path),
            n_gpu_layers=self.n_gpu_layers,
            n_ctx=self.n_ctx,
            verbose=False,
        )
        return self._llm

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling: SamplingConfig,
        **kwargs: Any,
    ) -> LLMResponse:
        # llama-cpp-python binds one model per Llama instance. `model`
        # is accepted for API parity with other backends but must
        # match the configured GGUF; mismatches are a caller bug.
        if model and model != self.model_path.name and model != str(self.model_path):
            logger.warning(
                "LlamaCppBackend loaded %s but caller requested %s — "
                "using the loaded model; reconfigure LLAMACPP_MODEL_PATH "
                "if you need a different GGUF.",
                self.model_path.name,
                model,
            )

        llm = self._ensure_loaded()

        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
        }
        if sampling.top_k is not None:
            call_kwargs["top_k"] = sampling.top_k
        if sampling.min_p is not None:
            call_kwargs["min_p"] = sampling.min_p
        if sampling.max_tokens is not None:
            call_kwargs["max_tokens"] = sampling.max_tokens
        if sampling.seed is not None:
            call_kwargs["seed"] = sampling.seed

        # Forward supported chat extras.
        for key in ("tools", "tool_choice", "response_format", "stop"):
            if key in kwargs:
                call_kwargs[key] = kwargs[key]

        resp = llm.create_chat_completion(**call_kwargs)

        choices = resp.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"llama.cpp returned no choices: {resp}"
            )
        message = choices[0].get("message") or {}
        raw_content = message.get("content") or ""

        cleaned, thinking = strip_thinking(raw_content)

        extra: dict[str, Any] = {}
        if message.get("tool_calls"):
            extra["tool_calls"] = message["tool_calls"]
        if resp.get("usage"):
            extra["usage"] = resp["usage"]

        return LLMResponse(
            content=cleaned,
            model=self.model_path.name,
            backend=self.name,
            sampling_config=sampling.as_dict(),
            thinking=thinking,
            extra=extra,
        )


__all__ = ["LlamaCppBackend"]
