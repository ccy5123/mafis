"""MLX backend — in-process Apple Silicon inference via mlx-lm.

Only usable on Apple Silicon (mlx is Apple-framework-backed).
On every other platform (Windows / Linux / Intel Mac) importing
`mlx_lm` raises ImportError; we catch that at instantiation time
and surface it as a runtime error with a clear message. The
factory still lets the module be imported so `LLM_BACKEND=mlx`
fails *loudly* rather than silently falling back.

No CrewAI bridge — MLX has no LiteLLM provider. Agents that need
MLX on Apple Silicon call `chat()` directly. Users who want MLX
behind a CrewAI path should run `mlx_lm.server …` and point the
`openai_compat` backend at it.

Environment:
  MLX_MODEL_CACHE_DIR   optional — passed through to mlx_lm.load.
                        Defaults to the user's HuggingFace cache.

Notes:
  - mlx-lm loads the weights once and keeps them in memory; we
    cache model+tokenizer pairs per-backend-instance to avoid
    re-loading across calls.
  - generate() takes a prompt string, so we render messages via
    the tokenizer's chat template.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from wise_investor.llm.base import LLMBackend, LLMResponse, SamplingConfig
from wise_investor.llm.utils.thinking import strip_thinking


logger = logging.getLogger(__name__)


class MLXBackend(LLMBackend):
    """Apple Silicon local inference."""

    def __init__(self, cache_dir: str | None = None) -> None:
        try:
            import mlx_lm  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "MLX backend requires mlx-lm, which only runs on Apple "
                "Silicon. Install via `pip install mlx-lm` on an M-series "
                "Mac, or switch to LLM_BACKEND=ollama. Original error: "
                f"{e}"
            ) from e

        self.cache_dir = cache_dir or os.getenv("MLX_MODEL_CACHE_DIR")
        # (model_id -> (model, tokenizer)) — keyed by HF repo / path.
        self._loaded: dict[str, tuple[Any, Any]] = {}

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        # Reaching __init__ without raising means mlx_lm imported fine
        # and the hardware supports Metal. No further probe needed.
        return True

    def list_models(self) -> list[str]:
        """Return cached model ids only. Enumerating the HF cache dir
        would be misleading — users often have many repos cached that
        aren't valid MLX model directories. Report what we've actually
        loaded this session.
        """
        return list(self._loaded.keys())

    # ---- model cache --------------------------------------------------

    def _get_or_load(self, model: str) -> tuple[Any, Any]:
        if model in self._loaded:
            return self._loaded[model]

        from mlx_lm import load

        logger.info("MLX loading model %s (first call is slow)", model)
        tokenizer_config: dict[str, Any] = {}
        model_path, tokenizer = load(model, tokenizer_config=tokenizer_config)
        self._loaded[model] = (model_path, tokenizer)
        return self._loaded[model]

    # ---- chat ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling: SamplingConfig,
        **kwargs: Any,
    ) -> LLMResponse:
        from mlx_lm import generate

        mlx_model, tokenizer = self._get_or_load(model)

        # Render messages into the model's chat template. Every Qwen /
        # Llama / DeepSeek tokenizer on HF carries an `apply_chat_
        # template` method; we assume it's present (mlx-lm itself
        # relies on this).
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # mlx-lm's generate() takes positional args (model, tokenizer,
        # prompt) plus kwargs. Only pass what mlx-lm accepts so future
        # API changes don't silently break.
        gen_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": sampling.max_tokens or 2048,
            "verbose": False,
        }
        # mlx-lm uses `temp` not `temperature`; top_p is native.
        if sampling.temperature is not None:
            gen_kwargs["temp"] = sampling.temperature
        if sampling.top_p is not None:
            gen_kwargs["top_p"] = sampling.top_p

        raw = generate(mlx_model, tokenizer, **gen_kwargs)

        cleaned, thinking = strip_thinking(raw or "")

        return LLMResponse(
            content=cleaned,
            model=model,
            backend=self.name,
            sampling_config=sampling.as_dict(),
            thinking=thinking,
        )


__all__ = ["MLXBackend"]
