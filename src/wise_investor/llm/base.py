"""Abstract LLM backend interface + sampling / response dataclasses.

The spec source for this layer is the Phase 2 section of the
"LLM backend abstraction" directive (stored in project history):

- `SamplingConfig` — captures the non-reproducible sampling knobs a
  caller wants applied. Each backend maps these to its native API
  (Ollama uses `options`, OpenAI-compat uses top-level fields, MLX
  uses generator kwargs).

- `LLMResponse` — thinking-aware wrapper around the model's reply.
  The `sampling_config` field is deliberately a plain dict so it can
  round-trip into `meta.txt` without caring which backend produced it.

- `LLMBackend` — minimal contract every backend implements. The
  CrewAI bridge (`make_crewai_llm`) exists because some agents go
  through CrewAI's `LLM` class, which wraps LiteLLM — the backend
  gets to decide how to hand CrewAI a compatible handle instead of
  the agents poking at LiteLLM directly.

Reproducibility note:
  Historically MAFIS pinned `temperature=0, seed=42`. The new policy
  honors each model's recommended sampling (see
  `wise_investor.llm.utils.sampling`), so two runs of the same ticker
  can produce different outputs. Callers who need byte-identical runs
  set `SamplingConfig(temperature=0, seed=42)` explicitly in
  `config/agent_models.yaml` — the infrastructure supports it but
  no longer defaults to it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SamplingConfig:
    """Non-reproducible sampling knobs handed to a backend.

    Defaults are intentionally generic ("safe" mid-temperature). The
    concrete value for each agent comes from
    `wise_investor.llm.utils.sampling.get_recommended_sampling(model)`
    or from `config/agent_models.yaml`.

    Fields:
      temperature      — 0.0 disables sampling (greedy).
      top_p            — nucleus sampling cutoff.
      top_k            — optional top-k truncation; None lets the
                         backend use its native default.
      min_p            — Qwen3 / newer models' min-p; backend ignores
                         if unsupported.
      max_tokens       — hard cap on generation length. `None` means
                         "let the model/backend decide" (matches
                         Ollama's `num_predict=-1` default, which is
                         how pre-abstraction MAFIS runs).
      enable_thinking  — forwarded to backends that support a
                         visible reasoning trace (Qwen3 thinking
                         mode, DeepSeek-R1, etc.). Ignored otherwise.
      seed             — deterministic-mode override. None = random.
                         Legacy MAFIS used 42 everywhere; the new
                         default is None so callers opt in explicitly.
    """

    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    enable_thinking: bool = False
    seed: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for `meta.txt` tracking. Keeps None fields so
        reviewers can see "top_k was not set" vs "top_k=0"."""
        return asdict(self)


@dataclass
class LLMResponse:
    """Normalized response shape across backends."""

    content: str                          # thinking block stripped if any
    model: str                            # actual model identifier used
    backend: str                          # "ollama" | "mlx" | ...
    sampling_config: dict[str, Any]       # what the backend actually received
    thinking: str | None = None           # original <think> body, if any
    extra: dict[str, Any] = field(default_factory=dict)


class LLMBackend(ABC):
    """One concrete backend (one runtime / server).

    Backends are stateless wrt. conversations — agents pass the full
    message history on every call. Backend objects own connection
    configuration (host, API key, model cache dir) read from env.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in `LLMResponse.backend` and logs."""

    @abstractmethod
    def is_available(self) -> bool:
        """True when the backend can serve requests right now.

        Checks must be cheap (no generation). Ollama hits /api/tags,
        OpenAI-compat hits /v1/models, MLX checks import + cache dir.
        """

    @abstractmethod
    def list_models(self) -> list[str]:
        """Models the backend can currently serve without downloading.

        Empty list is a valid answer for runtimes that auto-pull (MLX).
        """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str,
        sampling: SamplingConfig,
        **kwargs: Any,
    ) -> LLMResponse:
        """Run a single chat completion.

        `messages` follows the OpenAI / Ollama schema:
          [{"role": "system|user|assistant|tool", "content": "..."}, ...]

        `kwargs` is open-ended so callers can pass backend-specific
        features (Ollama's `tools`, `keep_alive`; OpenAI-compat's
        `response_format`; etc.) without fattening the contract.
        Backends that don't understand a kwarg should ignore it (not
        raise) so callers can write portable code.
        """

    def make_crewai_llm(self, model: str, sampling: SamplingConfig) -> Any:
        """Return a CrewAI-compatible `LLM` object for this backend.

        Optional — default raises NotImplementedError so backends that
        don't plug into CrewAI (or don't have a LiteLLM provider yet)
        can signal that explicitly. Agents falling back to direct
        `chat()` use a different code path.
        """
        raise NotImplementedError(
            f"{self.name} backend does not provide a CrewAI LLM handle. "
            "Use `chat()` directly or pick a supported backend."
        )


__all__ = ["LLMBackend", "LLMResponse", "SamplingConfig"]
