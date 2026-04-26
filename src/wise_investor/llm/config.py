"""Load per-agent model + sampling config from `config/agent_models.yaml`.

The YAML file is optional — if it's missing or malformed we fall
back to the legacy `.env`-driven mapping so the existing deployment
keeps working untouched. Precedence (highest first):

  1. `agents.<name>.backends.<backend>`
  2. `agents.<name>.model`
  3. `defaults.backends.<backend>`
  4. `defaults.model`
  5. Legacy `.env` (settings.analyst_model, valuer_model, …)

Sampling resolution follows the same tree, but when nothing is
specified the resolved model is fed to
`wise_investor.llm.utils.sampling.get_recommended_sampling` so each
agent gets the model-author-published best-practice settings.

Usage:

    from wise_investor.llm import get_backend
    from wise_investor.llm.config import get_agent_config

    backend = get_backend()
    cfg = get_agent_config("skeptic", backend=backend.name)
    response = backend.chat(
        messages=[…], model=cfg.model, sampling=cfg.sampling
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wise_investor.config import PROJECT_ROOT
from wise_investor.llm.base import SamplingConfig
from wise_investor.llm.utils.sampling import get_recommended_sampling


logger = logging.getLogger(__name__)


DEFAULT_YAML_PATH = PROJECT_ROOT / "config" / "agent_models.yaml"


# Legacy fallback: which `settings` attribute backs which agent when
# neither the YAML nor an agent-specific field is set. Mirrors the
# pre-Phase-4 runner.py aliases (economist + defender share the
# analyst model; skeptic + steward have their own env vars).
_LEGACY_SETTINGS_ATTR: dict[str, str] = {
    "economist": "analyst_model",
    "analyst": "analyst_model",
    "valuer": "valuer_model",
    "skeptic": "skeptic_model",
    "defender": "analyst_model",
    "steward": "steward_model",
}


@dataclass
class AgentConfig:
    """Fully resolved settings for one agent on a given backend."""

    agent: str
    model: str
    sampling: SamplingConfig
    # Where the model came from, for audit / meta.txt recording:
    #   "yaml:agents.<name>.backends.<backend>"
    #   "yaml:agents.<name>.model"
    #   "yaml:defaults.backends.<backend>"
    #   "yaml:defaults.model"
    #   "legacy:.env"
    source: str = "legacy:.env"


_cached_yaml: dict[str, Any] | None = None
_cached_yaml_path: Path | None = None


def _load_yaml(path: Path | None = None) -> dict[str, Any]:
    """Read and cache `config/agent_models.yaml`. Returns {} on any
    failure (missing file, parse error) so callers silently fall back
    to the legacy path.
    """
    global _cached_yaml, _cached_yaml_path
    target = path if path is not None else DEFAULT_YAML_PATH

    if _cached_yaml is not None and _cached_yaml_path == target:
        return _cached_yaml

    if not target.exists():
        _cached_yaml = {}
        _cached_yaml_path = target
        return {}

    try:
        import yaml
    except ImportError as e:  # pragma: no cover — pyyaml is in core
        logger.warning("pyyaml unavailable (%s); using legacy config path.", e)
        _cached_yaml = {}
        _cached_yaml_path = target
        return {}

    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            logger.warning(
                "agent_models.yaml top level must be a mapping; got %s. "
                "Falling back to legacy config.",
                type(data).__name__,
            )
            data = {}
    except Exception as e:
        logger.warning(
            "agent_models.yaml parse error (%s); using legacy config.", e
        )
        data = {}

    _cached_yaml = data
    _cached_yaml_path = target
    return data


def _reset_cache_for_test() -> None:
    """Testing hook: drop the cached YAML so the next load re-reads disk."""
    global _cached_yaml, _cached_yaml_path
    _cached_yaml = None
    _cached_yaml_path = None


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------


def _resolve_model(
    agent: str,
    backend: str,
    yaml_data: dict[str, Any],
) -> tuple[str, str]:
    """Return (model, source_tag)."""
    agent_cfg = (yaml_data.get("agents") or {}).get(agent) or {}
    defaults = yaml_data.get("defaults") or {}

    agent_backends = agent_cfg.get("backends") or {}
    if isinstance(agent_backends, dict) and agent_backends.get(backend):
        return str(agent_backends[backend]), f"yaml:agents.{agent}.backends.{backend}"

    if agent_cfg.get("model"):
        return str(agent_cfg["model"]), f"yaml:agents.{agent}.model"

    default_backends = defaults.get("backends") or {}
    if isinstance(default_backends, dict) and default_backends.get(backend):
        return str(default_backends[backend]), f"yaml:defaults.backends.{backend}"

    if defaults.get("model"):
        return str(defaults["model"]), "yaml:defaults.model"

    return _legacy_model(agent), "legacy:.env"


def _legacy_model(agent: str) -> str:
    """Fall back to the `.env`-backed settings for this agent."""
    attr = _LEGACY_SETTINGS_ATTR.get(agent, "analyst_model")
    from wise_investor.config import settings

    return getattr(settings, attr)


# ---------------------------------------------------------------------------
# Sampling resolution
# ---------------------------------------------------------------------------


_SAMPLING_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "max_tokens",
        "enable_thinking",
        "seed",
    }
)


def _sampling_from_dict(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Extract only known SamplingConfig fields from `raw`; ignore
    unknown keys silently so future YAML extensions don't crash the
    loader.
    """
    if not isinstance(raw, dict):
        return {}
    return {
        k: v for k, v in raw.items()
        if k in _SAMPLING_FIELD_NAMES and v is not None
    }


def _resolve_sampling(
    agent: str,
    model: str,
    yaml_data: dict[str, Any],
) -> SamplingConfig:
    """Produce an effective SamplingConfig.

    Starts from the model-family recommendation (so thinking-mode
    models get their proper top_p/min_p, qwen2.5 gets 0.7/0.8, etc.)
    and layers YAML defaults then agent-specific overrides on top.
    """
    agent_cfg = (yaml_data.get("agents") or {}).get(agent) or {}
    defaults = yaml_data.get("defaults") or {}

    default_sampling = _sampling_from_dict(defaults.get("sampling"))
    agent_sampling = _sampling_from_dict(agent_cfg.get("sampling"))

    enable_thinking_hint = agent_sampling.get(
        "enable_thinking", default_sampling.get("enable_thinking")
    )

    base = get_recommended_sampling(model, enable_thinking=enable_thinking_hint)

    # Overlay: defaults first, then agent overrides.
    for key, value in default_sampling.items():
        setattr(base, key, value)
    for key, value in agent_sampling.items():
        setattr(base, key, value)

    return base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_agent_config(
    agent: str,
    backend: str = "ollama",
    yaml_path: Path | None = None,
) -> AgentConfig:
    """Resolve the effective `AgentConfig` for one agent.

    Args:
        agent: agent name (economist / analyst / valuer / skeptic /
            defender / steward). Unknown names still resolve via the
            `defaults.model` fallback — useful for ad-hoc tools like
            the classifier, translator, onboarding brief generator.
        backend: backend short name (ollama / mlx / openai_compat /
            llamacpp). Used for backend-specific model overrides.
        yaml_path: override for tests; None uses the package default.
    """
    yaml_data = _load_yaml(yaml_path)
    model, source = _resolve_model(agent, backend, yaml_data)
    sampling = _resolve_sampling(agent, model, yaml_data)
    return AgentConfig(agent=agent, model=model, sampling=sampling, source=source)


__all__ = [
    "AgentConfig",
    "DEFAULT_YAML_PATH",
    "get_agent_config",
]
