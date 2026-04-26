"""Per-agent config resolution (wise_investor.llm.config)."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.llm import config as config_module
from wise_investor.llm.config import AgentConfig, get_agent_config


@pytest.fixture(autouse=True)
def _reset_cache():
    config_module._reset_cache_for_test()
    yield
    config_module._reset_cache_for_test()


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agent_models.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# YAML-less path: legacy .env fallback
# ---------------------------------------------------------------------------


def test_legacy_fallback_when_yaml_missing(tmp_path) -> None:
    """No YAML file → resolve each agent against the existing
    settings.* attributes. Matches pre-Phase-4 behavior so a user
    upgrading MAFIS with no config file keeps current model routing.
    """
    missing = tmp_path / "nope.yaml"

    cfg = get_agent_config("analyst", backend="ollama", yaml_path=missing)
    assert isinstance(cfg, AgentConfig)
    assert cfg.source == "legacy:.env"
    # settings.analyst_model default is "llama3.1:8b" on a fresh repo.
    assert cfg.model


def test_legacy_fallback_economist_aliases_to_analyst() -> None:
    """Pre-Phase-4 runner.py used settings.analyst_model for Economist
    (no separate economist_model env var existed). Loader preserves this.
    """
    from wise_investor.config import settings

    cfg = get_agent_config(
        "economist", backend="ollama", yaml_path=Path("/does/not/exist")
    )
    assert cfg.model == settings.analyst_model


def test_legacy_fallback_defender_aliases_to_analyst() -> None:
    from wise_investor.config import settings

    cfg = get_agent_config(
        "defender", backend="ollama", yaml_path=Path("/does/not/exist")
    )
    assert cfg.model == settings.analyst_model


def test_legacy_fallback_skeptic_uses_skeptic_model() -> None:
    from wise_investor.config import settings

    cfg = get_agent_config(
        "skeptic", backend="ollama", yaml_path=Path("/does/not/exist")
    )
    assert cfg.model == settings.skeptic_model


# ---------------------------------------------------------------------------
# Precedence: defaults → agent override → backend override
# ---------------------------------------------------------------------------


def test_defaults_model_picked_up_when_agent_omitted(tmp_path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: custom-default-model
agents:
  analyst:
""",
    )
    cfg = get_agent_config("analyst", backend="ollama", yaml_path=yaml_path)
    assert cfg.model == "custom-default-model"
    assert cfg.source == "yaml:defaults.model"


def test_agent_model_overrides_defaults(tmp_path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: global-default
agents:
  skeptic:
    model: skeptic-only
""",
    )
    cfg = get_agent_config("skeptic", backend="ollama", yaml_path=yaml_path)
    assert cfg.model == "skeptic-only"
    assert cfg.source == "yaml:agents.skeptic.model"


def test_backend_override_on_agent_wins(tmp_path) -> None:
    """agents.<n>.backends.<b> beats agents.<n>.model."""
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: global-default
agents:
  skeptic:
    model: skeptic-ollama
    backends:
      mlx: "mlx-community/Skeptic-MLX-4bit"
      openai_compat: "vendor/Skeptic-hosted"
""",
    )
    cfg_ollama = get_agent_config("skeptic", backend="ollama", yaml_path=yaml_path)
    assert cfg_ollama.model == "skeptic-ollama"

    cfg_mlx = get_agent_config("skeptic", backend="mlx", yaml_path=yaml_path)
    assert cfg_mlx.model == "mlx-community/Skeptic-MLX-4bit"
    assert cfg_mlx.source == "yaml:agents.skeptic.backends.mlx"

    cfg_oc = get_agent_config(
        "skeptic", backend="openai_compat", yaml_path=yaml_path
    )
    assert cfg_oc.model == "vendor/Skeptic-hosted"


def test_defaults_backends_applies_when_agent_missing(tmp_path) -> None:
    """defaults.backends.<b> activates for any agent that doesn't
    explicitly override.
    """
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: global-default
  backends:
    openai_compat: "vendor/default-hosted"
agents:
  analyst:
""",
    )
    cfg = get_agent_config(
        "analyst", backend="openai_compat", yaml_path=yaml_path
    )
    assert cfg.model == "vendor/default-hosted"
    assert cfg.source == "yaml:defaults.backends.openai_compat"

    # When a backend without a default is requested, fall back to
    # defaults.model.
    cfg_ollama = get_agent_config(
        "analyst", backend="ollama", yaml_path=yaml_path
    )
    assert cfg_ollama.model == "global-default"


# ---------------------------------------------------------------------------
# Sampling resolution
# ---------------------------------------------------------------------------


def test_sampling_falls_back_to_recommended_when_not_overridden(tmp_path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: qwen2.5:7b
agents:
  skeptic:
""",
    )
    cfg = get_agent_config("skeptic", backend="ollama", yaml_path=yaml_path)
    # Qwen2.5 recommended: temperature 0.7, top_p 0.8.
    assert cfg.sampling.temperature == 0.7
    assert cfg.sampling.top_p == 0.8
    assert cfg.sampling.enable_thinking is False


def test_sampling_agent_override_wins_over_defaults(tmp_path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: qwen2.5:7b
  sampling:
    temperature: 0.5
agents:
  steward:
    sampling:
      temperature: 0.0
      seed: 42
""",
    )
    # Steward overrides both defaults and the model recommendation.
    cfg_steward = get_agent_config(
        "steward", backend="ollama", yaml_path=yaml_path
    )
    assert cfg_steward.sampling.temperature == 0.0
    assert cfg_steward.sampling.seed == 42

    # Other agents inherit defaults.sampling on top of the recommendation.
    cfg_analyst = get_agent_config(
        "analyst", backend="ollama", yaml_path=yaml_path
    )
    assert cfg_analyst.sampling.temperature == 0.5
    assert cfg_analyst.sampling.seed is None


def test_enable_thinking_hint_selects_qwen3_variant(tmp_path) -> None:
    """When an agent explicitly opts into thinking mode AND the model
    is Qwen3, the recommended config should flip to the thinking
    variant (temp 0.6, top_p 0.95, min_p 0) before agent overrides.
    """
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: qwen3:14b
agents:
  skeptic:
    sampling:
      enable_thinking: true
""",
    )
    cfg = get_agent_config("skeptic", backend="ollama", yaml_path=yaml_path)
    assert cfg.sampling.temperature == 0.6
    assert cfg.sampling.top_p == 0.95
    assert cfg.sampling.min_p == 0.0
    assert cfg.sampling.enable_thinking is True


def test_unknown_sampling_field_ignored_silently(tmp_path) -> None:
    """If a user adds a typo or a future field, the loader must not
    crash — it simply drops unknown keys.
    """
    yaml_path = _write_yaml(
        tmp_path,
        """
defaults:
  model: qwen2.5:7b
agents:
  analyst:
    sampling:
      temperature: 0.9
      future_unknown_knob: "value"
      blorp: 42
""",
    )
    cfg = get_agent_config("analyst", backend="ollama", yaml_path=yaml_path)
    assert cfg.sampling.temperature == 0.9
    assert not hasattr(cfg.sampling, "future_unknown_knob")
    assert not hasattr(cfg.sampling, "blorp")


# ---------------------------------------------------------------------------
# Robustness — malformed YAML falls back cleanly
# ---------------------------------------------------------------------------


def test_malformed_yaml_falls_back_to_legacy(tmp_path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        "{ this is not: valid yaml [\n",
    )
    cfg = get_agent_config("analyst", backend="ollama", yaml_path=yaml_path)
    assert cfg.source == "legacy:.env"


def test_yaml_that_is_a_list_falls_back_to_legacy(tmp_path) -> None:
    """Top-level must be a mapping. A list / string falls through."""
    yaml_path = _write_yaml(tmp_path, "- not\n- a\n- mapping\n")
    cfg = get_agent_config("analyst", backend="ollama", yaml_path=yaml_path)
    assert cfg.source == "legacy:.env"


def test_empty_yaml_falls_back_to_legacy(tmp_path) -> None:
    yaml_path = _write_yaml(tmp_path, "")
    cfg = get_agent_config("analyst", backend="ollama", yaml_path=yaml_path)
    assert cfg.source == "legacy:.env"


# ---------------------------------------------------------------------------
# Bundled seed file sanity
# ---------------------------------------------------------------------------


def test_bundled_seed_yaml_resolves_every_agent() -> None:
    """The in-repo config/agent_models.yaml must cover every agent the
    runner knows about. If this breaks, a Phase 5 migrated agent would
    blow up at runtime looking up its model.
    """
    from wise_investor.llm.config import DEFAULT_YAML_PATH

    assert DEFAULT_YAML_PATH.exists(), "Seed YAML missing from repo"

    for agent in (
        "economist",
        "analyst",
        "valuer",
        "skeptic",
        "defender",
        "steward",
    ):
        cfg = get_agent_config(agent, backend="ollama")
        assert cfg.model, f"{agent} did not resolve a model"
        assert cfg.sampling.temperature > 0
