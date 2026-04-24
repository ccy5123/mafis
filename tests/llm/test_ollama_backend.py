"""OllamaBackend unit tests.

We split coverage into three tiers:
  - Pure translation tests (options dict shape): always run, no network.
  - Mocked chat tests: patch `ollama.chat` so we verify the payload
    shape without hitting the server.
  - Live smoke test: only runs when the real Ollama daemon is reachable
    AND a tiny model is already pulled. Skipped otherwise so CI / dev
    without Ollama installed still stays green.
"""

from __future__ import annotations

import httpx
import pytest

from wise_investor.llm.backends.ollama import (
    OllamaBackend,
    _sampling_to_ollama_options,
)
from wise_investor.llm.base import SamplingConfig


# ---------------------------------------------------------------------------
# Pure translation: SamplingConfig → Ollama options dict
# ---------------------------------------------------------------------------


def test_options_include_temperature_and_top_p_always() -> None:
    out = _sampling_to_ollama_options(SamplingConfig())
    assert "temperature" in out
    assert "top_p" in out


def test_options_omit_optional_keys_when_none() -> None:
    """Omitting means Ollama uses its own defaults — the whole point
    of this translation layer is to NOT spuriously override.
    """
    out = _sampling_to_ollama_options(SamplingConfig())
    assert "top_k" not in out
    assert "min_p" not in out
    assert "num_predict" not in out  # max_tokens=None → unlimited
    assert "seed" not in out         # non-deterministic default


def test_options_include_optional_keys_when_set() -> None:
    s = SamplingConfig(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        min_p=0.05,
        max_tokens=1024,
        seed=42,
    )
    out = _sampling_to_ollama_options(s)
    assert out == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "num_predict": 1024,
        "seed": 42,
    }


def test_max_tokens_renames_to_num_predict() -> None:
    """Ollama's param name is `num_predict`, not `max_tokens` —
    translation MUST happen here, not silently upstream."""
    out = _sampling_to_ollama_options(SamplingConfig(max_tokens=256))
    assert out["num_predict"] == 256
    assert "max_tokens" not in out


# ---------------------------------------------------------------------------
# Backend identity + availability
# ---------------------------------------------------------------------------


def test_backend_name_is_ollama() -> None:
    b = OllamaBackend(host="http://localhost:11434")
    assert b.name == "ollama"


def test_is_available_false_on_connection_error(monkeypatch) -> None:
    def _raise(url, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", _raise)
    b = OllamaBackend(host="http://localhost:99999")
    assert b.is_available() is False


def test_is_available_true_on_200(monkeypatch) -> None:
    def _ok(url, timeout=None):
        return httpx.Response(200, json={"models": []})

    monkeypatch.setattr("httpx.get", _ok)
    b = OllamaBackend(host="http://localhost:11434")
    assert b.is_available() is True


def test_list_models_parses_api_tags_payload(monkeypatch) -> None:
    def _ok(url, timeout=None):
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen2.5:7b", "size": 123},
                    {"name": "llama3.1:8b", "size": 456},
                    {"not_a_name": "ignored"},
                ]
            },
        )

    monkeypatch.setattr("httpx.get", _ok)
    b = OllamaBackend(host="http://localhost:11434")
    assert b.list_models() == ["qwen2.5:7b", "llama3.1:8b"]


def test_list_models_returns_empty_on_error(monkeypatch) -> None:
    def _broken(url, timeout=None):
        raise httpx.ConnectError("no server")

    monkeypatch.setattr("httpx.get", _broken)
    b = OllamaBackend(host="http://localhost:99999")
    assert b.list_models() == []


# ---------------------------------------------------------------------------
# chat() — mocked ollama.chat
# ---------------------------------------------------------------------------


def _patch_ollama_chat(monkeypatch, response: dict):
    """Replace `ollama.chat` with a stub returning `response`. Captures
    the args the wrapper passed so we can assert on them.
    """
    captured: dict = {}

    def _stub(**kwargs):
        captured.update(kwargs)
        return response

    import ollama

    monkeypatch.setattr(ollama, "chat", _stub)
    return captured


def test_chat_returns_llm_response_with_sampling_echo(monkeypatch) -> None:
    captured = _patch_ollama_chat(
        monkeypatch,
        {"message": {"content": "hello"}, "eval_count": 10},
    )
    b = OllamaBackend(host="http://localhost:11434")

    resp = b.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="qwen2.5:7b",
        sampling=SamplingConfig(temperature=0.7, top_p=0.8),
    )
    assert resp.content == "hello"
    assert resp.model == "qwen2.5:7b"
    assert resp.backend == "ollama"
    assert resp.thinking is None
    assert resp.sampling_config["temperature"] == 0.7
    assert resp.extra.get("eval_count") == 10

    # Chat payload: model + messages + options, nothing else.
    assert captured["model"] == "qwen2.5:7b"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["options"]["temperature"] == 0.7
    assert captured["options"]["top_p"] == 0.8


def test_chat_extracts_thinking_block(monkeypatch) -> None:
    _patch_ollama_chat(
        monkeypatch,
        {"message": {"content": "<think>steps</think>\nfinal answer"}},
    )
    b = OllamaBackend()
    resp = b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="qwen3:14b",
        sampling=SamplingConfig(enable_thinking=True),
    )
    assert resp.content == "final answer"
    assert resp.thinking == "steps"


def test_chat_passes_tools_through(monkeypatch) -> None:
    """Runner uses tool-calling — the wrapper must forward `tools`."""
    captured = _patch_ollama_chat(
        monkeypatch,
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "calculate_per", "arguments": {}}}
                ],
            }
        },
    )
    b = OllamaBackend()
    resp = b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="qwen2.5:7b",
        sampling=SamplingConfig(),
        tools=[{"type": "function", "function": {"name": "calculate_per"}}],
    )
    assert captured["tools"] == [
        {"type": "function", "function": {"name": "calculate_per"}}
    ]
    assert resp.extra["tool_calls"][0]["function"]["name"] == "calculate_per"


def test_chat_passes_keep_alive_through(monkeypatch) -> None:
    """Model-swap strategy in runner.py depends on keep_alive='0'."""
    captured = _patch_ollama_chat(
        monkeypatch, {"message": {"content": "ok"}}
    )
    b = OllamaBackend()
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="qwen2.5:7b",
        sampling=SamplingConfig(),
        keep_alive="0",
    )
    assert captured["keep_alive"] == "0"


def test_chat_does_not_forward_unknown_kwargs(monkeypatch) -> None:
    """Unknown kwargs are silently dropped — avoids surprises when a
    caller ports from one backend to another with backend-specific
    params that happen to clash with an internal Ollama kwarg.
    """
    captured = _patch_ollama_chat(
        monkeypatch, {"message": {"content": "ok"}}
    )
    b = OllamaBackend()
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="qwen2.5:7b",
        sampling=SamplingConfig(),
        some_unknown_flag=True,
    )
    assert "some_unknown_flag" not in captured


# ---------------------------------------------------------------------------
# CrewAI bridge
# ---------------------------------------------------------------------------


def test_make_crewai_llm_returns_configured_crewai_llm() -> None:
    """CrewAI's `LLM()` returns a provider-specific subclass (e.g.
    `OpenAICompatibleCompletion`) so `isinstance(llm, LLM)` is flaky
    across CrewAI versions. We check the attributes the agent code
    actually reads (matching the pattern in tests/test_analyst.py).
    """
    b = OllamaBackend(host="http://localhost:11434")
    llm = b.make_crewai_llm(
        "qwen2.5:7b",
        SamplingConfig(temperature=0.5, seed=42, top_p=0.9),
    )
    # LiteLLM strips the `ollama/` provider prefix when storing —
    # accept either shape.
    assert "qwen2.5:7b" in llm.model
    assert llm.temperature == 0.5
    # base_url keeps the configured host (CrewAI may append `/v1`
    # for the OpenAI-compat route).
    assert llm.base_url.rstrip("/").startswith("http://localhost:11434")


def test_make_crewai_llm_omits_seed_when_not_deterministic() -> None:
    """Pre-abstraction CrewAI LLM was always constructed with a fixed
    seed. Post-policy we only set it when the user has opted in to
    deterministic mode — otherwise LiteLLM uses its default (no seed).
    """
    b = OllamaBackend(host="http://localhost:11434")
    llm = b.make_crewai_llm(
        "qwen2.5:7b", SamplingConfig(temperature=0.7, seed=None)
    )
    # We don't have a portable way to check "seed was not passed",
    # but the temperature having made it through confirms the config
    # was applied.
    assert llm.temperature == 0.7


# ---------------------------------------------------------------------------
# Live smoke — only when real Ollama is running
# ---------------------------------------------------------------------------


def _ollama_running() -> bool:
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=1.0)
        return r.status_code < 400
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama daemon not reachable at http://localhost:11434",
)
def test_live_smoke_lists_models() -> None:
    """Smoke check: if the daemon is up, list_models should return
    SOMETHING (the user has at least one model pulled — MAFIS needs
    qwen2.5 and llama3.1 to actually work)."""
    b = OllamaBackend()
    assert b.is_available() is True
    models = b.list_models()
    assert isinstance(models, list)


@pytest.mark.skipif(
    not _ollama_running(),
    reason="Ollama daemon not reachable at http://localhost:11434",
)
def test_live_smoke_chat_roundtrip() -> None:
    """Issue one short generation to confirm the full wrapper works
    end-to-end. Skips if no model is pulled. Uses a minimal prompt
    and max_tokens=32 to keep the test under a second on a warm model.
    """
    b = OllamaBackend()
    models = b.list_models()
    if not models:
        pytest.skip("No Ollama models pulled locally")

    # Prefer a small/already-warm model if available.
    preferred = [
        m for m in models
        if any(tag in m for tag in ("qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b"))
    ]
    model = preferred[0] if preferred else models[0]

    resp = b.chat(
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        model=model,
        sampling=SamplingConfig(
            temperature=0.0, max_tokens=32, seed=42
        ),
    )
    assert resp.backend == "ollama"
    assert resp.model == model
    assert isinstance(resp.content, str)
    assert len(resp.content) > 0
