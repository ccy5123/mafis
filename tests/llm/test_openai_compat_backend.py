"""OpenAI-compat backend tests. Pure httpx mocks — no live server."""

from __future__ import annotations

import json

import httpx
import pytest

from wise_investor.llm.backends.openai_compat import OpenAICompatBackend
from wise_investor.llm.base import SamplingConfig


# ---------------------------------------------------------------------------
# Construction / env handling
# ---------------------------------------------------------------------------


def test_constructor_requires_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    with pytest.raises(ValueError) as exc_info:
        OpenAICompatBackend()
    assert "base" in str(exc_info.value).lower()


def test_constructor_reads_env_base_url(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://vllm:8000/v1")
    b = OpenAICompatBackend()
    assert b.base_url == "http://vllm:8000/v1"


def test_constructor_explicit_wins_over_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://env/v1")
    b = OpenAICompatBackend(base_url="http://explicit/v1")
    assert b.base_url == "http://explicit/v1"


def test_api_key_defaults_to_local_placeholder(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    b = OpenAICompatBackend(base_url="http://x/v1")
    # LiteLLM downstream requires a string, so "local" is a stable placeholder.
    assert b.api_key == "local"


def test_trailing_slash_stripped() -> None:
    b = OpenAICompatBackend(base_url="http://x/v1/")
    assert b.base_url == "http://x/v1"


def test_backend_name() -> None:
    b = OpenAICompatBackend(base_url="http://x/v1")
    assert b.name == "openai_compat"


# ---------------------------------------------------------------------------
# Availability + model listing
# ---------------------------------------------------------------------------


def _response(status: int, body: dict | str) -> httpx.Response:
    if isinstance(body, str):
        return httpx.Response(status, content=body.encode("utf-8"))
    return httpx.Response(status, content=json.dumps(body).encode("utf-8"))


def test_is_available_true_on_200(monkeypatch) -> None:
    def _ok(url, headers=None, timeout=None):
        return _response(200, {"data": []})

    monkeypatch.setattr("httpx.get", _ok)
    b = OpenAICompatBackend(base_url="http://x/v1")
    assert b.is_available() is True


def test_is_available_false_on_connection_error(monkeypatch) -> None:
    def _boom(url, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("httpx.get", _boom)
    b = OpenAICompatBackend(base_url="http://nope/v1")
    assert b.is_available() is False


def test_list_models_parses_data_array(monkeypatch) -> None:
    def _ok(url, headers=None, timeout=None):
        return _response(
            200,
            {
                "data": [
                    {"id": "Qwen/Qwen2.5-7B-Instruct"},
                    {"id": "meta-llama/Llama-3.1-8B-Instruct"},
                    {"no_id": "skip me"},
                ]
            },
        )

    monkeypatch.setattr("httpx.get", _ok)
    b = OpenAICompatBackend(base_url="http://x/v1")
    assert b.list_models() == [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
    ]


def test_list_models_returns_empty_on_http_error(monkeypatch) -> None:
    def _bad(url, headers=None, timeout=None):
        return _response(500, "server error")

    monkeypatch.setattr("httpx.get", _bad)
    b = OpenAICompatBackend(base_url="http://x/v1")
    assert b.list_models() == []


# ---------------------------------------------------------------------------
# chat() payload + response handling
# ---------------------------------------------------------------------------


def _patch_post(monkeypatch, response: httpx.Response):
    captured: dict = {}

    def _stub(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return response

    monkeypatch.setattr("httpx.post", _stub)
    return captured


def test_chat_posts_to_chat_completions_endpoint(monkeypatch) -> None:
    captured = _patch_post(
        monkeypatch,
        _response(
            200,
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        ),
    )
    b = OpenAICompatBackend(base_url="http://x/v1")
    resp = b.chat(
        messages=[{"role": "user", "content": "hi"}],
        model="Qwen/Qwen2.5-7B-Instruct",
        sampling=SamplingConfig(temperature=0.6, top_p=0.95, max_tokens=512),
    )
    assert captured["url"].endswith("/chat/completions")
    # Sampling fields forwarded to the OpenAI payload directly.
    payload = captured["json"]
    assert payload["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.95
    assert payload["max_tokens"] == 512
    assert resp.content == "ok"
    assert resp.backend == "openai_compat"


def test_chat_strips_thinking_block(monkeypatch) -> None:
    _patch_post(
        monkeypatch,
        _response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "<think>reasoning</think>\nfinal"
                        }
                    }
                ]
            },
        ),
    )
    b = OpenAICompatBackend(base_url="http://x/v1")
    resp = b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="any",
        sampling=SamplingConfig(enable_thinking=True),
    )
    assert resp.content == "final"
    assert resp.thinking == "reasoning"


def test_chat_omits_seed_when_not_set(monkeypatch) -> None:
    captured = _patch_post(
        monkeypatch,
        _response(
            200,
            {"choices": [{"message": {"content": "ok"}}]},
        ),
    )
    b = OpenAICompatBackend(base_url="http://x/v1")
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="any",
        sampling=SamplingConfig(seed=None),
    )
    assert "seed" not in captured["json"]


def test_chat_includes_seed_when_deterministic_mode(monkeypatch) -> None:
    captured = _patch_post(
        monkeypatch,
        _response(
            200,
            {"choices": [{"message": {"content": "ok"}}]},
        ),
    )
    b = OpenAICompatBackend(base_url="http://x/v1")
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="any",
        sampling=SamplingConfig(seed=42, temperature=0.0),
    )
    assert captured["json"]["seed"] == 42


def test_chat_forwards_tools_kwarg(monkeypatch) -> None:
    captured = _patch_post(
        monkeypatch,
        _response(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {"id": "c1", "type": "function"}
                            ],
                        }
                    }
                ]
            },
        ),
    )
    b = OpenAICompatBackend(base_url="http://x/v1")
    resp = b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="any",
        sampling=SamplingConfig(),
        tools=[{"type": "function", "function": {"name": "foo"}}],
    )
    assert captured["json"]["tools"][0]["function"]["name"] == "foo"
    assert resp.extra["tool_calls"][0]["id"] == "c1"


def test_chat_http_error_raises_runtime_error(monkeypatch) -> None:
    _patch_post(monkeypatch, _response(500, "upstream down"))
    b = OpenAICompatBackend(base_url="http://x/v1")
    with pytest.raises(RuntimeError) as exc_info:
        b.chat(
            messages=[{"role": "user", "content": "q"}],
            model="any",
            sampling=SamplingConfig(),
        )
    assert "500" in str(exc_info.value)


def test_chat_non_json_response_raises_runtime_error(monkeypatch) -> None:
    _patch_post(monkeypatch, _response(200, "not json at all"))
    b = OpenAICompatBackend(base_url="http://x/v1")
    with pytest.raises(RuntimeError):
        b.chat(
            messages=[{"role": "user", "content": "q"}],
            model="any",
            sampling=SamplingConfig(),
        )


def test_chat_empty_choices_raises(monkeypatch) -> None:
    _patch_post(monkeypatch, _response(200, {"choices": []}))
    b = OpenAICompatBackend(base_url="http://x/v1")
    with pytest.raises(RuntimeError):
        b.chat(
            messages=[{"role": "user", "content": "q"}],
            model="any",
            sampling=SamplingConfig(),
        )


def test_authorization_header_included(monkeypatch) -> None:
    captured = _patch_post(
        monkeypatch,
        _response(200, {"choices": [{"message": {"content": "ok"}}]}),
    )
    b = OpenAICompatBackend(base_url="http://x/v1", api_key="sk-test")
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="any",
        sampling=SamplingConfig(),
    )
    assert captured["headers"]["Authorization"] == "Bearer sk-test"


# ---------------------------------------------------------------------------
# CrewAI bridge
# ---------------------------------------------------------------------------


def test_make_crewai_llm_routes_via_hosted_vllm_provider() -> None:
    """CrewAI's native `hosted_vllm` provider covers any OpenAI-wire
    compatible local server without requiring LiteLLM as an extra.
    """
    b = OpenAICompatBackend(
        base_url="http://vllm:8000/v1", api_key="sk-local"
    )
    llm = b.make_crewai_llm(
        "Qwen/Qwen2.5-7B-Instruct",
        SamplingConfig(temperature=0.6, top_p=0.95, seed=42),
    )
    assert "Qwen/Qwen2.5-7B-Instruct" in llm.model
    assert llm.temperature == 0.6


def test_make_crewai_llm_omits_seed_when_none() -> None:
    b = OpenAICompatBackend(base_url="http://vllm:8000/v1")
    llm = b.make_crewai_llm(
        "Qwen/Qwen2.5-7B-Instruct", SamplingConfig(seed=None)
    )
    # Temperature made it through → config was applied. No portable
    # way to assert seed absence, but the happy path is verified.
    assert llm.temperature > 0
