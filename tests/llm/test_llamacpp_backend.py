"""llama.cpp backend tests.

Skip most tests unless llama-cpp-python is installed (optional dep).
When installed we can cover construction, env handling, and the
missing-file validation without ever loading a real GGUF.
"""

from __future__ import annotations

import pytest


from wise_investor.llm.backends import llamacpp as llamacpp_module  # noqa: E402


# ---------------------------------------------------------------------------
# Module import (graceful on platforms without the dep)
# ---------------------------------------------------------------------------


def test_module_imports_without_llama_cpp_installed() -> None:
    """Module must import so the factory can see the class."""
    assert hasattr(llamacpp_module, "LlamaCppBackend")


def test_constructor_raises_runtime_error_on_missing_lib(tmp_path) -> None:
    """If llama-cpp-python isn't installed, we want a clear RuntimeError
    with install hint, not a naked ImportError from deep in __init__.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        fake_gguf = tmp_path / "model.gguf"
        fake_gguf.write_bytes(b"not really a gguf")
        with pytest.raises(RuntimeError) as exc_info:
            llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
        assert "llama-cpp-python" in str(exc_info.value)
    else:
        pytest.skip("llama-cpp-python is installed; missing-dep path is untestable.")


# ---------------------------------------------------------------------------
# Config validation (dep installed)
# ---------------------------------------------------------------------------


llama_cpp = pytest.importorskip(
    "llama_cpp", reason="llama-cpp-python not installed"
)


def test_constructor_requires_model_path(monkeypatch) -> None:
    monkeypatch.delenv("LLAMACPP_MODEL_PATH", raising=False)
    with pytest.raises(ValueError) as exc_info:
        llamacpp_module.LlamaCppBackend()
    assert "model path" in str(exc_info.value).lower()


def test_constructor_rejects_missing_gguf(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.gguf"
    with pytest.raises(FileNotFoundError) as exc_info:
        llamacpp_module.LlamaCppBackend(model_path=str(missing))
    assert "not found" in str(exc_info.value).lower()


def test_constructor_reads_env_path(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    monkeypatch.setenv("LLAMACPP_MODEL_PATH", str(fake_gguf))
    b = llamacpp_module.LlamaCppBackend()
    assert b.model_path == fake_gguf
    assert b.name == "llamacpp"


def test_n_gpu_layers_defaults_to_all(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    monkeypatch.delenv("LLAMACPP_N_GPU_LAYERS", raising=False)
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert b.n_gpu_layers == -1


def test_n_gpu_layers_env_override(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    monkeypatch.setenv("LLAMACPP_N_GPU_LAYERS", "0")
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert b.n_gpu_layers == 0


def test_n_ctx_defaults_to_8192(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    monkeypatch.delenv("LLAMACPP_N_CTX", raising=False)
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert b.n_ctx == 8192


def test_is_available_true_when_file_exists(tmp_path) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert b.is_available() is True


def test_list_models_returns_single_gguf_filename(tmp_path) -> None:
    fake_gguf = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    fake_gguf.write_bytes(b"header")
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert b.list_models() == ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"]


def test_chat_happy_path_with_mocked_llama(tmp_path, monkeypatch) -> None:
    """Don't load a real GGUF — swap the Llama class for a stub that
    returns an OpenAI-shaped completion.
    """
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")

    class _FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": "pong"}}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }

    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)

    from wise_investor.llm.base import SamplingConfig

    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    resp = b.chat(
        messages=[{"role": "user", "content": "ping"}],
        model="model.gguf",
        sampling=SamplingConfig(temperature=0.6, top_p=0.9, max_tokens=16),
    )
    assert resp.content == "pong"
    assert resp.backend == "llamacpp"
    assert resp.extra["usage"]["prompt_tokens"] == 3


def test_chat_extracts_thinking(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")

    class _FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "<think>reason</think>\nfinal"
                        }
                    }
                ]
            }

    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)

    from wise_investor.llm.base import SamplingConfig

    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    resp = b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="model.gguf",
        sampling=SamplingConfig(enable_thinking=True),
    )
    assert resp.content == "final"
    assert resp.thinking == "reason"


def test_chat_errors_when_choices_missing(tmp_path, monkeypatch) -> None:
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")

    class _FakeLlama:
        def __init__(self, **kwargs):
            pass

        def create_chat_completion(self, **kwargs):
            return {"choices": []}

    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)

    from wise_investor.llm.base import SamplingConfig

    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    with pytest.raises(RuntimeError):
        b.chat(
            messages=[{"role": "user", "content": "q"}],
            model="model.gguf",
            sampling=SamplingConfig(),
        )


def test_chat_lazy_loads_on_first_call(tmp_path, monkeypatch) -> None:
    """Constructor should NOT load the model — probing availability
    or listing models shouldn't pay the 2-5s cost.
    """
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")

    load_counter = {"n": 0}

    class _FakeLlama:
        def __init__(self, **kwargs):
            load_counter["n"] += 1

        def create_chat_completion(self, **kwargs):
            return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)

    from wise_investor.llm.base import SamplingConfig

    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    assert load_counter["n"] == 0  # construction doesn't load
    b.is_available()
    assert load_counter["n"] == 0  # probing doesn't load
    b.list_models()
    assert load_counter["n"] == 0  # listing doesn't load
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="model.gguf",
        sampling=SamplingConfig(),
    )
    assert load_counter["n"] == 1  # first chat triggers load
    b.chat(
        messages=[{"role": "user", "content": "q"}],
        model="model.gguf",
        sampling=SamplingConfig(),
    )
    assert load_counter["n"] == 1  # second chat reuses cached model


def test_make_crewai_llm_raises_not_implemented(tmp_path) -> None:
    """No LiteLLM provider for in-process llama.cpp — users wanting
    a CrewAI path should run the llama-cpp-python server and point
    openai_compat at it.
    """
    fake_gguf = tmp_path / "model.gguf"
    fake_gguf.write_bytes(b"header")
    b = llamacpp_module.LlamaCppBackend(model_path=str(fake_gguf))
    from wise_investor.llm.base import SamplingConfig

    with pytest.raises(NotImplementedError):
        b.make_crewai_llm("model.gguf", SamplingConfig())
