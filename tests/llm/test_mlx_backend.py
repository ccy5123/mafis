"""MLX backend tests.

MLX is Apple Silicon only. On Windows / Linux / Intel Mac we can still
exercise the "graceful unavailability" path but never run the actual
chat code. `mlx_lm` import is the skip trigger.
"""

from __future__ import annotations

import pytest


# We always need the source module to be importable (it shouldn't
# import mlx_lm at module scope).
from wise_investor.llm.backends import mlx as mlx_module  # noqa: E402


# ---------------------------------------------------------------------------
# Module-level import (graceful on non-Apple)
# ---------------------------------------------------------------------------


def test_module_imports_without_mlx_installed() -> None:
    """Import must succeed on Windows where mlx_lm is unavailable —
    the factory needs to be able to inspect the class even on
    platforms that can't run it.
    """
    assert hasattr(mlx_module, "MLXBackend")


def test_constructor_raises_runtime_error_on_missing_mlx() -> None:
    """If mlx_lm isn't installed, constructing MLXBackend should raise
    a helpful RuntimeError (not ImportError directly). The RuntimeError
    mentions Apple Silicon so users on Windows immediately understand.
    """
    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError) as exc_info:
            mlx_module.MLXBackend()
        assert (
            "Apple Silicon" in str(exc_info.value)
            or "mlx-lm" in str(exc_info.value)
        )
    else:
        pytest.skip("mlx_lm is installed; cannot test the missing-dep path.")


# ---------------------------------------------------------------------------
# Instance behavior (Apple Silicon only)
# ---------------------------------------------------------------------------


mlx_lm = pytest.importorskip(
    "mlx_lm", reason="mlx-lm only installs on Apple Silicon"
)


def test_backend_name_is_mlx() -> None:
    b = mlx_module.MLXBackend()
    assert b.name == "mlx"


def test_is_available_true_after_successful_construction() -> None:
    """Reaching __init__ without raising means mlx_lm is importable
    and the hardware is right. No further probe should be needed.
    """
    b = mlx_module.MLXBackend()
    assert b.is_available() is True


def test_list_models_starts_empty() -> None:
    """list_models reports only what's been loaded in this session."""
    b = mlx_module.MLXBackend()
    assert b.list_models() == []
