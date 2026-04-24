"""Tests for the report-translation package (wise_investor.translation).

The actual Ollama call is never made in tests — we inject a stub
`llm_call` so behavior is deterministic and fast.
"""

from __future__ import annotations

from wise_investor.translation.translator import (
    SUPPORTED_TARGET_LANGUAGES,
    _build_system_prompt,
    _split_sections,
    translate_report,
)


# ---------------------------------------------------------------------------
# No-op / short-circuit paths
# ---------------------------------------------------------------------------


def test_translate_report_english_is_noop() -> None:
    """English target returns the input unchanged — no LLM call."""
    src = "# NVDA\n\nSome text.\n"

    def _should_not_be_called(system: str, user: str) -> str:
        raise AssertionError("LLM must not be called for English target")

    out = translate_report(src, "en", llm_call=_should_not_be_called)
    assert out == src


def test_translate_report_empty_returns_input() -> None:
    def _should_not_be_called(system: str, user: str) -> str:
        raise AssertionError("LLM must not be called on empty input")

    assert translate_report("", "ko", llm_call=_should_not_be_called) == ""
    assert (
        translate_report("   \n\n", "ko", llm_call=_should_not_be_called)
        == "   \n\n"
    )


def test_translate_report_unknown_lang_returns_input() -> None:
    """Typos like 'kr' should not crash — fall back to English."""
    src = "# Title\n\nbody\n"

    def _should_not_be_called(system: str, user: str) -> str:
        raise AssertionError("LLM must not be called for unsupported lang")

    out = translate_report(src, "kr", llm_call=_should_not_be_called)
    assert out == src


# ---------------------------------------------------------------------------
# Chunking behavior
# ---------------------------------------------------------------------------


def test_split_sections_splits_at_hr_boundaries() -> None:
    report = (
        "# Part 1 · Economist\n\nA\n\n"
        "---\n\n"
        "# Part 2 · Analyst\n\nB\n\n"
        "---\n\n"
        "# Part 3 · Valuer\n\nC"
    )
    chunks = _split_sections(report)
    assert len(chunks) == 3
    assert chunks[0].startswith("# Part 1 · Economist")
    assert chunks[1].startswith("# Part 2 · Analyst")
    assert chunks[2].startswith("# Part 3 · Valuer")


def test_split_sections_drops_empty_chunks() -> None:
    # A report wrapped in redundant separators should not produce
    # empty chunks in the joined output.
    report = "\n---\n\n---\n\n# Part 1\n\nbody\n\n---\n\n"
    chunks = _split_sections(report)
    assert len(chunks) == 1
    assert chunks[0].startswith("# Part 1")


# ---------------------------------------------------------------------------
# Translation dispatch uses the injected LLM
# ---------------------------------------------------------------------------


def test_translate_report_calls_llm_per_section() -> None:
    """One LLM call per section. Output is joined with HR separators."""
    report = (
        "# Part 1\n\nalpha\n\n"
        "---\n\n"
        "# Part 2\n\nbeta"
    )
    calls: list[tuple[str, str]] = []

    def _stub(system: str, user: str) -> str:
        calls.append((system, user))
        # Pretend to translate by uppercasing the body.
        return user.upper()

    out = translate_report(report, "ko", llm_call=_stub)
    assert len(calls) == 2
    # Both sections included.
    assert "ALPHA" in out or "# PART 1" in out
    assert "BETA" in out or "# PART 2" in out
    # Sections joined by HR boundary.
    assert "\n\n---\n\n" in out


def test_translate_report_falls_back_on_chunk_error() -> None:
    """If the LLM fails on one chunk, keep the English for that chunk
    rather than dropping it or failing the whole report.
    """
    report = (
        "# Part 1\n\nalpha\n\n"
        "---\n\n"
        "# Part 2\n\nbeta"
    )
    call_count = {"n": 0}

    def _stub_flaky(system: str, user: str) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("mock LLM outage")
        return "translated-" + user.splitlines()[-1]

    out = translate_report(report, "ko", llm_call=_stub_flaky)
    # First chunk kept as-is (English); second chunk translated.
    assert "alpha" in out
    assert "translated-beta" in out


def test_translate_report_keeps_english_when_llm_returns_empty() -> None:
    """Empty LLM response should fall back to the source chunk, not
    silently drop the section.
    """
    report = "# Part 1\n\nalpha"

    def _stub_empty(system: str, user: str) -> str:
        return ""

    out = translate_report(report, "ko", llm_call=_stub_empty)
    assert "alpha" in out


# ---------------------------------------------------------------------------
# System prompt content
# ---------------------------------------------------------------------------


def test_system_prompt_names_target_language() -> None:
    assert "Korean" in _build_system_prompt("ko")
    assert "Japanese" in _build_system_prompt("ja")
    assert "Chinese" in _build_system_prompt("zh")


def test_system_prompt_mentions_source_citation_preservation() -> None:
    for lang in ("ko", "ja", "zh"):
        prompt = _build_system_prompt(lang)
        assert "[Source:" in prompt or "Source" in prompt
        assert "markdown" in prompt.lower()


def test_supported_target_languages_excludes_english() -> None:
    """English is a no-op, not a translation target."""
    assert "en" not in SUPPORTED_TARGET_LANGUAGES
    assert set(SUPPORTED_TARGET_LANGUAGES) == {"ko", "ja", "zh"}


