"""Translate long English crew reports into user-preferred languages.

The Telegram summary renderer has its own deterministic LOCALE pack
(notify/summary.py — fixed vocabulary labels), but the attached .md
file contains LLM-generated prose that needs actual translation.
We use Ollama's Qwen 2.5 7B (the Analyst model, already resident on
the GPU after a crew run) at temp=0, seed=42 for determinism.

Design rules:
  - English target is a no-op — return the input unchanged.
  - Reports are chunked at '---' horizontal-rule boundaries so each
    Part (Economist / Analyst / Valuer / Skeptic / Defender / Steward)
    fits comfortably in Qwen's 16K context window.
  - The system prompt tells the LLM to preserve [Source: ...]
    citations, numeric values, markdown structure, and proper nouns
    verbatim. This keeps the Universal Citation Rule intact across
    the translation layer — the English report and the translated
    report point at the same fact sources.
  - Per-chunk LLM failures fall back to the English chunk with a
    logged warning. Partial translation beats a blank file.
"""

from __future__ import annotations

import logging
from typing import Callable

from wise_investor.config import settings


logger = logging.getLogger(__name__)


# Human-readable language names handed to the LLM so it doesn't have
# to guess what ISO code `ko` means. Map intentionally small — each
# addition requires testing the prompt with the new language.
_LANG_NAMES: dict[str, str] = {
    "ko": "Korean (한국어)",
    "ja": "Japanese (日本語)",
    "zh": "Simplified Chinese (简体中文)",
}


SUPPORTED_TARGET_LANGUAGES: tuple[str, ...] = tuple(_LANG_NAMES.keys())


def _build_system_prompt(target_lang: str) -> str:
    """System prompt for translating a FULL report section (hundreds
    to thousands of characters of markdown)."""
    lang_name = _LANG_NAMES.get(target_lang, "Korean (한국어)")
    return (
        f"You are a financial-research translator. Translate the user's "
        f"markdown text into {lang_name}. Rules:\n"
        "1. Preserve ALL markdown structure EXACTLY - heading levels, "
        "bullet points, tables, code blocks, bold/italic markers, and "
        "horizontal rules.\n"
        "2. Preserve ALL numbers, dates, tickers, model names, and "
        "percentage values unchanged. Do not convert units, round, or "
        "reformat.\n"
        "3. Preserve every [Source: ...] citation VERBATIM - do not "
        "translate the citation key or the surrounding brackets.\n"
        "4. Keep proper nouns (company names, product names, acronyms "
        "like GDP, AI, FCF, EBIT, FDA) in their standard localized "
        "form when one exists; otherwise leave them in the original "
        "script.\n"
        "5. Output ONLY the translated markdown - no preamble, no "
        "commentary, no closing remark. Do not wrap the output in "
        "code fences."
    )


def _split_sections(report: str) -> list[str]:
    """Split the report into translation-sized chunks.

    Each Part in a crew report ends with a '---' horizontal rule,
    so splitting on that boundary yields one chunk per Part plus
    any preamble / trailing block. Empty chunks are filtered out
    so the joiner doesn't produce '\\n\\n---\\n\\n---\\n\\n' runs.
    """
    parts = report.split("\n---\n")
    return [p.strip() for p in parts if p.strip()]


def translate_report(
    text: str,
    target_lang: str,
    llm_call: Callable[[str, str], str] | None = None,
) -> str:
    """Translate a crew-report markdown into `target_lang`.

    `target_lang` in ("ko", "ja", "zh"). "en" or any unrecognized code
    short-circuits and returns `text` unchanged — callers then skip
    writing the translated file.

    `llm_call(system, user) -> str` is injectable so tests can run
    without Ollama. Default is an Ollama chat call using the Analyst
    model at temp=0, seed=42 (same reproducibility contract as the
    rest of the crew).
    """
    if not text or not text.strip():
        return text
    if target_lang in ("en", "", None):
        return text
    if target_lang not in _LANG_NAMES:
        logger.warning(
            "translate_report: unsupported target_lang=%r; returning "
            "English unchanged.",
            target_lang,
        )
        return text

    if llm_call is None:
        llm_call = _default_llm_call

    system = _build_system_prompt(target_lang)
    sections = _split_sections(text)
    if not sections:
        return text

    translated: list[str] = []
    for i, section in enumerate(sections, start=1):
        try:
            out = llm_call(system, section)
        except Exception as e:
            logger.warning(
                "translate_report: chunk %d/%d LLM call failed (%s); "
                "keeping English for this section.",
                i,
                len(sections),
                e,
            )
            translated.append(section)
            continue
        translated.append((out or section).strip() or section)

    return "\n\n---\n\n".join(translated)


def _default_llm_call(system: str, user: str) -> str:
    """Production Ollama call using the Analyst model at temp 0, seed 42."""
    import ollama

    resp = ollama.chat(
        model=settings.analyst_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
    )
    return resp["message"]["content"]


__all__ = [
    "translate_report",
    "SUPPORTED_TARGET_LANGUAGES",
]
