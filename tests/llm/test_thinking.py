"""Extract / strip <think>...</think> blocks from model output."""

from __future__ import annotations

from wise_investor.llm.utils.thinking import strip_thinking


def test_no_think_block_returns_input_unchanged() -> None:
    cleaned, thinking = strip_thinking("Just the answer.")
    assert cleaned == "Just the answer."
    assert thinking is None


def test_empty_text_returns_empty_and_none() -> None:
    cleaned, thinking = strip_thinking("")
    assert cleaned == ""
    assert thinking is None


def test_single_think_block_is_extracted() -> None:
    text = (
        "<think>Let me work through this step by step…\n"
        "The number is 42.</think>\n\n"
        "The answer is 42."
    )
    cleaned, thinking = strip_thinking(text)
    assert cleaned == "The answer is 42."
    assert thinking is not None
    assert "step by step" in thinking
    assert "42" in thinking


def test_multiple_think_blocks_are_joined() -> None:
    """Poorly-behaved models sometimes emit two chains of thought.
    Preserve both on the `thinking` side; drop both from `cleaned`.
    """
    text = (
        "<think>first reasoning</think>\n"
        "intermediate\n"
        "<think>second reasoning</think>\n"
        "final answer"
    )
    cleaned, thinking = strip_thinking(text)
    assert "<think>" not in cleaned
    assert "</think>" not in cleaned
    assert "intermediate" in cleaned
    assert "final answer" in cleaned
    assert thinking is not None
    assert "first reasoning" in thinking
    assert "second reasoning" in thinking


def test_think_block_is_case_insensitive() -> None:
    text = "<THINK>upper</THINK>\nanswer"
    cleaned, thinking = strip_thinking(text)
    assert cleaned == "answer"
    assert thinking == "upper"


def test_think_block_spans_newlines() -> None:
    """re.DOTALL must be enabled — model reasoning is almost always
    multiline.
    """
    text = "<think>line1\nline2\nline3</think>\nfinal"
    cleaned, thinking = strip_thinking(text)
    assert cleaned == "final"
    assert thinking == "line1\nline2\nline3"


def test_leading_whitespace_is_collapsed_after_strip() -> None:
    text = "<think>reasoning</think>\n\n\n\nanswer"
    cleaned, _ = strip_thinking(text)
    # Long runs of blank lines collapse to at most one paragraph break.
    assert "\n\n\n" not in cleaned


def test_whitespace_only_think_block_does_not_pollute_thinking_field() -> None:
    """If the model emits <think>  </think> (common on some fine-tunes),
    we should not return an empty-string `thinking` — that clutters
    logs and breaks `if resp.thinking:` checks.
    """
    cleaned, thinking = strip_thinking("<think>   \n\n   </think>\nanswer")
    assert cleaned == "answer"
    assert thinking is None


def test_only_think_block_yields_empty_cleaned_content() -> None:
    """Degenerate case — the model emitted ONLY reasoning and no
    final answer. `cleaned` should be empty string, not None.
    """
    cleaned, thinking = strip_thinking("<think>all reasoning</think>")
    assert cleaned == ""
    assert thinking == "all reasoning"
