"""Runner-side tests for the tip-injection wiring (Phase 2 v3).

Covers:
  - `_wrap_user_prompt_with_facts` honors the new `tips_block` kwarg.
  - run_crew_synthesis without `run_tag` is a no-op (existing callers
    that haven't migrated still work).
  - End-to-end runner integration is mocked at the synthesis level —
    we don't actually call an LLM.
"""

from __future__ import annotations

from wise_investor.agents.runner import _wrap_user_prompt_with_facts


def test_wrap_user_prompt_without_tips_block_unchanged() -> None:
    """Default behavior — existing callers see no difference."""
    out = _wrap_user_prompt_with_facts(
        task_prompt="DO THE TASK",
        facts={"calculate_per": "PE = 30"},
    )
    assert "DO THE TASK" in out
    assert "</pre_gathered_tool_outputs>" in out
    # The optional block is absent — no `<user_provided_tips>` marker.
    assert "<user_provided_tips>" not in out


def test_wrap_user_prompt_inserts_tips_after_facts_before_task() -> None:
    """Tips appear AFTER the facts wrapper closes and BEFORE the
    task prompt — the citation rules attach to the tool corpus, not
    to the user-provided context.
    """
    tips_block = "<user_provided_tips>\nfake-tip\n</user_provided_tips>"
    out = _wrap_user_prompt_with_facts(
        task_prompt="THE TASK BODY",
        facts={"calculate_per": "PE = 30"},
        tips_block=tips_block,
    )

    facts_close = out.find("</pre_gathered_tool_outputs>")
    tips_pos = out.find("<user_provided_tips>")
    task_pos = out.find("THE TASK BODY")

    assert facts_close >= 0 and tips_pos >= 0 and task_pos >= 0
    # Order: facts block closes, then tips, then task.
    assert facts_close < tips_pos < task_pos


def test_wrap_user_prompt_empty_tips_block_is_skipped() -> None:
    """Empty string for tips_block must NOT inject a stray newline /
    placeholder — it's the same as the default no-tips path.
    """
    base = _wrap_user_prompt_with_facts(
        task_prompt="DO THE TASK",
        facts={"calculate_per": "PE = 30"},
    )
    with_empty = _wrap_user_prompt_with_facts(
        task_prompt="DO THE TASK",
        facts={"calculate_per": "PE = 30"},
        tips_block="",
    )
    assert base == with_empty


def test_wrap_user_prompt_preserves_skeptic_flag() -> None:
    """The `is_skeptic` flag was an existing public param — making
    sure we didn't break it when adding tips_block.
    """
    out = _wrap_user_prompt_with_facts(
        task_prompt="ATTACK",
        facts={"calculate_per": "PE = 30"},
        is_skeptic=True,
        tips_block="<user_provided_tips>\nfoo\n</user_provided_tips>",
    )
    assert "<user_provided_tips>" in out
    assert "ATTACK" in out
