"""Parse / strip thinking-mode blocks from model output.

Qwen3, DeepSeek-R1, and related reasoning models emit a visible
chain-of-thought wrapped in `<think>...</think>` tags before the
final answer. The crew report format was designed pre-thinking-mode
and does not have a place for reasoning traces, so the backend layer
strips these tags before handing content to agents.

The `thinking` substring is still surfaced on `LLMResponse.thinking`
for debug / transcript purposes — just never inlined into the .md
report.
"""

from __future__ import annotations

import re


# Matches the FIRST <think>...</think> block. We split rather than
# remove-all because a well-behaved model only emits one; if a model
# emits multiple, joining them with newlines keeps the transcript
# readable on the `.thinking` side.
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> tuple[str, str | None]:
    """Return (content_without_think_blocks, thinking_or_None).

    - Removes every `<think>...</think>` block from `text`.
    - Collapses the leading whitespace left behind so the cleaned
      text reads like a normal reply.
    - If multiple blocks appear, they are joined with blank lines on
      the thinking side.
    - If no block is present, returns (text, None) so callers can
      no-op with `if thinking:` checks.
    """
    if not text:
        return text, None

    blocks = _THINK_BLOCK_RE.findall(text)
    if not blocks:
        return text, None

    cleaned = _THINK_BLOCK_RE.sub("", text)
    # Trim whitespace artifacts left by removed blocks.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    thinking = "\n\n".join(b.strip() for b in blocks if b.strip())
    return cleaned, (thinking or None)


__all__ = ["strip_thinking"]
