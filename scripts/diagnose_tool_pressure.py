"""Pressure-test tool calling with progressively heavier prompts.

Goal: isolate whether the problem is prompt volume, tool count, or tool_choice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import ollama
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.agents.runner import TOOL_SPECS  # noqa: E402
from wise_investor.config import settings  # noqa: E402

console = Console()
MODEL = settings.analyst_model


def probe(label: str, messages: list[dict], tool_choice=None) -> None:
    console.rule(f"[bold]{label}[/bold]")
    kwargs = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOL_SPECS,
        "options": {"temperature": 0.0, "seed": 42},
    }
    if tool_choice:
        # Pass via raw HTTP since ollama python lib may not forward tool_choice.
        import httpx
        r = httpx.post(
            f"{settings.ollama_host}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOL_SPECS,
                "tool_choice": tool_choice,
                "temperature": 0.0,
                "seed": 42,
            },
            timeout=180.0,
        )
        data = r.json()
        msg = data["choices"][0]["message"]
        tc = msg.get("tool_calls") or []
    else:
        resp = ollama.chat(**kwargs)
        msg = resp["message"]
        tc = msg.get("tool_calls") or []

    if tc:
        console.print(f"[green]tool_calls={len(tc)}[/green]")
        for c in tc[:3]:
            fn = c["function"]
            console.print(f"  • {fn['name']}({fn.get('arguments')})")
    else:
        console.print("[red]tool_calls=0[/red]  → fallback to text reply:")
        text = msg.get("content", "")
        console.print(f"  {text[:250]}")


# Test 1: minimal ask, 6 tools available
probe(
    "A. Minimal ask, 6 tools",
    [
        {"role": "system", "content": "You are a research analyst. Use your tools to answer."},
        {"role": "user", "content": "What is NVDA's latest annual revenue?"},
    ],
)

# Test 2: add a long backstory
with open(REPO_ROOT / "src/wise_investor/agents/analyst.py") as f:
    analyst_src = f.read()
probe(
    "B. Minimal ask + large backstory-like system prompt",
    [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst. Every number you cite "
                "must come from a tool call. Never invent numbers. You have six "
                "tools available; call them before answering any numeric question."
            ),
        },
        {"role": "user", "content": "What is NVDA's latest annual revenue and PER?"},
    ],
)

# Test 3: with value chain doc embedded
vc_text = (REPO_ROOT / "docs/value_chains/NVDA.md").read_text()
probe(
    "C. Large prompt with value chain injected",
    [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst. Every number you cite "
                "must come from a tool call. Never invent numbers. Tools are "
                "available; you MUST call them for any numeric data."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given the following value chain document, write a 1-paragraph "
                "description of NVDA's current revenue. Use the verify_number "
                "tool with field='revenue' first.\n\n" + vc_text
            ),
        },
    ],
)

# Test 4: same as C but with tool_choice="required"
probe(
    "D. Same as C + tool_choice='required'",
    [
        {
            "role": "system",
            "content": (
                "You are a senior equity research analyst. Every number you cite "
                "must come from a tool call. Never invent numbers."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write a description of NVDA's revenue given this context:\n\n"
                + vc_text
            ),
        },
    ],
    tool_choice="required",
)
