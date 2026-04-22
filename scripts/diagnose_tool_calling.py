"""Diagnose tool-calling support path by path.

Three layers, probed from bottom to top:
  1. Direct ollama.chat(tools=...) — native Ollama Python client
  2. httpx POST /v1/chat/completions with tools — OpenAI-compatible endpoint
  3. (skipped here) CrewAI — already known to fail

If layer 1 works and layer 2 fails, the bug is in Ollama's OpenAI compatibility
layer; workaround is to bypass CrewAI's LiteLLM path and drive agents with the
native client.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import ollama
from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


console = Console()


TOOL_SPEC_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_nvda_revenue",
        "description": "Return NVIDIA's latest annual revenue in USD.",
        "parameters": {
            "type": "object",
            "properties": {
                "fiscal_year": {
                    "type": "string",
                    "description": "Fiscal year, e.g. 'FY2025' or 'latest'",
                }
            },
            "required": ["fiscal_year"],
        },
    },
}


def probe_ollama_native(model: str) -> None:
    console.rule(f"[bold]Layer 1: ollama.chat native — {model}[/bold]")
    try:
        resp = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "What is NVIDIA's latest annual revenue? Use the tool.",
                }
            ],
            tools=[TOOL_SPEC_OPENAI],
            options={"temperature": 0.0, "seed": 42},
        )
    except Exception as e:
        console.print(f"[red]FAILED:[/red] {e}")
        return

    msg = resp["message"]
    has_calls = bool(msg.get("tool_calls"))
    console.print(f"has tool_calls: [{'green' if has_calls else 'red'}]{has_calls}[/]")
    if has_calls:
        for tc in msg["tool_calls"]:
            console.print(f"  • {tc['function']['name']}({tc['function']['arguments']})")
    else:
        text = msg.get("content", "")[:300]
        console.print(f"  fallback text reply: {text}")


def probe_openai_endpoint(model: str) -> None:
    console.rule(f"[bold]Layer 2: POST /v1/chat/completions — {model}[/bold]")
    url = f"{settings.ollama_host}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is NVIDIA's latest annual revenue? Use the tool.",
            }
        ],
        "tools": [TOOL_SPEC_OPENAI],
        "tool_choice": "auto",
        "temperature": 0.0,
        "seed": 42,
    }
    try:
        r = httpx.post(url, json=body, timeout=120.0)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        console.print(f"[red]FAILED:[/red] {e}")
        return

    msg = data["choices"][0]["message"]
    tc = msg.get("tool_calls")
    has_calls = bool(tc)
    console.print(f"has tool_calls: [{'green' if has_calls else 'red'}]{has_calls}[/]")
    if has_calls:
        for call in tc:
            console.print(f"  • {call['function']['name']}({call['function']['arguments']})")
    else:
        text = msg.get("content", "")[:300]
        console.print(f"  fallback text reply: {text}")


if __name__ == "__main__":
    for model in [settings.analyst_model, "llama3.1:8b-16k", "qwen2.5:7b"]:
        probe_ollama_native(model)
        probe_openai_endpoint(model)
