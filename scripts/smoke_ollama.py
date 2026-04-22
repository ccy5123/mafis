"""Phase 1B Task #1: verify Ollama + local models are usable with deterministic
settings (temperature=0, seed=42) required by design-v2.2 re-review Critical #1.

Checks:
1. Ollama Python client connects to the configured host
2. Both analyst_model and skeptic_model are installed
3. Each model responds to a simple prompt
4. Back-to-back calls with the same seed return byte-identical text
5. Rough tool-call format capability (ask for JSON, verify it parses)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from ollama import Client
from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


console = Console()


def deterministic_options() -> dict:
    return {
        "temperature": settings.llm_temperature,
        "seed": settings.llm_seed,
    }


def check_deterministic(client: Client, model: str, prompt: str) -> tuple[bool, str, float]:
    """Run the same prompt twice, return (matched, first_reply, elapsed_sec)."""
    t0 = time.perf_counter()
    r1 = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=deterministic_options(),
    )
    r2 = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=deterministic_options(),
    )
    elapsed = time.perf_counter() - t0
    text1 = r1["message"]["content"]
    text2 = r2["message"]["content"]
    return text1 == text2, text1, elapsed


def check_json_capability(client: Client, model: str) -> tuple[bool, str]:
    """Ask the model for strict JSON output — does it produce parseable JSON?"""
    prompt = (
        "Respond with a single JSON object and nothing else, no code fences. "
        'Shape: {"symbol": "AAPL", "action": "analyze", "fields": ["per", "ev_ebitda"]}. '
        "Return exactly that object."
    )
    r = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=deterministic_options(),
    )
    text = r["message"]["content"].strip()
    # Strip code fences if the model added them despite the instruction.
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        parsed = json.loads(text)
        ok = (
            isinstance(parsed, dict)
            and parsed.get("symbol") == "AAPL"
            and "per" in parsed.get("fields", [])
        )
        return ok, text[:200]
    except json.JSONDecodeError as e:
        return False, f"JSON parse failed: {e}; text={text[:200]}"


def main() -> int:
    console.rule("[bold]Phase 1B Task #1 — Ollama smoke[/bold]")

    client = Client(host=settings.ollama_host)

    # -- 1. List installed models
    console.print("\n[bold]1. Installed models[/bold]")
    try:
        resp = client.list()
    except Exception as e:
        console.print(f"[red]Cannot reach Ollama at {settings.ollama_host}: {e}[/red]")
        return 1

    installed = [m.model for m in resp.models]
    for name in installed:
        console.print(f"  • {name}")

    required = sorted({settings.analyst_model, settings.skeptic_model})
    missing = [m for m in required if m not in installed]
    if missing:
        console.print(f"[red]Missing models: {missing}[/red]")
        console.print(f"  Run: {' '.join(f'ollama pull {m}' for m in missing)}")
        return 1

    # -- 2 & 3. Determinism check on each required model
    console.print("\n[bold]2. Determinism check (same seed → byte-identical output)[/bold]")
    det_table = Table()
    det_table.add_column("Model")
    det_table.add_column("Deterministic", justify="center")
    det_table.add_column("2x latency", justify="right")
    det_table.add_column("Reply excerpt")
    prompt = (
        "Give a one-sentence description of the P/E ratio. "
        "Keep it under 30 words."
    )
    for model in required:
        try:
            matched, reply, elapsed = check_deterministic(client, model, prompt)
        except Exception as e:
            det_table.add_row(model, "[red]ERR[/red]", "—", f"error: {e}")
            continue
        flag = "[green]YES[/green]" if matched else "[red]NO[/red]"
        det_table.add_row(model, flag, f"{elapsed:.1f}s", reply[:90])
    console.print(det_table)

    # -- 4. JSON-format capability (proxy for tool-calling readiness)
    console.print("\n[bold]3. JSON-formatted response (tool-call readiness)[/bold]")
    json_table = Table()
    json_table.add_column("Model")
    json_table.add_column("JSON ok", justify="center")
    json_table.add_column("Reply excerpt")
    for model in required:
        ok, text = check_json_capability(client, model)
        flag = "[green]OK[/green]" if ok else "[yellow]WEAK[/yellow]"
        json_table.add_row(model, flag, text[:120])
    console.print(json_table)

    console.print(
        "\n[dim]If JSON-ok is WEAK, Phase 1B may need stricter prompt templates "
        "or a retry-on-parse-fail wrapper around CrewAI tool calls.[/dim]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
