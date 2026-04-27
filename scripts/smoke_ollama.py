"""Smoke test for the active LLM backend.

Phase 1B's original goal — verifying byte-identical output across two
calls — is obsolete under the Phase 5 policy (MAFIS now follows
each model's recommended sampling, which is non-zero temperature).
This script is repurposed as a connectivity / capability probe:

  1. The factory builds the configured backend (default Ollama).
  2. The backend reaches its server and reports the installed models.
  3. Each agent in `config/agent_models.yaml` resolves to a known
     model (so a typo in the YAML or .env surfaces here, not in the
     middle of a 20-minute crew run).
  4. One round-trip chat call against the resolved Analyst model to
     confirm generation works.
  5. Soft JSON-format capability check (proxy for tool-call readiness)
     — same as before, with the new sampling settings.

Usage:
    python scripts/smoke_ollama.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.llm import get_agent_config, get_backend  # noqa: E402


console = Console()


def main() -> int:
    console.rule("[bold]Backend smoke[/bold]")

    backend = get_backend()
    console.print(f"Backend: [yellow]{backend.name}[/yellow]")

    # -- 1. Availability + model listing
    if not backend.is_available():
        console.print(
            f"[red]Backend {backend.name} unreachable. Start the server "
            "(e.g. `ollama serve`) and retry.[/red]"
        )
        return 1

    installed = backend.list_models()
    console.print(f"\n[bold]Installed models on {backend.name}[/bold]")
    if not installed:
        console.print(
            "[yellow]list_models() returned empty — fine for some "
            "backends (mlx, llamacpp), unusual for ollama.[/yellow]"
        )
    else:
        for name in installed:
            console.print(f"  • {name}")

    # -- 2. Per-agent config resolution
    console.print("\n[bold]Per-agent configuration[/bold]")
    cfg_table = Table()
    cfg_table.add_column("Agent")
    cfg_table.add_column("Model")
    cfg_table.add_column("temp", justify="right")
    cfg_table.add_column("top_p", justify="right")
    cfg_table.add_column("seed", justify="right")
    cfg_table.add_column("source")
    for agent in (
        "economist", "analyst", "valuer", "skeptic", "defender", "steward",
    ):
        cfg = get_agent_config(agent, backend=backend.name)
        s = cfg.sampling
        cfg_table.add_row(
            agent,
            cfg.model,
            f"{s.temperature}",
            f"{s.top_p}",
            "—" if s.seed is None else str(s.seed),
            cfg.source,
        )
    console.print(cfg_table)

    # -- 3. Round-trip chat against the Analyst model
    console.print("\n[bold]Round-trip chat (Analyst model)[/bold]")
    cfg = get_agent_config("analyst", backend=backend.name)
    prompt = (
        "Give a one-sentence description of the P/E ratio. "
        "Keep it under 30 words."
    )
    t0 = time.perf_counter()
    try:
        response = backend.chat(
            messages=[{"role": "user", "content": prompt}],
            model=cfg.model,
            sampling=cfg.sampling,
        )
    except Exception as e:
        console.print(f"[red]chat() failed: {e}[/red]")
        return 1
    elapsed = time.perf_counter() - t0
    console.print(f"  latency: {elapsed:.1f}s, len={len(response.content)} chars")
    console.print(f"  excerpt: [dim]{response.content[:140]}[/dim]")

    # -- 4. JSON-format capability — proxy for tool-call readiness
    console.print("\n[bold]JSON-format capability (tool-call readiness)[/bold]")
    json_prompt = (
        "Respond with a single JSON object and nothing else, no code fences. "
        'Shape: {"symbol": "AAPL", "action": "analyze", '
        '"fields": ["per", "ev_ebitda"]}. Return exactly that object.'
    )
    try:
        json_resp = backend.chat(
            messages=[{"role": "user", "content": json_prompt}],
            model=cfg.model,
            sampling=cfg.sampling,
        )
    except Exception as e:
        console.print(f"[red]JSON probe failed: {e}[/red]")
        return 1
    raw = json_resp.content.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(line for line in lines if not line.startswith("```"))
    try:
        parsed = json.loads(raw)
        ok = (
            isinstance(parsed, dict)
            and parsed.get("symbol") == "AAPL"
            and "per" in parsed.get("fields", [])
        )
        flag = "[green]OK[/green]" if ok else "[yellow]WEAK[/yellow]"
        console.print(f"  {flag}: {raw[:120]}")
    except json.JSONDecodeError as e:
        console.print(f"[yellow]WEAK[/yellow]: parse failed ({e})")
        console.print(f"  excerpt: [dim]{raw[:120]}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
