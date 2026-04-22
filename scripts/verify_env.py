"""Phase 0 environment verification.

Checks that every prerequisite described in README.md is actually available.
Run: python scripts/verify_env.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


console = Console()

REQUIRED_MODELS = [
    settings.analyst_model,
    settings.valuer_model,
    settings.skeptic_model,
]
REQUIRED_MODELS = sorted(set(REQUIRED_MODELS))


def check_python_version() -> tuple[bool, str]:
    major, minor = sys.version_info.major, sys.version_info.minor
    ok = (major, minor) >= (3, 12)
    return ok, f"{major}.{minor}.{sys.version_info.micro}"


def check_ollama_binary() -> tuple[bool, str]:
    path = shutil.which("ollama")
    if path is None:
        return False, "ollama CLI not found on PATH"
    return True, path


def check_ollama_reachable() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        r.raise_for_status()
        return True, f"{settings.ollama_host} responding"
    except Exception as e:
        return False, f"{settings.ollama_host} unreachable: {e}"


def check_ollama_models() -> tuple[bool, str]:
    try:
        r = httpx.get(f"{settings.ollama_host}/api/tags", timeout=3.0)
        r.raise_for_status()
        installed = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        return False, f"cannot query models: {e}"

    missing = [m for m in REQUIRED_MODELS if m not in installed]
    if missing:
        hint = " ".join(f"ollama pull {m}" for m in missing)
        return False, f"missing: {missing}  |  run:  {hint}"
    return True, f"all installed: {REQUIRED_MODELS}"


def check_fmp_key() -> tuple[bool, str]:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        return False, "FMP_API_KEY not set in .env"

    # FMP deprecated /api/v3/ on 2025-08-31. New base is /stable/.
    # See https://site.financialmodelingprep.com/developer/docs
    try:
        url = "https://financialmodelingprep.com/stable/search-symbol"
        r = httpx.get(
            url,
            params={"query": "AAPL", "apikey": settings.fmp_api_key},
            timeout=5.0,
        )
        if r.status_code == 401:
            return False, "FMP rejected key (401 — invalid or revoked)"
        if r.status_code == 403:
            return False, "FMP 403 — key invalid or endpoint restricted"
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "Error Message" in data:
            return False, f"FMP error: {data['Error Message']}"
        if not data:
            return False, "FMP returned empty response"
        return True, f"FMP /stable/ works (found {len(data)} symbol match(es))"
    except Exception as e:
        return False, f"FMP ping failed: {e}"


def check_storage_dirs() -> tuple[bool, str]:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return True, f"chroma={settings.chroma_persist_dir}, sqlite={settings.sqlite_path.parent}"


CHECKS = [
    ("Python >= 3.12", check_python_version),
    ("Ollama CLI on PATH", check_ollama_binary),
    ("Ollama service reachable", check_ollama_reachable),
    ("Required models pulled", check_ollama_models),
    ("FMP API key valid", check_fmp_key),
    ("Storage directories", check_storage_dirs),
]


def main() -> int:
    table = Table(title="Phase 0 Environment Check")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    fail = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"raised: {e!r}"
        status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(name, status, detail)
        if not ok:
            fail += 1

    console.print(table)
    if fail:
        console.print(f"\n[red]{fail} check(s) failed. See README.md setup steps.[/red]")
        return 1
    console.print("\n[green]All checks passed. Ready for Phase 1A.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
