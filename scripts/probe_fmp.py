"""Probe which FMP /stable/ endpoints are accessible on the current plan.

Run: python scripts/probe_fmp.py

Endpoints listed here cover what Phase 1A calculation tools need. Those that fail
with 403/402 are paid-only and will require alternatives.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


console = Console()
BASE = "https://financialmodelingprep.com/stable"


# Each entry: (label, path, query params beyond apikey, expected shape note)
PROBES: list[tuple[str, str, dict[str, Any], str]] = [
    ("search-symbol", "/search-symbol", {"query": "AAPL"}, "list[{symbol,name,...}]"),
    ("quote", "/quote", {"symbol": "AAPL"}, "list[{price, marketCap, pe, ...}]"),
    ("profile", "/profile", {"symbol": "AAPL"}, "list[{sector, industry, beta, ...}]"),
    ("income-statement", "/income-statement", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of annual income stmts"),
    ("balance-sheet-statement", "/balance-sheet-statement", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of annual balance sheets"),
    ("cash-flow-statement", "/cash-flow-statement", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of annual cash flows"),
    ("ratios", "/ratios", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of financial ratios"),
    ("key-metrics", "/key-metrics", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of key metrics"),
    ("enterprise-values", "/enterprise-values", {"symbol": "AAPL", "period": "annual", "limit": "3"}, "list of enterprise values"),
    ("historical-price-eod-full", "/historical-price-eod/full", {"symbol": "AAPL"}, "historical EOD prices"),
    ("stock-peers", "/stock-peers", {"symbol": "AAPL"}, "list of peer symbols"),
    ("stock-list", "/stock-list", {}, "list of all stocks"),
]


def probe_one(label: str, path: str, params: dict[str, Any]) -> tuple[str, int | str, str]:
    url = f"{BASE}{path}"
    full_params = {**params, "apikey": settings.fmp_api_key}
    try:
        r = httpx.get(url, params=full_params, timeout=10.0)
    except Exception as e:
        return label, "ERR", f"request failed: {e}"

    status = r.status_code

    # Try to parse body preview (without echoing the URL which contains the key)
    try:
        body = r.json()
    except Exception:
        body = r.text[:200]

    if isinstance(body, dict) and "Error Message" in body:
        return label, status, f"error: {body['Error Message'][:160]}"

    if isinstance(body, list):
        preview = f"list len={len(body)}"
        if body and isinstance(body[0], dict):
            preview += f", keys={list(body[0].keys())[:6]}"
        return label, status, preview

    if isinstance(body, dict):
        return label, status, f"dict keys={list(body.keys())[:6]}"

    return label, status, f"text[:120]={str(body)[:120]}"


def main() -> int:
    if not settings.fmp_api_key or settings.fmp_api_key == "your_fmp_api_key_here":
        console.print("[red]FMP_API_KEY not set. Fill .env first.[/red]")
        return 1

    table = Table(title="FMP /stable/ endpoint probe — free tier")
    table.add_column("Endpoint")
    table.add_column("Status", justify="right")
    table.add_column("Response preview")

    ok = 0
    fail = 0
    details: list[dict[str, Any]] = []
    for label, path, params, _note in PROBES:
        lbl, status, preview = probe_one(label, path, params)
        if isinstance(status, int) and 200 <= status < 300 and "error:" not in preview:
            status_cell = f"[green]{status}[/green]"
            ok += 1
        else:
            status_cell = f"[red]{status}[/red]"
            fail += 1
        table.add_row(lbl, status_cell, preview)
        details.append({"endpoint": label, "path": path, "status": status, "preview": preview})

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] OK={ok}  FAIL={fail}\n")

    # Save raw report so we can reference it when building the client
    report_path = REPO_ROOT / "data" / "fmp_probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(details, indent=2, ensure_ascii=False))
    console.print(f"Report saved: [dim]{report_path}[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
