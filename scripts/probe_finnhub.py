"""Probe which Finnhub endpoints we can actually use on the free tier.

Run after FINNHUB_API_KEY is set in .env:
    python scripts/probe_finnhub.py

For each endpoint we care about (quote, profile, financials, peers, historical
prices, metric), prints HTTP status and a preview of the response keys. Output
is saved to data/finnhub_probe_report.json for reference when building the
client.
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
BASE = "https://finnhub.io/api/v1"


# Endpoints documented at https://finnhub.io/docs/api
# Note: Finnhub charges requests per endpoint in "credits"; free tier allows
# 60 calls/min up to a rolling 60s window.
PROBES: list[tuple[str, str, dict[str, Any]]] = [
    ("quote", "/quote", {"symbol": "AAPL"}),
    ("profile2", "/stock/profile2", {"symbol": "AAPL"}),
    ("peers", "/stock/peers", {"symbol": "AAPL"}),
    # financials-reported = full standardized income/balance/cash-flow
    ("financials-reported-annual", "/stock/financials-reported",
     {"symbol": "AAPL", "freq": "annual"}),
    ("financials-reported-quarterly", "/stock/financials-reported",
     {"symbol": "AAPL", "freq": "quarterly"}),
    # metric = pre-computed metrics (some premium)
    ("metric-all", "/stock/metric", {"symbol": "AAPL", "metric": "all"}),
    # basic financials (lighter endpoint)
    ("basic-financials", "/stock/metric",
     {"symbol": "AAPL", "metric": "financial"}),
    # EPS, revenue over time
    ("earnings", "/stock/earnings", {"symbol": "AAPL"}),
    # candle = OHLCV history
    ("candle", "/stock/candle",
     {"symbol": "AAPL", "resolution": "D", "from": 1_700_000_000, "to": 1_776_000_000}),
    # insider trades, news, recommendation — optional
    ("recommendation-trends", "/stock/recommendation", {"symbol": "AAPL"}),
    # Symbol lookup
    ("symbol-lookup", "/search", {"q": "apple"}),
]


def probe_one(label: str, path: str, params: dict[str, Any]) -> tuple[int | str, str]:
    url = f"{BASE}{path}"
    full_params = {**params, "token": settings.finnhub_api_key}
    try:
        r = httpx.get(url, params=full_params, timeout=10.0)
    except Exception as e:
        return "ERR", f"request failed: {e}"

    status = r.status_code
    try:
        body = r.json()
    except Exception:
        return status, f"non-json body: {r.text[:120]}"

    if status >= 400:
        return status, f"error body: {str(body)[:200]}"

    if isinstance(body, list):
        preview = f"list len={len(body)}"
        if body and isinstance(body[0], dict):
            preview += f", keys={list(body[0].keys())[:6]}"
        elif body and isinstance(body[0], str):
            preview += f", sample={body[:3]}"
        return status, preview
    if isinstance(body, dict):
        keys = list(body.keys())
        preview = f"dict keys={keys[:8]}"
        # Some endpoints return { "data": [...] } or "metric": {...} — note that
        if "data" in body and isinstance(body["data"], list):
            preview += f" | data list len={len(body['data'])}"
        if "metric" in body and isinstance(body["metric"], dict):
            mkeys = list(body["metric"].keys())
            preview += f" | metric subkeys sample: {mkeys[:6]} (+{len(mkeys)} total)"
        return status, preview
    return status, f"unexpected shape: {type(body).__name__}"


def main() -> int:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        console.print("[red]FINNHUB_API_KEY not set. Fill .env first.[/red]")
        return 1

    table = Table(title="Finnhub endpoint probe — free tier")
    table.add_column("Endpoint")
    table.add_column("Status", justify="right")
    table.add_column("Response preview")

    ok = 0
    fail = 0
    details: list[dict[str, Any]] = []
    for label, path, params in PROBES:
        status, preview = probe_one(label, path, params)
        if isinstance(status, int) and 200 <= status < 300 and "error body" not in preview:
            status_cell = f"[green]{status}[/green]"
            ok += 1
        else:
            status_cell = f"[red]{status}[/red]"
            fail += 1
        table.add_row(label, status_cell, preview[:120])
        details.append({"endpoint": label, "path": path, "status": status, "preview": preview})

    console.print(table)
    console.print(f"\n[bold]Summary:[/bold] OK={ok}  FAIL={fail}\n")

    report_path = REPO_ROOT / "data" / "finnhub_probe_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(details, indent=2, ensure_ascii=False))
    console.print(f"Report saved: [dim]{report_path}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
