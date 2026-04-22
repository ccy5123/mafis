"""Dump all keys from Finnhub /stock/metric?metric=all for AAPL.

Needed because Finnhub's metric endpoint returns a huge flat dict of
pre-computed ratios; the exact key names determine how we populate our
calculation tools. Run once to discover the schema, reference output when
building the FinnhubClient pydantic models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


def main() -> int:
    if not settings.finnhub_api_key:
        print("FINNHUB_API_KEY not set")
        return 1

    url = "https://finnhub.io/api/v1/stock/metric"
    r = httpx.get(
        url,
        params={"symbol": "AAPL", "metric": "all", "token": settings.finnhub_api_key},
        timeout=10.0,
    )
    r.raise_for_status()
    body = r.json()

    metric = body.get("metric", {})
    print(f"Total metric keys: {len(metric)}\n")

    # Print keys sorted, showing value type and sample
    for k in sorted(metric.keys()):
        v = metric[k]
        t = type(v).__name__
        preview = str(v)[:40]
        print(f"  {k:50s} [{t:5s}] = {preview}")

    # Also look for revenue/ebitda/debt/fcf specific fields
    print("\n--- Keys containing 'revenue' ---")
    for k in sorted(metric.keys()):
        if "evenue" in k.lower():
            print(f"  {k} = {metric[k]}")

    print("\n--- Keys containing 'ebitda' ---")
    for k in sorted(metric.keys()):
        if "bitda" in k.lower():
            print(f"  {k} = {metric[k]}")

    print("\n--- Keys containing 'debt' or 'cash' or 'fcf' or 'freecash' ---")
    for k in sorted(metric.keys()):
        lk = k.lower()
        if "debt" in lk or "cash" in lk or "fcf" in lk:
            print(f"  {k} = {metric[k]}")

    print("\n--- Keys containing 'pe' or 'ratio' or 'ev' ---")
    for k in sorted(metric.keys()):
        if any(t in k.lower() for t in ["pe", "ratio", "ev"]):
            print(f"  {k} = {metric[k]}")

    # Save full body to data/ for reference
    out = REPO_ROOT / "data" / "finnhub_metric_sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2))
    print(f"\nFull payload saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
