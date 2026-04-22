"""Diagnostic: dump raw JSON from FMP /quote and /key-metrics to identify
which fields are actually present. Used to debug the Phase 1A findings where
pe and ev_to_ebitda came through as None.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402


BASE = "https://financialmodelingprep.com/stable"


def dump(path: str, **params: object) -> None:
    full = {**params, "apikey": settings.fmp_api_key}
    r = httpx.get(f"{BASE}{path}", params=full, timeout=10.0)
    r.raise_for_status()
    body = r.json()

    if isinstance(body, list) and body:
        first = body[0]
        print(f"\n=== {path} params={ {k: v for k, v in params.items()} } ===")
        print(f"Response: list len={len(body)}")
        print(f"First item keys ({len(first.keys())}):")
        for k in sorted(first.keys()):
            v = first[k]
            preview = str(v)[:60] if v is not None else "None"
            print(f"  {k:40s} = {preview}")
    elif isinstance(body, dict):
        print(f"\n=== {path} ===")
        print(f"Dict keys ({len(body.keys())}):")
        for k in sorted(body.keys()):
            print(f"  {k}")
    else:
        print(f"\n=== {path} ===")
        print(f"Unexpected: {body}")


if __name__ == "__main__":
    dump("/quote", symbol="AAPL")
    dump("/key-metrics", symbol="AAPL", period="annual", limit=3)
    dump("/ratios", symbol="AAPL", period="annual", limit=3)
