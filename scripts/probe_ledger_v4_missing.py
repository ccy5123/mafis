"""Identify which manifest symbols are missing from a ledger run."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "data/calibration_ledger/v2.0_2018-06-30_20260430T020830Z.json"
MANIFEST = REPO_ROOT / "data/calibration/manifest.yaml"

def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text())
    universe = {t["symbol"] for t in manifest["tickers"]}
    ledger = json.loads(LEDGER.read_text())
    seen = {r["symbol"] for r in ledger["per_ticker_records"]}
    missing = sorted(universe - seen)
    extra = sorted(seen - universe)
    print(f"Manifest size: {len(universe)}")
    print(f"Ledger size:   {len(seen)}")
    print(f"Missing from ledger: {missing}")
    print(f"Extra in ledger (shouldn't happen): {extra}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
