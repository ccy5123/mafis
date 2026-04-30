"""Sample one ledger record to discover the per_ticker schema."""
from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "data/calibration_ledger/v2.0_2018-06-30_20260430T005948Z.json"
d = json.loads(p.read_text())
recs = d["per_ticker_records"]
print(f"records: {len(recs)}")
r0 = recs[0]
print(f"first record keys: {list(r0.keys())}")
for k, v in r0.items():
    if isinstance(v, dict):
        print(f"  {k}: dict, keys={list(v.keys())}")
    elif isinstance(v, list):
        print(f"  {k}: list, len={len(v)}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
