"""Inspect the structure of a calibration ledger file."""
from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "data/calibration_ledger/v2.0_2018-06-30_20260430T005948Z.json"
d = json.loads(p.read_text())
print("top keys:", list(d.keys()))
for k in list(d.keys())[:6]:
    v = d[k]
    if isinstance(v, list):
        print(f"  {k}: list, len={len(v)}")
        if v and isinstance(v[0], dict):
            print(f"    first elem keys: {list(v[0].keys())}")
    elif isinstance(v, dict):
        print(f"  {k}: dict, keys={list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")
