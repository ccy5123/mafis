"""Why does SPY advance to Stage 3 despite being an ETF?"""
from __future__ import annotations

import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "data/calibration_ledger/v2.0_2018-06-30_20260430T011543Z.json"
d = json.loads(p.read_text())
recs = d["per_ticker_records"]
spy = next((r for r in recs if r["symbol"] == "SPY"), None)
if spy is None:
    print("SPY not in ledger")
    raise SystemExit(0)
import pprint
pprint.pprint(spy, depth=4, width=120)
