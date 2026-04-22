"""Inspect NVDA's cash-flow concepts to find the right XBRL tags for capex."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.finnhub import FinnhubClient  # noqa: E402


with FinnhubClient() as c:
    latest = c.latest_annual_financials("NVDA")

print(f"Form={latest.form}  end_date={latest.end_date}  year={latest.year}")
print(f"\nAll cash-flow concepts ({len(latest.report.cf)}):")
for item in latest.report.cf:
    val = item.value
    val_str = f"${val/1e9:.2f}B" if val is not None and abs(val) >= 1e6 else str(val)
    print(f"  {item.concept or '<no concept>':70s}  {val_str}")

print(f"\nAll income concepts ({len(latest.report.ic)}):")
for item in latest.report.ic:
    val = item.value
    if val is not None and abs(val) >= 1e9:
        print(f"  {item.concept or '<no concept>':70s}  ${val/1e9:.2f}B")
