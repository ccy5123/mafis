"""Probe what currency units ASML/TSM/NVO actually report in."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.edgar_facts import fetch_company_facts  # noqa: E402

CIKS = {
    "TSM": "0001046179",
    "ASML": "0000937966",
    "NVO": "0000353278",
}


def main() -> int:
    for sym, cik in CIKS.items():
        print(f"=== {sym} (CIK {cik}) ===")
        facts = fetch_company_facts(cik)
        ns_map = facts.get("facts", {}) or {}
        # Pick a few representative concepts to check units
        check_concepts = [
            ("us-gaap", "Assets"),
            ("ifrs-full", "Assets"),
            ("us-gaap", "OperatingIncomeLoss"),
            ("ifrs-full", "ProfitLossFromOperatingActivities"),
            ("us-gaap", "Revenues"),
            ("ifrs-full", "Revenue"),
            ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
        ]
        for ns, concept in check_concepts:
            block = (ns_map.get(ns, {}) or {}).get(concept)
            if not block:
                continue
            units = block.get("units", {}) or {}
            print(f"  {ns}:{concept} units={list(units.keys())}")
            for unit, items in list(units.items())[:1]:
                fy_items = [i for i in items if i.get("fp") == "FY"]
                if fy_items:
                    sample = fy_items[0]
                    print(f"    {unit} sample fy={sample.get('fy')} val={sample.get('val')}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
