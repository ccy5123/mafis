"""One-off: list ASML's revenue/sales us-gaap concepts to find the right tag."""
from __future__ import annotations

import httpx

UA = {"User-Agent": "MAFIS research ccy5123ccy@gmail.com"}
URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000937966.json"


def main() -> int:
    data = httpx.get(URL, headers=UA, timeout=30.0).json()
    usgaap = data["facts"].get("us-gaap", {})
    revs = [k for k in usgaap if "Revenue" in k or "Sales" in k]
    print(f"Revenue/Sales-like us-gaap concepts on ASML ({len(revs)} found):")
    for k in revs:
        units = usgaap[k].get("units", {})
        yrs: set[int] = set()
        for items in units.values():
            for it in items:
                if it.get("fp") == "FY" and it.get("fy") is not None:
                    yrs.add(int(it["fy"]))
        if yrs:
            ys = sorted(yrs)
            print(f"  {k}: {ys[0]}-{ys[-1]}, n={len(ys)}")
        else:
            print(f"  {k}: (no FY data)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
