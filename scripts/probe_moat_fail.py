"""P3-4: classify *why* each moat=FAIL ticker failed.

Reads the latest calibration ledger and breaks the moat FAILs into
the exact §10 sub-cause: insufficient history (Auto-PASS 1 sentinel),
ROIC advantage <5pp, advantage trend eroding >0.5pp/yr, GM
variability >1.2x industry, or some combination.

Output is informational — feeds the conversation about whether §10
thresholds are too strict, or whether peer aggregator's
industry_median is producing inflated benchmarks.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEDGER = REPO_ROOT / "data/calibration_ledger/v2.0_2018-06-30_20260430T020830Z.json"


def main() -> int:
    d = json.loads(LEDGER.read_text())
    print(f"Ledger: {LEDGER.name}\n")
    print(f"{'Symbol':<10} {'Moat':<10} {'Reason snippet':<60}")
    print("-" * 90)
    fail_rows: list[tuple[str, dict, str]] = []
    for r in d["per_ticker_records"]:
        sym = r["symbol"]
        moat = r["prefilter"]["moat"]
        verdict = moat["verdict"]
        details = moat.get("details", {}) or {}
        reason = (moat.get("reason") or "").replace("\n", " ")[:60]
        if verdict == "FAIL":
            fail_rows.append((sym, details, moat.get("reason") or ""))
        print(f"{sym:<10} {verdict:<10} {reason}")
    print()
    print(f"=== moat=FAIL diagnostic ({len(fail_rows)} tickers) ===")
    print(
        f"{'Sym':<10} {'roic_3y':>10} {'industry':>10} {'advantage':>10} "
        f"{'trend':>10} {'gm_ratio':>10}"
    )
    print("-" * 70)

    cause_counts: dict[str, int] = {}
    for sym, details, reason in fail_rows:
        roic = details.get("roic_3y_avg")
        industry = details.get("industry_roic_3y_median")
        advantage = details.get("roic_advantage")
        trend = details.get("roic_advantage_trend")
        gm_ratio = details.get("gross_margin_industry_ratio")
        def f(v):
            if v is None:
                return "—"
            if isinstance(v, (int, float)):
                if abs(v) < 1.0:
                    return f"{v:.3f}"
                return f"{v:.2f}"
            return str(v)
        print(f"{sym:<10} {f(roic):>10} {f(industry):>10} {f(advantage):>10} {f(trend):>10} {f(gm_ratio):>10}")

        # Categorize the failure cause
        if "auto-PASS 1" in reason.lower() or "<3 years" in reason.lower():
            cause_counts["insufficient_history"] = cause_counts.get("insufficient_history", 0) + 1
        elif advantage is not None and advantage < 0.05:
            cause_counts["low_advantage"] = cause_counts.get("low_advantage", 0) + 1
        elif trend is not None and trend < -0.005:
            cause_counts["eroding_trend"] = cause_counts.get("eroding_trend", 0) + 1
        elif gm_ratio is not None and gm_ratio > 1.2:
            cause_counts["high_gm_variability"] = cause_counts.get("high_gm_variability", 0) + 1
        else:
            cause_counts["other_unknown"] = cause_counts.get("other_unknown", 0) + 1

    print()
    print("Cause breakdown:")
    for cause, n in sorted(cause_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cause}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
