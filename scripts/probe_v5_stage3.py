"""Inspect Stage 3 LLM verdicts in calibration v5 ledger."""
from __future__ import annotations

import json
from pathlib import Path

LEDGER = Path(__file__).resolve().parents[1] / "data/calibration_ledger/v2.0_2018-06-30_20260430T052705Z.json"


def main() -> int:
    d = json.loads(LEDGER.read_text())
    recs = d["per_ticker_records"]
    print(f"Ledger: {LEDGER.name}")
    print(f"Records: {len(recs)}\n")

    print(f"{'Sym':<10} {'S2 dec':<8} {'S2 moat':<10} {'S3 dec':<22} {'S3 moat':<12} {'S3 frontier':<12} {'S3 bottleneck':<14} {'Excess':>10}")
    print("-" * 120)

    s3_present = 0
    s3_advance = 0
    s3_reject = 0
    moat_rescued = 0  # Stage 2 FAIL → Stage 3 PASS
    s3_axis_invalid = 0
    for r in recs:
        sym = r["symbol"]
        s2 = r["prefilter"]
        s2_dec = s2["hierarchy_decision"].replace("ADVANCE_TO_STAGE_3", "→S3")
        s2_moat = s2["moat"]["verdict"]
        excess = r.get("return_outcome", {}).get("excess_return")
        excess_str = f"{excess:+.1%}" if excess is not None else "—"

        s3 = r.get("stage3")
        if s3 is None:
            s3_dec = "(none)"
            s3_moat = s3_frontier = s3_bottleneck = "—"
        else:
            s3_present += 1
            s3_dec = s3["hierarchy_decision"]
            s3_moat = s3["moat"]["verdict"]
            s3_frontier = s3["new_frontier"]["verdict"]
            s3_bottleneck = s3["bottleneck"]["verdict"]
            if s3_dec == "ADVANCE_TO_STAGE_4":
                s3_advance += 1
            elif s3_dec == "REJECT":
                s3_reject += 1
            if s2_moat == "FAIL" and s3_moat == "PASS":
                moat_rescued += 1
            if s3_moat == "INVALID" or s3_frontier == "INVALID" or s3_bottleneck == "INVALID":
                s3_axis_invalid += 1

        print(f"{sym:<10} {s2_dec:<8} {s2_moat:<10} {s3_dec:<22} {s3_moat:<12} {s3_frontier:<12} {s3_bottleneck:<14} {excess_str:>10}")

    print()
    print("=== Stage 3 summary ===")
    print(f"  records with stage3 block: {s3_present}/{len(recs)}")
    print(f"  S3 ADVANCE_TO_STAGE_4:     {s3_advance}")
    print(f"  S3 REJECT:                 {s3_reject}")
    print(f"  moat FAIL → S3 PASS:       {moat_rescued} (Stage 3 rescue rate)")
    print(f"  records with INVALID axis: {s3_axis_invalid}")

    # Show first INVALID reasoning so we can see what's breaking
    invalid_samples = [r for r in recs if r.get("stage3") and r["stage3"]["moat"]["verdict"] == "INVALID"]
    if invalid_samples:
        print("\n=== First INVALID-moat sample reasoning ===")
        first = invalid_samples[0]
        print(f"Symbol: {first['symbol']}")
        print(f"Reason: {first['stage3']['moat']['reasoning'][:400]}")
        if first['stage3'].get('rejection_reason'):
            print(f"Rejection: {first['stage3']['rejection_reason'][:200]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
