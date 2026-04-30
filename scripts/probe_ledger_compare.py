"""Compare two calibration ledger entries side by side.

P2 (2026-04): Compare the post-P1a-Full ledger (with-rag, ADR fallback,
KR quarterly, explicit industry filter) against the Option A baseline
to surface which axis verdicts changed and quantify the precision/
recall delta.

Manual confusion matrix: a ticker is a TRUE POSITIVE when the rubric
ADVANCED it AND its 5-year excess return >= 0 (beat S&P 500). FALSE
POSITIVE = ADVANCED but underperformed. TRUE NEGATIVE = REJECTED and
underperformed. FALSE NEGATIVE = REJECTED but outperformed.

This is a Stage-2-only metric. Stage 3 LLM hasn't been called in
either ledger — neither contributes to constitution adjustments yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

OLD = REPO_ROOT / "data/calibration_ledger/v2.0_2018-06-30_20260429T011825Z.json"
NEW = REPO_ROOT / "data/calibration_ledger/v2.0_2018-06-30_20260430T005948Z.json"


def _confusion(records: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    rejected_no_outcome = 0
    advanced_no_outcome = 0
    for r in records:
        decision = r.get("prefilter", {}).get("hierarchy_decision")
        excess = r.get("return_outcome", {}).get("excess_return")
        if excess is None:
            if decision == "ADVANCE_TO_STAGE_3":
                advanced_no_outcome += 1
            else:
                rejected_no_outcome += 1
            continue
        outperformed = excess >= 0
        advanced = decision == "ADVANCE_TO_STAGE_3"
        if advanced and outperformed:
            tp += 1
        elif advanced and not outperformed:
            fp += 1
        elif not advanced and outperformed:
            fn += 1
        else:
            tn += 1
    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall
        else None
    )
    accuracy = (tp + tn) / n if n else None
    return {
        "n": n,
        "advanced_n": tp + fp,
        "rejected_n": tn + fn,
        "advanced_no_outcome": advanced_no_outcome,
        "rejected_no_outcome": rejected_no_outcome,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _per_ticker(records: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in records:
        sym = r.get("symbol")
        out[sym] = {
            "decision": r.get("prefilter", {}).get("hierarchy_decision"),
            "moat": r.get("prefilter", {}).get("moat", {}).get("verdict"),
            "frontier": r.get("prefilter", {}).get("new_frontier", {}).get("verdict"),
            "bottleneck": r.get("prefilter", {}).get("bottleneck", {}).get("verdict"),
            "excess": r.get("return_outcome", {}).get("excess_return"),
        }
    return out


def main() -> int:
    old = json.loads(OLD.read_text())
    new = json.loads(NEW.read_text())

    old_records = old.get("per_ticker_records", [])
    new_records = new.get("per_ticker_records", [])

    print(f"OLD ledger: {OLD.name}")
    print(f"NEW ledger: {NEW.name}")
    print()

    old_pt = _per_ticker(old_records)
    new_pt = _per_ticker(new_records)

    all_symbols = sorted(set(old_pt.keys()) | set(new_pt.keys()))
    print(f"{'Symbol':<12} {'OLD decision':<18} {'OLD moat':<12} {'NEW decision':<18} {'NEW moat':<12} {'Excess':>10}")
    print("-" * 96)
    for sym in all_symbols:
        o = old_pt.get(sym, {})
        n = new_pt.get(sym, {})
        excess = n.get("excess") if n.get("excess") is not None else o.get("excess")
        excess_str = f"{excess:+.1%}" if excess is not None else "—"
        old_dec = o.get("decision") or "MISSING"
        new_dec = n.get("decision") or "MISSING"
        old_moat = o.get("moat") or "—"
        new_moat = n.get("moat") or "—"
        # Highlight changes
        marker = ""
        if old_dec != new_dec:
            marker = "  *DECISION*"
        elif old_moat != new_moat:
            marker = "  *moat*"
        print(f"{sym:<12} {old_dec:<18} {old_moat:<12} {new_dec:<18} {new_moat:<12} {excess_str:>10}{marker}")

    print()
    print("=== Confusion matrix ===")
    print(f"{'':30} {'OLD (Option A)':>16} {'NEW (P1*+RAG)':>16}")
    old_c = _confusion(old_records)
    new_c = _confusion(new_records)
    for key in ("n", "advanced_n", "rejected_n", "tp", "fp", "tn", "fn"):
        print(f"{key:30} {old_c[key]:>16} {new_c[key]:>16}")
    for key in ("precision", "recall", "f1", "accuracy"):
        ov = old_c[key]
        nv = new_c[key]
        ov_s = f"{ov:.3f}" if ov is not None else "—"
        nv_s = f"{nv:.3f}" if nv is not None else "—"
        print(f"{key:30} {ov_s:>16} {nv_s:>16}")
    print()
    print(f"OLD missing ({old_c['rejected_no_outcome']} rejected, {old_c['advanced_no_outcome']} advanced) — no excess return")
    print(f"NEW missing ({new_c['rejected_no_outcome']} rejected, {new_c['advanced_no_outcome']} advanced) — no excess return")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
