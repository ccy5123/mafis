"""Tests for ledger_analysis — confusion-matrix and comparison logic."""

from __future__ import annotations

from wise_investor.screening.ledger_analysis import (
    analyze_entry,
    compare_entries,
)

# ---------------------------------------------------------------------------
# Synthetic ledger entries
# ---------------------------------------------------------------------------


def _record(
    symbol: str,
    *,
    decision: str = "ADVANCE_TO_STAGE_3",
    excess: float | None = 0.10,
) -> dict:
    return {
        "symbol": symbol,
        "calibration_date": "2018-06-30",
        "horizon_date": "2023-06-30",
        "constitution_version": "2.0",
        "prefilter": {
            "hierarchy_decision": decision,
            "moat": {"verdict": "PASS"},
            "new_frontier": {"verdict": "FAIL"},
            "bottleneck": {"verdict": "PASS"},
            "passed_axes": ["moat", "bottleneck"],
            "need_llm_axes": [],
            "excluded_reason": None,
        },
        "return_outcome": {
            "ticker_return": 0.5,
            "benchmark_return": 0.5 - (excess or 0.0),
            "excess_return": excess,
        },
        "axis_persistence": [],
    }


def _entry(records: list[dict], **overrides) -> dict:
    return {
        "constitution_version": overrides.get("constitution_version", "2.0"),
        "calibration_date": overrides.get("calibration_date", "2018-06-30"),
        "horizon_date": overrides.get("horizon_date", "2023-06-30"),
        "n_tickers": len(records),
        "n_advanced": sum(
            1 for r in records
            if r["prefilter"]["hierarchy_decision"] == "ADVANCE_TO_STAGE_3"
        ),
        "n_rejected": sum(
            1 for r in records
            if r["prefilter"]["hierarchy_decision"] == "REJECT"
        ),
        "advanced_avg_excess_return": overrides.get("advanced_avg_excess_return"),
        "rejected_avg_excess_return": overrides.get("rejected_avg_excess_return"),
        "per_ticker_records": records,
    }


# ---------------------------------------------------------------------------
# analyze_entry — confusion matrix
# ---------------------------------------------------------------------------


def test_analyze_classifies_TP_FP_TN_FN_correctly() -> None:
    records = [
        _record("A", decision="ADVANCE_TO_STAGE_3", excess=0.30),  # TP
        _record("B", decision="ADVANCE_TO_STAGE_3", excess=-0.20), # FP
        _record("C", decision="REJECT", excess=-0.10),             # TN
        _record("D", decision="REJECT", excess=0.50),              # FN
    ]
    result = analyze_entry(_entry(records))
    cm = result.confusion
    assert cm.tp == 1
    assert cm.fp == 1
    assert cm.tn == 1
    assert cm.fn == 1
    assert cm.undefined == 0


def test_analyze_handles_missing_excess_as_undefined() -> None:
    records = [
        _record("A", decision="ADVANCE_TO_STAGE_3", excess=None),
        _record("B", decision="REJECT", excess=None),
    ]
    result = analyze_entry(_entry(records))
    assert result.confusion.undefined == 2
    assert result.confusion.tp == 0
    assert result.confusion.fp == 0


def test_analyze_metric_calculations() -> None:
    records = [
        _record("A", decision="ADVANCE_TO_STAGE_3", excess=0.20),  # TP
        _record("B", decision="ADVANCE_TO_STAGE_3", excess=0.10),  # TP
        _record("C", decision="ADVANCE_TO_STAGE_3", excess=-0.30), # FP
        _record("D", decision="REJECT", excess=-0.10),             # TN
        _record("E", decision="REJECT", excess=0.40),              # FN
    ]
    result = analyze_entry(_entry(records))
    cm = result.confusion
    # Precision = TP / (TP + FP) = 2/3 ≈ 0.6667
    assert cm.precision is not None
    assert abs(cm.precision - 2/3) < 1e-6
    # Recall = TP / (TP + FN) = 2/3 ≈ 0.6667
    assert cm.recall is not None
    assert abs(cm.recall - 2/3) < 1e-6
    # F1 = 2 * 2/3 / (4/3) = 2/3
    assert cm.f1 is not None
    assert abs(cm.f1 - 2/3) < 1e-6


def test_analyze_precision_none_when_no_advances() -> None:
    records = [_record("A", decision="REJECT", excess=-0.1)]
    result = analyze_entry(_entry(records))
    assert result.confusion.precision is None


def test_analyze_recall_none_when_no_winners() -> None:
    records = [
        _record("A", decision="ADVANCE_TO_STAGE_3", excess=-0.1),
        _record("B", decision="REJECT", excess=-0.2),
    ]
    result = analyze_entry(_entry(records))
    assert result.confusion.recall is None


def test_analyze_classifications_preserved_in_output() -> None:
    records = [
        _record("AAA", decision="ADVANCE_TO_STAGE_3", excess=0.20),
        _record("BBB", decision="REJECT", excess=-0.05),
    ]
    result = analyze_entry(_entry(records))
    by_sym = {c.symbol: c for c in result.classifications}
    assert by_sym["AAA"].classification == "TP"
    assert by_sym["BBB"].classification == "TN"


def test_analyze_is_correct_property() -> None:
    records = [
        _record("TP_TICK", decision="ADVANCE_TO_STAGE_3", excess=0.10),
        _record("FP_TICK", decision="ADVANCE_TO_STAGE_3", excess=-0.10),
    ]
    result = analyze_entry(_entry(records))
    by_sym = {c.symbol: c for c in result.classifications}
    assert by_sym["TP_TICK"].is_correct is True
    assert by_sym["FP_TICK"].is_correct is False


# ---------------------------------------------------------------------------
# compare_entries
# ---------------------------------------------------------------------------


def test_compare_detects_verdict_flips() -> None:
    a = _entry([
        _record("X", decision="REJECT", excess=0.10),
        _record("Y", decision="ADVANCE_TO_STAGE_3", excess=0.20),
        _record("Z", decision="ADVANCE_TO_STAGE_3", excess=-0.10),
    ])
    b = _entry([
        _record("X", decision="ADVANCE_TO_STAGE_3", excess=0.10),  # flipped → advance
        _record("Y", decision="REJECT", excess=0.20),               # flipped → reject
        _record("Z", decision="ADVANCE_TO_STAGE_3", excess=-0.10),  # unchanged
    ])
    comparison = compare_entries(a, b)
    assert "X" in comparison.flipped_to_advance
    assert "Y" in comparison.flipped_to_reject
    assert "Z" not in comparison.flipped_to_advance
    assert "Z" not in comparison.flipped_to_reject


def test_compare_handles_disjoint_manifests() -> None:
    a = _entry([_record("AAA", decision="ADVANCE_TO_STAGE_3", excess=0.1)])
    b = _entry([_record("BBB", decision="ADVANCE_TO_STAGE_3", excess=0.1)])
    comparison = compare_entries(a, b)
    assert "AAA" in comparison.a_only_symbols
    assert "BBB" in comparison.b_only_symbols
    assert comparison.common_symbols == ()


def test_compare_metrics_per_side() -> None:
    a = _entry([
        _record("X", decision="ADVANCE_TO_STAGE_3", excess=0.20),
        _record("Y", decision="ADVANCE_TO_STAGE_3", excess=-0.20),
    ])
    b = _entry([
        _record("X", decision="ADVANCE_TO_STAGE_3", excess=0.20),
        _record("Y", decision="REJECT", excess=-0.20),
    ])
    comparison = compare_entries(a, b)
    # B should have higher precision (fewer FPs)
    a_prec = comparison.a_metrics.precision
    b_prec = comparison.b_metrics.precision
    assert a_prec is not None and b_prec is not None
    assert b_prec > a_prec


def test_compare_default_label_combines_version_and_date() -> None:
    a = _entry([_record("X")], constitution_version="2.0", calibration_date="2018-06-30")
    b = _entry([_record("X")], constitution_version="2.1", calibration_date="2018-06-30")
    comparison = compare_entries(a, b)
    assert "v2.0" in comparison.a_label
    assert "v2.1" in comparison.b_label
    assert "2018-06-30" in comparison.a_label
