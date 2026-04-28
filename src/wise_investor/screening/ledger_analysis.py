"""Calibration ledger analysis — read-only metrics over JSON entries.

The calibration ledger (`data/calibration_ledger/v*.json`) accumulates
back-validation runs across constitution versions and calibration
dates. This module provides the analysis layer:

  - Confusion-matrix metrics on the rubric-as-binary-classifier:
    precision (of advances, fraction that outperformed the benchmark),
    recall (of winners, fraction that the rubric advanced).

  - Per-ticker classification: TP / FP / TN / FN, where the prediction
    is the rubric's hierarchy_decision and the ground truth is
    excess_return > 0 over the back-validation horizon.

  - Cross-entry comparison: side-by-side diff of two runs (different
    constitution versions, or same version on different calibration
    dates). Highlights verdict flips and excess-return deltas.

The analysis is intentionally read-only: it never mutates the ledger.
A future calibration loop that wants to "vote" on whether to keep a
constitution change should use this module to score, then write a new
constitution version, then back-validate against the same manifest —
the loop's correctness depends on the ledger staying immutable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TickerClassification:
    """Per-ticker binary classification outcome."""

    symbol: str
    predicted_advance: bool        # rubric said ADVANCE_TO_STAGE_3
    actually_outperformed: bool    # excess_return > 0 over horizon
    excess_return: float | None
    classification: str            # "TP" | "FP" | "TN" | "FN" | "UNDEFINED"

    @property
    def is_correct(self) -> bool:
        """True iff the rubric's prediction matched the outcome."""
        return self.classification in {"TP", "TN"}


@dataclass(frozen=True)
class ConfusionMatrix:
    """Aggregate confusion matrix + derived metrics."""

    tp: int   # advanced + outperformed
    fp: int   # advanced + underperformed
    tn: int   # rejected + underperformed
    fn: int   # rejected + outperformed
    undefined: int  # missing excess_return

    @property
    def n_classified(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float | None:
        """Of tickers we advanced, fraction that beat benchmark."""
        denom = self.tp + self.fp
        if denom == 0:
            return None
        return self.tp / denom

    @property
    def recall(self) -> float | None:
        """Of actual winners, fraction the rubric advanced."""
        denom = self.tp + self.fn
        if denom == 0:
            return None
        return self.tp / denom

    @property
    def f1(self) -> float | None:
        p = self.precision
        r = self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> float | None:
        if self.n_classified == 0:
            return None
        return (self.tp + self.tn) / self.n_classified


@dataclass(frozen=True)
class LedgerAnalysis:
    """Full analysis of one ledger entry."""

    constitution_version: str
    calibration_date: str
    horizon_date: str
    n_tickers: int
    n_advanced: int
    n_rejected: int
    advanced_avg_excess_return: float | None
    rejected_avg_excess_return: float | None
    confusion: ConfusionMatrix
    classifications: tuple[TickerClassification, ...]


@dataclass(frozen=True)
class LedgerComparison:
    """Diff between two ledger entries (same manifest preferred)."""

    a_label: str
    b_label: str
    common_symbols: tuple[str, ...]
    flipped_to_advance: tuple[str, ...]    # rejected in A, advanced in B
    flipped_to_reject: tuple[str, ...]     # advanced in A, rejected in B
    a_only_symbols: tuple[str, ...]        # in A but not B
    b_only_symbols: tuple[str, ...]        # in B but not A
    a_metrics: ConfusionMatrix
    b_metrics: ConfusionMatrix


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def load_ledger_entry(path: Path) -> dict[str, Any]:
    """Read one ledger JSON file. Trivial wrapper for symmetry with
    the calibration_ledger module's writer."""
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_entry(entry: dict[str, Any]) -> LedgerAnalysis:
    """Project a ledger payload into a LedgerAnalysis.

    Tickers with `excess_return = None` (price data missing for
    benchmark or ticker) are bucketed as "undefined" — they don't
    contribute to confusion-matrix counts but ARE preserved in the
    classifications tuple so the caller can see they exist.
    """
    classifications: list[TickerClassification] = []
    tp = fp = tn = fn = undefined = 0

    for record in entry.get("per_ticker_records", []):
        sym = record.get("symbol", "?")
        decision = record.get("prefilter", {}).get("hierarchy_decision", "")
        predicted_advance = decision == "ADVANCE_TO_STAGE_3"
        excess = record.get("return_outcome", {}).get("excess_return")

        if excess is None:
            cls = "UNDEFINED"
            actually_out = False
            undefined += 1
        else:
            actually_out = excess > 0
            if predicted_advance and actually_out:
                cls, _inc = "TP", "tp"
                tp += 1
            elif predicted_advance and not actually_out:
                cls = "FP"
                fp += 1
            elif not predicted_advance and not actually_out:
                cls = "TN"
                tn += 1
            else:
                cls = "FN"
                fn += 1

        classifications.append(
            TickerClassification(
                symbol=sym,
                predicted_advance=predicted_advance,
                actually_outperformed=actually_out,
                excess_return=excess,
                classification=cls,
            )
        )

    confusion = ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn, undefined=undefined)

    return LedgerAnalysis(
        constitution_version=entry.get("constitution_version", "?"),
        calibration_date=entry.get("calibration_date", "?"),
        horizon_date=entry.get("horizon_date", "?"),
        n_tickers=entry.get("n_tickers", 0),
        n_advanced=entry.get("n_advanced", 0),
        n_rejected=entry.get("n_rejected", 0),
        advanced_avg_excess_return=entry.get("advanced_avg_excess_return"),
        rejected_avg_excess_return=entry.get("rejected_avg_excess_return"),
        confusion=confusion,
        classifications=tuple(classifications),
    )


def compare_entries(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    *,
    a_label: str | None = None,
    b_label: str | None = None,
) -> LedgerComparison:
    """Diff two ledger entries.

    Common manifest assumed (e.g., same 30 tickers). Tickers in only
    one side are surfaced separately. The comparison is most useful
    when both entries used the same manifest + calibration date but
    different constitution versions — it reveals which tickers the
    rubric change actually flipped.
    """
    a_label = a_label or _label_for(entry_a)
    b_label = b_label or _label_for(entry_b)

    a_decisions = _ticker_decisions(entry_a)
    b_decisions = _ticker_decisions(entry_b)

    a_syms = set(a_decisions.keys())
    b_syms = set(b_decisions.keys())
    common = a_syms & b_syms

    flipped_advance = []
    flipped_reject = []
    for sym in sorted(common):
        a_dec = a_decisions[sym]
        b_dec = b_decisions[sym]
        if a_dec != b_dec:
            if b_dec == "ADVANCE_TO_STAGE_3":
                flipped_advance.append(sym)
            else:
                flipped_reject.append(sym)

    a_metrics = analyze_entry(entry_a).confusion
    b_metrics = analyze_entry(entry_b).confusion

    return LedgerComparison(
        a_label=a_label,
        b_label=b_label,
        common_symbols=tuple(sorted(common)),
        flipped_to_advance=tuple(flipped_advance),
        flipped_to_reject=tuple(flipped_reject),
        a_only_symbols=tuple(sorted(a_syms - b_syms)),
        b_only_symbols=tuple(sorted(b_syms - a_syms)),
        a_metrics=a_metrics,
        b_metrics=b_metrics,
    )


def list_ledger_entries(ledger_dir: Path) -> list[Path]:
    """Return all v*.json files under ledger_dir, sorted chronologically."""
    if not ledger_dir.exists():
        return []
    return sorted(ledger_dir.glob("v*.json"))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _label_for(entry: dict[str, Any]) -> str:
    """Build a short label like 'v2.0 @ 2018-06-30'."""
    v = entry.get("constitution_version", "?")
    d = entry.get("calibration_date", "?")
    return f"v{v} @ {d}"


def _ticker_decisions(entry: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for record in entry.get("per_ticker_records", []):
        sym = record.get("symbol")
        if not sym:
            continue
        decision = record.get("prefilter", {}).get("hierarchy_decision", "")
        out[sym] = decision
    return out


__all__ = [
    "ConfusionMatrix",
    "LedgerAnalysis",
    "LedgerComparison",
    "TickerClassification",
    "analyze_entry",
    "compare_entries",
    "list_ledger_entries",
    "load_ledger_entry",
]
