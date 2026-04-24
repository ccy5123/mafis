"""Parse crew reports → structured signals → side-by-side diff.

Pipeline:

    text (crew.md)
        │
        └─→ extract_report_signals()
                │
                ├── parse_crew_report()       # verdict, conviction, audit flag
                ├── all 6 quality metrics     # citation_rate, refusal_count, …
                └── count edgar citations, audit violations

    (signals_baseline, signals_new)
        │
        └─→ compare_reports()
                │
                └── per-field MetricDelta (baseline, new, change, direction)

Directions:
  IMPROVED  — metric moved in the "better" direction
  REGRESSED — metric moved in the "worse" direction
  UNCHANGED — same value (within tolerance)
  NEUTRAL   — not clear which direction is "better" (e.g. conviction
              number is neither good nor bad in isolation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from wise_investor.paper_trading.report_parser import parse_crew_report
from wise_investor.quality.metrics import score_report


# Human-friendly direction labels used by the CLI and tests.
IMPROVED = "IMPROVED"
REGRESSED = "REGRESSED"
UNCHANGED = "UNCHANGED"
NEUTRAL = "NEUTRAL"


@dataclass
class MetricDelta:
    """One field's baseline → new comparison."""

    name: str
    baseline: Any
    new: Any
    direction: str  # IMPROVED / REGRESSED / UNCHANGED / NEUTRAL
    note: str = ""


@dataclass
class ReportDiff:
    """Full comparison of two crew reports."""

    symbol_baseline: str
    symbol_new: str
    deltas: list[MetricDelta] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.direction == REGRESSED]

    @property
    def improvements(self) -> list[MetricDelta]:
        return [d for d in self.deltas if d.direction == IMPROVED]


@dataclass
class _Signals:
    """Parsed signals from a single crew report."""

    symbol: str
    verdict: str | None
    conviction: int | None
    original_verdict: str | None
    original_conviction: int | None
    audit_downgraded: bool
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    metrics_passed: dict[str, bool] = field(default_factory=dict)
    edgar_citation_count: int = 0
    citation_audit_violations: int = 0


def extract_report_signals(report_text: str, symbol_hint: str | None = None) -> _Signals:
    """Parse a crew report into the structured signals we compare."""
    parsed = parse_crew_report(report_text, symbol_hint=symbol_hint)

    metrics: dict[str, float | int | None] = {}
    passed: dict[str, bool] = {}
    try:
        results = score_report(report_text)
        for r in results:
            metrics[r.name] = r.value
            passed[r.name] = bool(r.passed)
    except Exception:
        # Parsing metrics is best-effort; keep comparing even if one
        # metric crashes on a malformed report.
        pass

    edgar_count = len(
        re.findall(
            r"\[Source:\s*(?:edgar\.[a-z_]+|10-K\s+[a-z_]+)",
            report_text,
            re.IGNORECASE,
        )
    )

    # Count "ungrounded" or "mis-translation" notes in the System
    # Audit block — the key signal of discipline failures.
    audit_violation_count = len(
        re.findall(
            r"(?:ungrounded|MIS-TRANSLATION|VIOLATION:)",
            report_text,
            re.IGNORECASE,
        )
    )

    return _Signals(
        symbol=parsed.symbol,
        verdict=parsed.verdict,
        conviction=parsed.conviction,
        original_verdict=parsed.original_verdict,
        original_conviction=parsed.original_conviction,
        audit_downgraded=parsed.audit_downgraded,
        metrics=metrics,
        metrics_passed=passed,
        edgar_citation_count=edgar_count,
        citation_audit_violations=audit_violation_count,
    )


# ---------------------------------------------------------------------------
# Direction inference
# ---------------------------------------------------------------------------


# Metrics where HIGHER is better.
_HIGHER_IS_BETTER = {
    "citation_rate",
    "refusal_count",
    "vulnerable_link_grounding",
    "hard_vs_scenario",
    "skeptic_coverage",
    "edgar_citation_count",
}

# Metrics where LOWER is better.
_LOWER_IS_BETTER = {
    "invention_audit",
    "citation_audit_violations",
}


def _direction_for_number(name: str, baseline: float, new: float, tol: float = 1e-6) -> str:
    """Return IMPROVED / REGRESSED / UNCHANGED for a numeric metric."""
    if abs(baseline - new) <= tol:
        return UNCHANGED
    better = new > baseline
    if name in _HIGHER_IS_BETTER:
        return IMPROVED if better else REGRESSED
    if name in _LOWER_IS_BETTER:
        return REGRESSED if better else IMPROVED
    return NEUTRAL


# Verdict ordering for direction inference. Stricter (PASS) is considered
# "better" than BUY when the baseline BUY was overclaimed, but without
# context we label verdict changes NEUTRAL — the comparison tool's user
# decides whether HOLD → BUY is a regression (if ground-truth is clearly
# "no change in thesis") or improvement.
_VERDICT_RANK = {"PASS": 0, "HOLD": 1, "BUY": 2}


def compare_reports(baseline_text: str, new_text: str) -> ReportDiff:
    """Diff two crew reports on their structured signals.

    Tolerates differing symbols (e.g. compare NVDA baseline to AMD
    new) — the deltas still surface, with a note flagging the symbol
    change.
    """
    bs = extract_report_signals(baseline_text)
    ns = extract_report_signals(new_text)
    diff = ReportDiff(symbol_baseline=bs.symbol, symbol_new=ns.symbol)

    if bs.symbol and ns.symbol and bs.symbol != ns.symbol:
        diff.deltas.append(
            MetricDelta(
                name="symbol",
                baseline=bs.symbol,
                new=ns.symbol,
                direction=NEUTRAL,
                note="different tickers — interpret other deltas carefully",
            )
        )

    # ---- Verdict / conviction (NEUTRAL — context-dependent) ----
    if bs.verdict != ns.verdict:
        diff.deltas.append(
            MetricDelta(
                name="verdict",
                baseline=bs.verdict,
                new=ns.verdict,
                direction=NEUTRAL,
                note="verdict flipped — inspect Rationale to decide if this is an improvement",
            )
        )
    if bs.conviction != ns.conviction:
        diff.deltas.append(
            MetricDelta(
                name="conviction",
                baseline=bs.conviction,
                new=ns.conviction,
                direction=NEUTRAL,
            )
        )

    # ---- Audit downgrade flag ----
    # An audit DOWNGRADE is a sign the discipline system caught
    # something. Compared across two runs, a new downgrade where
    # the baseline was clean MIGHT be a regression in the LLM's
    # self-discipline (but not in the audit itself). Mark NEUTRAL and
    # let the human judge.
    if bs.audit_downgraded != ns.audit_downgraded:
        diff.deltas.append(
            MetricDelta(
                name="audit_downgraded",
                baseline=bs.audit_downgraded,
                new=ns.audit_downgraded,
                direction=NEUTRAL,
                note=(
                    "LLM's verdict alignment with the discipline matrix "
                    "changed; check whether the Defender labels flipped."
                ),
            )
        )

    # ---- Numeric quality metrics ----
    all_metric_names = set(bs.metrics) | set(ns.metrics)
    for name in sorted(all_metric_names):
        b_val = bs.metrics.get(name)
        n_val = ns.metrics.get(name)
        if b_val is None or n_val is None:
            diff.deltas.append(
                MetricDelta(
                    name=f"metric:{name}",
                    baseline=b_val,
                    new=n_val,
                    direction=NEUTRAL,
                    note="metric missing in one of the reports",
                )
            )
            continue
        direction = _direction_for_number(name, float(b_val), float(n_val))
        if direction == UNCHANGED:
            continue
        diff.deltas.append(
            MetricDelta(
                name=f"metric:{name}",
                baseline=b_val,
                new=n_val,
                direction=direction,
            )
        )

    # ---- Citation counts ----
    if bs.edgar_citation_count != ns.edgar_citation_count:
        diff.deltas.append(
            MetricDelta(
                name="edgar_citation_count",
                baseline=bs.edgar_citation_count,
                new=ns.edgar_citation_count,
                direction=_direction_for_number(
                    "edgar_citation_count",
                    bs.edgar_citation_count,
                    ns.edgar_citation_count,
                ),
            )
        )
    if bs.citation_audit_violations != ns.citation_audit_violations:
        diff.deltas.append(
            MetricDelta(
                name="citation_audit_violations",
                baseline=bs.citation_audit_violations,
                new=ns.citation_audit_violations,
                direction=_direction_for_number(
                    "citation_audit_violations",
                    bs.citation_audit_violations,
                    ns.citation_audit_violations,
                ),
            )
        )

    return diff


def render_diff_markdown(diff: ReportDiff) -> str:
    """Render the ReportDiff as a readable markdown summary."""
    if not diff.deltas:
        return "_No material differences between the reports._\n"

    lines = ["# Regression diff", ""]
    lines.append(
        f"- Baseline symbol: **{diff.symbol_baseline or '(unknown)'}**"
    )
    lines.append(
        f"- New symbol: **{diff.symbol_new or '(unknown)'}**"
    )
    lines.append("")

    # Group by direction for quick scanning.
    grouped: dict[str, list[MetricDelta]] = {}
    for d in diff.deltas:
        grouped.setdefault(d.direction, []).append(d)

    for direction in (REGRESSED, NEUTRAL, IMPROVED, UNCHANGED):
        entries = grouped.get(direction, [])
        if not entries:
            continue
        lines.append(f"## {direction} ({len(entries)})")
        for d in entries:
            lines.append(
                f"- **{d.name}**: `{d.baseline!r}` → `{d.new!r}`"
                + (f"  _({d.note})_" if d.note else "")
            )
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "IMPROVED",
    "NEUTRAL",
    "REGRESSED",
    "UNCHANGED",
    "MetricDelta",
    "ReportDiff",
    "compare_reports",
    "extract_report_signals",
    "render_diff_markdown",
]
