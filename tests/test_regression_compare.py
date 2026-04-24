"""Tests for the Phase 4 regression-diff tool."""

from __future__ import annotations

from wise_investor.regression.compare import (
    IMPROVED,
    NEUTRAL,
    REGRESSED,
    UNCHANGED,
    MetricDelta,
    _direction_for_number,
    compare_reports,
    extract_report_signals,
    render_diff_markdown,
)


# Minimal but valid crew-report fixtures.
def _report(
    symbol: str = "NVDA",
    verdict: str = "BUY",
    conviction: int = 4,
    extra: str = "",
) -> str:
    body = f"""\
# {symbol} — Equity Research Note

# Part 6 · Steward

## Verdict
{verdict}

## Conviction Level
Conviction: {conviction}

## Rationale
- **NEUTRALIZED**: Cash flow strong [Source: fetch.free_cash_flow].
- **NEUTRALIZED**: Moat durable [Source: edgar.moat_signals, filed 2026-02-25].
"""
    return body + extra


# ---------------------------------------------------------------------------
# Direction inference
# ---------------------------------------------------------------------------


def test_direction_higher_is_better_citation_rate() -> None:
    assert _direction_for_number("citation_rate", 0.73, 0.85) == IMPROVED
    assert _direction_for_number("citation_rate", 0.85, 0.73) == REGRESSED
    assert _direction_for_number("citation_rate", 0.80, 0.80) == UNCHANGED


def test_direction_lower_is_better_invention_audit() -> None:
    # invention_audit: lower = fewer hallucinations
    assert _direction_for_number("invention_audit", 5, 2) == IMPROVED
    assert _direction_for_number("invention_audit", 2, 5) == REGRESSED


def test_direction_unknown_metric_is_neutral() -> None:
    assert _direction_for_number("random_unknown", 1, 2) == NEUTRAL


# ---------------------------------------------------------------------------
# extract_report_signals
# ---------------------------------------------------------------------------


def test_extract_signals_parses_verdict_and_conviction() -> None:
    signals = extract_report_signals(_report(verdict="BUY", conviction=4))
    assert signals.symbol == "NVDA"
    assert signals.verdict == "BUY"
    assert signals.conviction == 4


def test_extract_signals_counts_edgar_citations() -> None:
    report = _report() + "\n[Source: edgar.risk_factors, filed 2026-02-25]\n"
    signals = extract_report_signals(report)
    # The fixture already has one edgar citation, plus the appended
    # one — so at least 2 matches.
    assert signals.edgar_citation_count >= 2


def test_extract_signals_counts_audit_violations() -> None:
    report = _report() + (
        "\n- VIOLATION: some mismatch\n- MIS-TRANSLATION: another\n"
    )
    signals = extract_report_signals(report)
    # Two matches: VIOLATION + MIS-TRANSLATION.
    assert signals.citation_audit_violations >= 2


# ---------------------------------------------------------------------------
# compare_reports end-to-end
# ---------------------------------------------------------------------------


def test_compare_identical_reports_yields_no_deltas() -> None:
    text = _report()
    diff = compare_reports(text, text)
    # Metrics are identical → no deltas surfaced.
    assert diff.deltas == []


def test_compare_flags_verdict_change_as_neutral() -> None:
    baseline = _report(verdict="BUY", conviction=4)
    new = _report(verdict="HOLD", conviction=2)
    diff = compare_reports(baseline, new)
    verdict_deltas = [d for d in diff.deltas if d.name == "verdict"]
    assert len(verdict_deltas) == 1
    assert verdict_deltas[0].baseline == "BUY"
    assert verdict_deltas[0].new == "HOLD"
    # Verdict change is context-dependent → NEUTRAL direction.
    assert verdict_deltas[0].direction == NEUTRAL


def test_compare_flags_conviction_change() -> None:
    baseline = _report(verdict="BUY", conviction=4)
    new = _report(verdict="BUY", conviction=3)
    diff = compare_reports(baseline, new)
    conv_deltas = [d for d in diff.deltas if d.name == "conviction"]
    assert len(conv_deltas) == 1
    assert conv_deltas[0].baseline == 4
    assert conv_deltas[0].new == 3


def test_compare_flags_edgar_citation_count_change() -> None:
    baseline = _report()
    # Add an extra edgar citation in the new report.
    new = _report() + "\nExtra [Source: edgar.business_segments, filed 2026-02-25]\n"
    diff = compare_reports(baseline, new)
    citation_deltas = [
        d for d in diff.deltas if d.name == "edgar_citation_count"
    ]
    assert len(citation_deltas) == 1
    # More citations = IMPROVED (higher is better for edgar grounding).
    assert citation_deltas[0].direction == IMPROVED


def test_compare_flags_audit_violations_increase_as_regression() -> None:
    baseline = _report()
    new = _report() + "\n- VIOLATION: claim X not grounded\n"
    diff = compare_reports(baseline, new)
    violation_deltas = [
        d for d in diff.deltas if d.name == "citation_audit_violations"
    ]
    assert len(violation_deltas) == 1
    # More violations = REGRESSED (lower is better for violations).
    assert violation_deltas[0].direction == REGRESSED


def test_compare_flags_symbol_mismatch_with_note() -> None:
    baseline = _report(symbol="NVDA")
    new = _report(symbol="AMD")
    diff = compare_reports(baseline, new)
    symbol_deltas = [d for d in diff.deltas if d.name == "symbol"]
    assert len(symbol_deltas) == 1
    assert "different tickers" in symbol_deltas[0].note.lower()


def test_regressions_property_returns_only_regressed() -> None:
    diff = compare_reports(
        _report() + "\n- VIOLATION: x\n",
        _report() + "\n- VIOLATION: x\n- VIOLATION: y\n",
    )
    assert all(d.direction == REGRESSED for d in diff.regressions)


def test_improvements_property_returns_only_improved() -> None:
    # Extra edgar citation in the new report → improvement.
    baseline = _report()
    new = _report() + "\n[Source: edgar.mdna_highlights, filed 2026-02-25]\n"
    diff = compare_reports(baseline, new)
    assert all(d.direction == IMPROVED for d in diff.improvements)


# ---------------------------------------------------------------------------
# render_diff_markdown
# ---------------------------------------------------------------------------


def test_render_markdown_no_deltas() -> None:
    from wise_investor.regression.compare import ReportDiff

    md = render_diff_markdown(
        ReportDiff(symbol_baseline="NVDA", symbol_new="NVDA")
    )
    assert "No material differences" in md


def test_render_markdown_groups_by_direction() -> None:
    from wise_investor.regression.compare import ReportDiff

    diff = ReportDiff(symbol_baseline="NVDA", symbol_new="NVDA")
    diff.deltas.extend([
        MetricDelta(name="citation_rate", baseline=0.73, new=0.85, direction=IMPROVED),
        MetricDelta(name="verdict", baseline="BUY", new="HOLD", direction=NEUTRAL),
        MetricDelta(
            name="citation_audit_violations", baseline=0, new=2, direction=REGRESSED
        ),
    ])
    md = render_diff_markdown(diff)
    assert "## REGRESSED" in md
    assert "## IMPROVED" in md
    assert "## NEUTRAL" in md
    # REGRESSED section appears before IMPROVED in the rendering
    # (most important first).
    assert md.find("## REGRESSED") < md.find("## IMPROVED")
