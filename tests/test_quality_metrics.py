"""Tests for Phase 1D automated quality metrics.

Uses synthetic report fragments chosen to exercise each metric's edge cases
without needing a live LLM. Baseline measurements against the real
reports/NVDA_* files are done ad-hoc via scripts/score_report.py — they
are not part of pytest because real reports change with each run.
"""

from __future__ import annotations

from wise_investor.quality.metrics import (
    ALL_METRICS,
    REFUSAL_PHRASES,
    _normalise_number,
    citation_rate,
    hard_vs_scenario,
    invention_audit,
    refusal_count,
    score_report,
    skeptic_coverage,
    vulnerable_link_grounding,
)


# ---------------------------------------------------------------------------
# refusal_count (#5)
# ---------------------------------------------------------------------------


def test_refusal_count_zero_when_no_refusals() -> None:
    r = refusal_count("Analyst says NVDA revenue is $215.94B. Confident analysis.")
    assert r.value == 0.0
    assert r.passed is False


def test_refusal_count_detects_skeptic_phrases() -> None:
    report = """
    ## Skeptic
    - Downside not quantifiable from current facts — details missing.
    - Unknown from current facts — I cannot name one.
    - Not computable from current facts — no peer-median provided.
    - Downside not quantifiable from current facts — another field.
    """
    r = refusal_count(report)
    # "Downside not quantifiable..." x2 + "Unknown..." x1 + "Not computable..." x1 = 4
    assert r.value == 4.0
    assert r.passed is True  # threshold is 3


def test_refusal_count_case_insensitive() -> None:
    r = refusal_count("DOWNSIDE NOT QUANTIFIABLE FROM CURRENT FACTS.")
    assert r.value == 1.0


def test_refusal_phrases_non_empty() -> None:
    assert len(REFUSAL_PHRASES) >= 3


# ---------------------------------------------------------------------------
# citation_rate (#1)
# ---------------------------------------------------------------------------


def test_citation_rate_all_cited() -> None:
    report = """
    - **Revenue**: $215.94B — [Source: fetch.revenue]
    - **Net Income**: $120.07B — [Source: fetch.net_income]
    """
    r = citation_rate(report)
    assert r.value == 1.0
    assert r.passed is True


def test_citation_rate_none_cited() -> None:
    report = "NVDA revenue is $215.94B this year. Net income was $120.07B."
    r = citation_rate(report)
    assert r.value == 0.0
    assert r.passed is False


def test_citation_rate_ignores_code_fences() -> None:
    # Peer multiples tables are inside ``` fences and shouldn't be penalised.
    report = """
    ## Analyst
    - Revenue $100.00B — [Source: fetch.revenue]

    ```plaintext
    Symbol   PER    EV/EBITDA
    NVDA    40.79   35.01
    MSFT    31.10   23.60
    ```
    """
    r = citation_rate(report)
    # Only one numeric line outside the fence; it has a citation.
    assert r.value == 1.0


def test_citation_rate_mixed() -> None:
    report = """
    - **Revenue**: $215.94B — [Source: fetch.revenue]
    - Some narrative claims growth of 20.27% without any source.
    """
    r = citation_rate(report)
    # Numbers: $215.94B (cited), 20.27% (uncited). 1 of 2.
    assert r.value == 0.5


# ---------------------------------------------------------------------------
# vulnerable_link_grounding (#6)
# ---------------------------------------------------------------------------


def test_vulnerable_link_grounding_counts_unique_numbers() -> None:
    report = """
    # Part 3 · Skeptic
    Rebuttal 1 grounds in Vulnerable link #1.
    Rebuttal 2 grounds in Vulnerable link #2.
    Rebuttal 3 grounds in Vulnerable link #1 again (duplicate).
    Rebuttal 4 grounds in Vulnerable link #3.
    Rebuttal 5 grounds in Vulnerable link #6.
    """
    r = vulnerable_link_grounding(report)
    # Distinct links: {1, 2, 3, 6} = 4
    assert r.value == 4.0
    assert r.passed is True  # threshold 3


def test_vulnerable_link_grounding_below_threshold_fails() -> None:
    report = """
    # Part 3 · Skeptic
    Only Vulnerable link #1 mentioned here.
    """
    r = vulnerable_link_grounding(report)
    assert r.value == 1.0
    assert r.passed is False


def test_vulnerable_link_grounding_counts_only_in_skeptic() -> None:
    report = """
    # Part 1 · Analyst
    Per value chain Vulnerable link #1 analysis is fine.
    Per value chain Vulnerable link #2 more context.

    # Part 3 · Skeptic
    Only Vulnerable link #5 in the Skeptic.
    """
    r = vulnerable_link_grounding(report)
    # Skeptic slice starts at "# Part 3 · Skeptic"; only #5 is counted.
    assert r.value == 1.0


# ---------------------------------------------------------------------------
# hard_vs_scenario (#4)
# ---------------------------------------------------------------------------


def test_hard_vs_scenario_pure_facts() -> None:
    # 5 numbers, 0 scenario words → infinite ratio (reported as 99.0)
    report = "Revenue $215.94B, $120.07B net income, EPS 7.46, PER 40.79, EV/EBITDA 35.01."
    r = hard_vs_scenario(report)
    assert r.value == 99.0
    assert r.passed is True


def test_hard_vs_scenario_balanced() -> None:
    report = (
        "Revenue $215.94B could grow at 20%. Net income $120.07B may expand if "
        "demand holds; margins could compress."
    )
    # Numbers: $215.94B, 20%, $120.07B = 3
    # Scenario words: could, may, if, could = 4
    r = hard_vs_scenario(report)
    assert r.value == 0.75  # 3 / 4
    # 0.75 >= 0.5 threshold
    assert r.passed is True


def test_hard_vs_scenario_heavy_hedging_fails() -> None:
    report = (
        "Revenue could grow if conditions hold. Margins might expand, may compress, "
        "could decline. Only one $1.00B hard figure."
    )
    r = hard_vs_scenario(report)
    assert r.value < 0.5
    assert r.passed is False


# ---------------------------------------------------------------------------
# invention_audit (#2)
# ---------------------------------------------------------------------------


def test_invention_audit_all_numbers_sourced_passes() -> None:
    facts = {
        "fetch.revenue": "Source value: $215.94B",
        "calculate_per": "Computed PER: 40.79",
    }
    report = "Revenue $215.94B [Source: fetch.revenue]. PER is 40.79 [Source: calculate_per]."
    r = invention_audit(report, facts)
    assert r.value == 0.0
    assert r.passed is True


def test_invention_audit_catches_fabricated_number() -> None:
    facts = {"fetch.revenue": "Source value: $215.94B"}
    # Report invents a "$20B stock compression" figure not in facts.
    report = (
        "Revenue $215.94B supports thesis. Downside: $20B stock compression "
        "[Source: invented]."
    )
    r = invention_audit(report, facts)
    assert r.value >= 1.0
    # $20B is not 20, not 215.94, not a structural number — should be flagged.
    flagged = r.details["suspected_inventions"]
    assert any("$20B" in t or "20B" in t for t in flagged)


def test_invention_audit_tolerates_rounding() -> None:
    # Source has 20.27%; report rounds to 20.3%.
    facts = {"reverse_dcf": "Implied growth 20.27%"}
    report = "Implied growth ~20.3% per [Source: reverse_dcf]."
    r = invention_audit(report, facts, tolerance_pct=2.0)
    assert r.value == 0.0  # within tolerance


def test_invention_audit_skips_structural_small_integers() -> None:
    facts = {"reverse_dcf": "Implied growth 20.27%"}
    # "5 rebuttals", "100%", "10 years" are structural numbers.
    report = "5 rebuttals follow. Over 10 years, 100% of the thesis hinges on 20.27%."
    r = invention_audit(report, facts)
    # 20.27% matches. 5, 10, 100 are whitelisted.
    assert r.value == 0.0


def test_normalise_number() -> None:
    assert _normalise_number("$215.94B") == 215_940_000_000.0
    assert _normalise_number("120.07M") == 120_070_000.0
    assert _normalise_number("20.27%") == 0.2027
    assert _normalise_number("40.79x") == 40.79
    assert _normalise_number("40.79") == 40.79
    assert _normalise_number("1,234.5") == 1234.5
    assert _normalise_number("-5%") == -0.05


# ---------------------------------------------------------------------------
# skeptic_coverage (#3)
# ---------------------------------------------------------------------------


def test_skeptic_coverage_counts_target_markers() -> None:
    report = """
    # Part 3 · Skeptic
    ### 1. Target claim (Analyst): X
    ### 2. Target claim (Valuer): Y
    ### 3. Target claim (Analyst): Z
    ### 4. Target claim (Analyst|Valuer): W
    ### 5. Target claim (Analyst): V
    """
    r = skeptic_coverage(report)
    assert r.value == 5.0
    assert r.passed is True


def test_skeptic_coverage_below_threshold() -> None:
    report = """
    # Part 3 · Skeptic
    Only one rebuttal here: Target claim (Analyst): X.
    """
    r = skeptic_coverage(report)
    assert r.value == 1.0
    assert r.passed is False


# ---------------------------------------------------------------------------
# score_report aggregate
# ---------------------------------------------------------------------------


def test_score_report_without_facts_skips_invention() -> None:
    report = "Minimal report with no structure."
    results = score_report(report)
    names = [r.name for r in results]
    assert "invention_audit" not in names
    # Other 5 should all be present.
    assert len(results) == 5


def test_score_report_with_facts_includes_invention() -> None:
    facts = {"fetch.revenue": "Source value: $1.00B"}
    report = "Revenue $1.00B [Source: fetch.revenue]."
    results = score_report(report, facts)
    names = [r.name for r in results]
    assert "invention_audit" in names
    assert len(results) == 6


def test_all_metrics_list_matches_score_report() -> None:
    # Sanity check: ALL_METRICS names should all appear in a full scoring.
    facts = {"fact1": "100"}
    results = score_report("x", facts)
    result_names = {r.name for r in results}
    for m in ALL_METRICS:
        assert m in result_names
