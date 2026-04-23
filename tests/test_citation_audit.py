"""Tests for the citation grounding audit.

Verifies that `[Source: edgar.*]` citations in the combined crew report
can be matched against the indexed 10-K passages (or flagged when they
cannot). Uses a stub `search_fn` so the suite runs fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wise_investor.quality.citation_audit import (
    audit_edgar_citations,
    render_citation_audit_section,
)


# --- Stub passage type (mirrors PassageHit interface used by the audit) ---


@dataclass
class _FakeHit:
    text: str
    distance: float = 0.5


def _make_search(passages_by_query: dict[tuple[str | None, str | None], list[_FakeHit]]):
    """Return a stub search_fn that returns pre-canned hits keyed by
    (section, <marker substring found in query>).
    Tests can supply distinct passage sets for different sections.
    """

    def _stub(
        query: str,
        symbol: str,
        section: str | None = None,
        k: int = 5,
    ) -> list[_FakeHit]:
        # Look up by exact (section, None) first, then (None, None) fallback.
        hits = passages_by_query.get((section, None))
        if hits is None:
            hits = passages_by_query.get((None, None), [])
        return hits[:k]

    return _stub


# ---------------------------------------------------------------------------
# Extraction + sentence slicing
# ---------------------------------------------------------------------------


def test_audit_reports_no_citations_gracefully() -> None:
    result = audit_edgar_citations("Plain report with no edgar refs.", symbol="NVDA")
    assert result.citations_checked == 0
    assert result.ungrounded == []
    assert not result.violation


def test_audit_extracts_edgar_sentence_and_skips_no_claim_sentences() -> None:
    text = (
        "NVIDIA designs GPUs for data centers [Source: edgar.business_segments].\n"
    )
    search_fn = _make_search(
        {(None, None): [_FakeHit(text="NVIDIA designs GPUs", distance=0.2)]}
    )
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.citations_checked == 1
    # Sentence has no numeric claim — not flagged.
    assert result.ungrounded == []


# ---------------------------------------------------------------------------
# Grounded vs ungrounded claims
# ---------------------------------------------------------------------------


def test_audit_marks_number_present_in_passage_as_grounded() -> None:
    text = (
        "Gross margin decreased to 71.1% in fiscal year 2026 "
        "[Source: edgar.mdna_highlights]."
    )
    search_fn = _make_search(
        {
            ("mdna", None): [
                _FakeHit(
                    text=(
                        "Gross margins decreased to 71.1% in fiscal year 2026 "
                        "from 75.0% in fiscal year 2025"
                    ),
                    distance=0.3,
                )
            ]
        }
    )
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.citations_checked == 1
    assert result.ungrounded == []


def test_audit_flags_hallucinated_percentage_range() -> None:
    """The run-2 failure: Valuer wrote '15-20%' with edgar.mdna_highlights
    citation, but the number does not appear in the retrieved passages.
    """
    text = (
        "The implied FCF growth rate is significantly higher than "
        "historical semiconductor company growth rates, which have "
        "averaged around 15-20% [Source: edgar.mdna_highlights]."
    )
    search_fn = _make_search(
        {
            ("mdna", None): [
                _FakeHit(
                    text=(
                        "Cost of revenue consists of the cost of "
                        "semiconductors, including wafer fabrication..."
                    ),
                    distance=1.4,
                )
            ]
        }
    )
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.citations_checked == 1
    # Two ungrounded claims produced because the range expansion finds
    # both "15-20%" and single-endpoint "20%" absent from passage.
    ungrounded_claims = {u.claim_number for u in result.ungrounded}
    assert "15-20%" in ungrounded_claims
    assert result.violation is True


def test_audit_matches_dollar_amounts_across_formatting_variants() -> None:
    """"$1.5B" in the report should match "1.5 billion" in the passage."""
    text = "Capex of $1.5B for FY26 [Source: edgar.mdna_highlights]."
    search_fn = _make_search(
        {
            ("mdna", None): [
                _FakeHit(
                    text="We plan capital expenditures of 1.5 billion in FY26",
                    distance=0.4,
                )
            ]
        }
    )
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.ungrounded == []


def test_audit_falls_back_to_unfiltered_search_when_section_empty() -> None:
    """When section-scoped retrieval returns nothing, retry without the
    section filter so a claim still has a chance to be grounded.
    """
    text = (
        "Operating margin improved to 60% this year "
        "[Source: edgar.moat_signals]."
    )
    # moat_signals maps to section=None (any section), so the single
    # passage with matching text grounds the claim.
    search_fn = _make_search(
        {
            (None, None): [
                _FakeHit(
                    text="Operating margin improved to 60% this year.",
                    distance=0.2,
                )
            ]
        }
    )
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.ungrounded == []


def test_audit_ignores_python_tool_sourced_numbers_even_with_edgar_cite() -> None:
    """If the same sentence also carries a Python-tool citation, the
    numbers are attributed to that tool, not to edgar — skip the
    grounding check.
    """
    text = (
        "Revenue is $215.94B [Source: fetch.revenue, edgar.mdna_highlights]."
    )
    # Empty passages — without skipping, this would fail.
    search_fn = _make_search({(None, None): []})
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    assert result.ungrounded == []


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_returns_empty_when_no_violation() -> None:
    text = "No citations here."
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=lambda **k: [])
    md = render_citation_audit_section(result)
    assert md == ""


def test_render_formats_markdown_block_on_violation() -> None:
    text = (
        "Historical growth was 15-20% [Source: edgar.mdna_highlights]."
    )
    search_fn = _make_search({("mdna", None): [_FakeHit(text="unrelated text", distance=1.1)]})
    result = audit_edgar_citations(text, symbol="NVDA", search_fn=search_fn)
    md = render_citation_audit_section(result)
    assert "System Audit" in md
    assert "NVDA" in md
    assert "15-20%" in md
    assert "Citation Grounding" in md


# ---------------------------------------------------------------------------
# Section label mapping sanity
# ---------------------------------------------------------------------------


_REPORT_WITH_SKEPTIC_NO_EDGAR_RISK = """\
# Part 4 · Skeptic

## Attack on the Bull Thesis
1. Target claim: X — attacks vulnerable link #2 [per value chain brief].
2. Target claim: Y — attacks vulnerable link #4 [per value chain brief].
"""


_REPORT_WITH_SKEPTIC_CITES_EDGAR_RISK = """\
# Part 4 · Skeptic

## Attack on the Bull Thesis
1. Target claim: X — edgar risk-factors passage shows dependence on a
   single foundry [Source: 10-K risk_factors, filed 2026-02-25].
"""


def test_audit_flags_skeptic_without_edgar_risk_factors_citation() -> None:
    r = audit_edgar_citations(
        _REPORT_WITH_SKEPTIC_NO_EDGAR_RISK, symbol="NVDA", search_fn=lambda **k: []
    )
    assert r.skeptic_missing_edgar_risk is True
    assert r.violation is True


def test_audit_does_not_flag_skeptic_with_edgar_risk_citation() -> None:
    r = audit_edgar_citations(
        _REPORT_WITH_SKEPTIC_CITES_EDGAR_RISK,
        symbol="NVDA",
        search_fn=lambda **k: [],
    )
    assert r.skeptic_missing_edgar_risk is False


def test_render_includes_skeptic_mandate_section_on_violation() -> None:
    r = audit_edgar_citations(
        _REPORT_WITH_SKEPTIC_NO_EDGAR_RISK, symbol="NVDA", search_fn=lambda **k: []
    )
    md = render_citation_audit_section(r)
    assert "Skeptic mandate violation" in md
    assert "Risk Factors" in md


def test_render_empty_when_skeptic_and_edgar_both_ok() -> None:
    text = """\
# Part 4 · Skeptic

[Source: 10-K risk_factors, filed 2026-02-25]
"""
    r = audit_edgar_citations(text, symbol="NVDA", search_fn=lambda **k: [])
    assert r.violation is False
    assert render_citation_audit_section(r) == ""


def test_label_business_segments_maps_to_business_section() -> None:
    """The integration tests are a better place for full end-to-end
    coverage; here we just verify that the mapping dict is wired up.
    """
    from wise_investor.quality.citation_audit import _EDGAR_LABEL_TO_SECTION

    assert _EDGAR_LABEL_TO_SECTION["business_segments"] == "business"
    assert _EDGAR_LABEL_TO_SECTION["mdna_highlights"] == "mdna"
    assert _EDGAR_LABEL_TO_SECTION["risk_factors"] == "risk_factors"


@pytest.mark.network
def test_live_audit_against_real_run2_report_flags_valuer_hallucination() -> None:
    """Opt-in integration test: reads the actual 20260423_1401 report and
    runs the audit with real Chroma. Only runs with `pytest -m network`.
    """
    import pathlib

    report_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "reports"
        / "NVDA_20260423_1401.crew.md"
    )
    if not report_path.exists():
        pytest.skip("historical report not present")
    result = audit_edgar_citations(report_path.read_text(), symbol="NVDA")
    claims = {u.claim_number for u in result.ungrounded}
    assert "15-20%" in claims, (
        "audit failed to flag the Valuer's 15-20% hallucination"
    )
