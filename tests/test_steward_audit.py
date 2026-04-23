"""Tests for the Steward discipline audit.

The audit parses a Steward markdown section, counts the
SURVIVED/NEUTRALIZED labels inside Rationale, and checks whether the
reported Verdict + Conviction honor the discipline matrix:

  Both NEUTRALIZED (survived=0)    → BUY allowed (C3-C5)
  One NEUTRALIZED, one SURVIVED    → HOLD/PASS (C1-C2)
  Both SURVIVED (neutralized=0)    → PASS (C1)
"""

from __future__ import annotations

from wise_investor.agents.steward_audit import (
    apply_audit_to_section,
    audit_steward_section,
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


_COMPLIANT_BUY = """\
## Verdict
**BUY**

## Conviction Level
**Conviction: 4**

## Rationale
The Bull thesis is strong cash flow.

- **NEUTRALIZED**: Skeptic claim 1 refuted by concrete FCF number of $96B.
- **NEUTRALIZED**: Skeptic claim 2 refuted by named customer contract.

## Position Sizing Guidance
3-5% equity allocation.
"""


def test_audit_compliant_buy_is_not_violation() -> None:
    r = audit_steward_section(_COMPLIANT_BUY)
    assert r.verdict == "BUY"
    assert r.conviction == 4
    assert r.neutralized_count == 2
    assert r.survived_count == 0
    assert r.violation is False


_COMPLIANT_HOLD = """\
## Verdict
HOLD

## Conviction Level
Conviction: 2

## Rationale
Bull thesis exists but Skeptic's strongest rebuttal is unresolved.

- **NEUTRALIZED**: Claim 1 refuted by concrete data.
- **SURVIVED**: Claim 2 has no concrete Bull counter-evidence in report.
"""


def test_audit_compliant_hold_passes() -> None:
    r = audit_steward_section(_COMPLIANT_HOLD)
    assert r.verdict == "HOLD"
    assert r.conviction == 2
    assert r.neutralized_count == 1
    assert r.survived_count == 1
    assert r.violation is False


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


_ACTUAL_BUG_ONE_SURVIVED_BUT_BUY_C4 = """\
## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
The Bull thesis is strong cash flow. The Skeptic's strongest rebuttal is 20% growth sustainability.

- **NEUTRALIZED**: The reverse DCF implies 20.44% growth over 10 years, reflecting strong demand.
- **SURVIVED**: Skeptic's claim that 20%+ FCF growth for 10 years is not historically sustainable remains unchallenged due to absence of historical benchmark data.
"""


def test_audit_catches_one_survived_but_buy_c4_violation() -> None:
    """The exact failure observed in NVDA_20260423_1311.crew.md."""
    r = audit_steward_section(_ACTUAL_BUG_ONE_SURVIVED_BUT_BUY_C4)
    assert r.verdict == "BUY"
    assert r.conviction == 4
    assert r.neutralized_count == 1
    assert r.survived_count == 1
    assert r.violation is True
    assert r.corrected_verdict == "HOLD"
    assert r.corrected_conviction == 2


_BOTH_SURVIVED_BUT_HOLD = """\
## Verdict
HOLD

## Conviction Level
Conviction: 2

## Rationale
- **SURVIVED**: Claim 1 has no Bull counter-evidence.
- **SURVIVED**: Claim 2 has no Bull counter-evidence.
"""


def test_audit_both_survived_forces_pass() -> None:
    r = audit_steward_section(_BOTH_SURVIVED_BUT_HOLD)
    assert r.neutralized_count == 0
    assert r.survived_count == 2
    assert r.violation is True
    assert r.corrected_verdict == "PASS"
    assert r.corrected_conviction == 1


_BUY_C5_WITH_SURVIVED = """\
## Verdict
BUY

## Conviction Level
Conviction: 5

## Rationale
- **NEUTRALIZED**: X.
- **SURVIVED**: Y.
"""


def test_audit_buy_c5_with_survived_is_violation() -> None:
    r = audit_steward_section(_BUY_C5_WITH_SURVIVED)
    assert r.violation is True
    assert r.corrected_verdict == "HOLD"


# ---------------------------------------------------------------------------
# Conviction-only violation (verdict allowed, conviction too high)
# ---------------------------------------------------------------------------


_HOLD_C5_ALL_NEUTRALIZED = """\
## Verdict
HOLD

## Conviction Level
Conviction: 5

## Rationale
- **NEUTRALIZED**: X.
- **NEUTRALIZED**: Y.
"""


def test_audit_flags_hold_with_impossible_conviction() -> None:
    # All-neutralized → BUY allowed up to C5. HOLD+C5 is still valid here,
    # because BUY is a ceiling, not a requirement. HOLD stays OK.
    r = audit_steward_section(_HOLD_C5_ALL_NEUTRALIZED)
    # HOLD with no SURVIVED is actually fine — analyst may decline BUY for
    # other reasons. Audit only flags TOO-OPTIMISTIC verdicts, not
    # TOO-CONSERVATIVE ones. So this should NOT be a violation.
    assert r.violation is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_audit_handles_missing_verdict_heading() -> None:
    r = audit_steward_section("Some text without headings.")
    assert r.verdict is None
    assert r.violation is False


def test_audit_labels_in_caveats_section_are_ignored() -> None:
    """SURVIVED mentioned in Confidence Caveats should not count as a
    Rationale label (they're just restatements).
    """
    text = """\
## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
- **NEUTRALIZED**: Both main rebuttals refuted by concrete numbers.
- **NEUTRALIZED**: Second one also.

## Confidence Caveats
- If the SURVIVED rebuttal about FCF growth were quantified ...
"""
    r = audit_steward_section(text)
    # The SURVIVED in Caveats should NOT count toward Rationale labels.
    assert r.survived_count == 0
    assert r.neutralized_count == 2
    assert r.violation is False


def test_audit_parses_asterisk_markdown_verdict() -> None:
    text = """\
## Verdict
**BUY**

## Conviction Level
**Conviction: 3**

## Rationale
- **NEUTRALIZED**: X.
- **NEUTRALIZED**: Y.
"""
    r = audit_steward_section(text)
    assert r.verdict == "BUY"
    assert r.conviction == 3
    assert r.violation is False


# ---------------------------------------------------------------------------
# apply_audit_to_section (rendering)
# ---------------------------------------------------------------------------


def test_apply_audit_returns_unchanged_when_compliant() -> None:
    r = audit_steward_section(_COMPLIANT_BUY)
    out = apply_audit_to_section(_COMPLIANT_BUY, r)
    assert out == _COMPLIANT_BUY


def test_apply_audit_appends_note_on_violation() -> None:
    r = audit_steward_section(_ACTUAL_BUG_ONE_SURVIVED_BUT_BUY_C4)
    out = apply_audit_to_section(_ACTUAL_BUG_ONE_SURVIVED_BUT_BUY_C4, r)
    assert "System Audit" in out
    assert "HOLD" in out
    assert "NEUTRALIZED=1" in out
    assert "SURVIVED=1" in out
    # Original narrative preserved verbatim.
    assert _ACTUAL_BUG_ONE_SURVIVED_BUT_BUY_C4.strip() in out


# ---------------------------------------------------------------------------
# Speculative-only NEUTRALIZATIONs (reclassified as effective SURVIVED)
# ---------------------------------------------------------------------------


_RUN2_FAKE_NEUTRALIZATIONS = """\
## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
The Bull thesis is strong cash flow. The Skeptic's strongest rebuttal is historical growth unsustainability.

The Skeptic's first rebuttal, that no semiconductor company has historically sustained such high growth rates for an extended period, is NEUTRALIZED by the fact that NVIDIA's current market position and technological advancements support a higher growth rate than historical averages. The reverse DCF tool inputs include a discount rate of 10.00%, terminal growth of 2.50%, and high-growth years of 10, which are reasonable assumptions given the company's competitive edge.

The second top Skeptic rebuttal is that TSMC's single point of supply risk could cost NVDA a full quarter of revenue in the event of a CoWoS outage. This is NEUTRALIZED by the fact that TSMC has diversified its foundry services and is working on alternative suppliers, reducing the likelihood of such an outage.
"""


def test_audit_catches_run2_fake_neutralizations() -> None:
    """The exact failure from NVDA_20260423_1401.crew.md — two labels
    marked NEUTRALIZED but the justifications are pure speculative
    language with no [Source: ...] citations.
    """
    r = audit_steward_section(_RUN2_FAKE_NEUTRALIZATIONS)
    assert r.verdict == "BUY"
    assert r.neutralized_count == 2
    assert r.survived_count == 0
    # Both paragraphs reclassified as speculative-only.
    assert r.invalid_neutralized_count == 2
    assert r.effective_neutralized == 0
    assert r.effective_survived == 2
    # Matrix says PASS C1 when all are effectively SURVIVED.
    assert r.violation is True
    assert r.corrected_verdict == "PASS"
    assert r.corrected_conviction == 1


def test_audit_accepts_neutralization_with_concrete_citation() -> None:
    """A NEUTRALIZED paragraph with a `[Source: ...]` citation passes
    even if it contains speculative words — the citation grounds the
    claim and the reader can judge whether the source refutes the
    scenario.
    """
    text = """\
## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
- **NEUTRALIZED**: Skeptic claim that margins could compress is refuted by NVDA's operating income of $130.39B [Source: fetch.operating_income].
- **NEUTRALIZED**: Capital discipline shown by total debt of $7.47B [Source: fetch.total_debt].
"""
    r = audit_steward_section(text)
    assert r.invalid_neutralized_count == 0
    assert r.effective_neutralized == 2
    assert r.effective_survived == 0
    assert r.violation is False


def test_audit_pure_speculation_without_citation_is_invalid() -> None:
    """Isolated speculative neutralization, no citation — invalid."""
    text = """\
## Verdict
BUY

## Conviction Level
Conviction: 3

## Rationale
- **NEUTRALIZED**: The moat should remain strong as NVIDIA is well-positioned.
- **NEUTRALIZED**: Growth could continue given the competitive edge.
"""
    r = audit_steward_section(text)
    assert r.invalid_neutralized_count == 2
    assert r.effective_neutralized == 0
    assert r.effective_survived == 2
    assert r.violation is True
    assert r.corrected_verdict == "PASS"


def test_audit_mixed_one_valid_one_speculative() -> None:
    text = """\
## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
- **NEUTRALIZED**: FCF of $96B [Source: fetch.free_cash_flow] refutes the cash-burn concern.
- **NEUTRALIZED**: TSMC risk should be manageable as they are working on diversification.
"""
    r = audit_steward_section(text)
    # One valid, one speculative → 1 / 1
    assert r.invalid_neutralized_count == 1
    assert r.effective_neutralized == 1
    assert r.effective_survived == 1
    # Matrix: one N + one S → HOLD ceiling
    assert r.violation is True
    assert r.corrected_verdict == "HOLD"


def test_audit_notes_list_each_reclassified_paragraph() -> None:
    r = audit_steward_section(_RUN2_FAKE_NEUTRALIZATIONS)
    reclassified_notes = [
        n for n in r.notes if n.startswith("  - reclassified:")
    ]
    assert len(reclassified_notes) == 2
    # Each reclassified entry names the speculative markers matched.
    assert any("could" in n for n in reclassified_notes)
    assert any("support a higher" in n.lower() for n in reclassified_notes)


def test_audit_apply_renders_speculative_downgrade() -> None:
    r = audit_steward_section(_RUN2_FAKE_NEUTRALIZATIONS)
    out = apply_audit_to_section(_RUN2_FAKE_NEUTRALIZATIONS, r)
    assert "Speculative-only NEUTRALIZATIONs" in out
    assert "Effective labels used for matrix" in out
    assert "PASS" in out
