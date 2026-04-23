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
