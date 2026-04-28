"""Constitution v2.0 audit tests (§21 RULE 4)."""

from __future__ import annotations

from wise_investor.agents.v2.audit import (
    audit_v2_attacks,
    parse_attacks,
    parse_defenses,
)


# ---------------------------------------------------------------------------
# Parsers — robust against the prompt formats we ship
# ---------------------------------------------------------------------------


def test_parse_attacks_extracts_axis_and_number() -> None:
    skeptic_text = """\
## Attack on the Bull Thesis

1. **[axis: bottleneck] Attack type: Substitution emerging**
   - **Target claim (Analyst)**: "TSMC is the sole supplier" [Source: 10-K business_segments].
   - **Counter-evidence**: Apple Silicon migration.

2. **[axis: moat] Attack type: Erosion**
   - **Target claim (Valuer)**: "ROIC sustained" [Source: calculate_roic].
"""
    parsed = parse_attacks(skeptic_text)
    assert len(parsed) == 2
    assert parsed[0].number == 1
    assert parsed[0].axis == "bottleneck"
    assert parsed[1].number == 2
    assert parsed[1].axis == "moat"


def test_parse_attacks_handles_overall_thesis_axis() -> None:
    skeptic_text = "1. **[axis: overall_thesis] Attack type: Cross-axis**\n"
    parsed = parse_attacks(skeptic_text)
    assert len(parsed) == 1
    assert parsed[0].axis == "overall_thesis"


def test_parse_attacks_returns_empty_for_unstructured_input() -> None:
    assert parse_attacks("just some prose, no numbered attacks") == []


def test_parse_defenses_captures_label_and_citations() -> None:
    defender_text = """\
## Defender Response

1. **[axis: bottleneck] DEFENDED**
   - Apple Silicon is internal-use only [Source: 10-K business_segments, filed 2026-02-25].
   - The single-supplier framing is preserved.

2. **[axis: moat] CONCEDED**
   - PyTorch backend abstraction reducing CUDA-specific switching costs.

**Tally:** 1 DEFENDED, 1 CONCEDED
"""
    parsed = parse_defenses(defender_text)
    assert len(parsed) == 2
    assert parsed[0].number == 1
    assert parsed[0].label == "DEFENDED"
    assert parsed[0].axis == "bottleneck"
    assert any("[Source: 10-K" in c for c in parsed[0].citations)
    assert parsed[1].label == "CONCEDED"
    assert parsed[1].axis == "moat"


# ---------------------------------------------------------------------------
# Full audit — outcome scoring per RULE 4
# ---------------------------------------------------------------------------


_SKEPTIC_5 = """\
## Attack on the Bull Thesis

1. **[axis: bottleneck] Attack type: Substitution emerging**
   - text
2. **[axis: bottleneck] Attack type: Geopolitical**
   - text
3. **[axis: bottleneck] Attack type: Tech obsolescence**
   - text
4. **[axis: moat] Attack type: Erosion**
   - text
5. **[axis: moat] Attack type: Mischaracterization**
   - text
"""


def _defender_5(labels: list[tuple[str, str, list[str]]]) -> str:
    """Build a Defender section from (axis, label, citations) per attack."""
    sections = ["## Defender Response\n"]
    for i, (axis, label, citations) in enumerate(labels, start=1):
        block = f"{i}. **[axis: {axis}] {label}**\n"
        block += f"   - Body sentence.\n"
        for c in citations:
            block += f"   - Cited as {c}.\n"
        sections.append(block)
    n_def = sum(1 for (_, label, _) in labels if label == "DEFENDED")
    n_con = sum(1 for (_, label, _) in labels if label == "CONCEDED")
    sections.append(f"\n**Tally:** {n_def} DEFENDED, {n_con} CONCEDED\n")
    return "\n".join(sections)


def test_audit_all_passed_yields_full_score() -> None:
    """Five DEFENDED with verifiable citations → defended_ratio == 1.0."""
    defender = _defender_5([
        ("bottleneck", "DEFENDED", ["[Source: 10-K risk_factors, filed 2026-02-25]"]),
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),
        ("bottleneck", "DEFENDED", ["[Source: edgar.business_segments, filed 2026-02-25]"]),
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),
        ("moat", "DEFENDED", ["[Source: fred.GDPC1]"]),
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert result.defended_ratio == 1.0
    assert result.defended_ratio_pretty == "5.0/5"
    assert all(o.outcome == "PASSED" for o in result.outcomes)
    assert result.axes_with_concession == ()


def test_audit_concession_yields_axis_in_concession_set() -> None:
    """A single CONCEDED on the moat axis lands moat in
    axes_with_concession.
    """
    defender = _defender_5([
        ("bottleneck", "DEFENDED", ["[Source: 10-K risk_factors, filed 2026-02-25]"]),
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),
        ("bottleneck", "DEFENDED", ["[Source: edgar.business_segments, filed 2026-02-25]"]),
        ("moat", "CONCEDED", []),
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert "moat" in result.axes_with_concession
    # The CONCEDED outcome carries score 0.0; total ratio 4/5.
    assert result.defended_ratio_pretty == "4.0/5"


def test_audit_defended_without_citation_fails() -> None:
    """DEFENDED with no [Source: ...] tag at all → audit FAILED."""
    defender = _defender_5([
        ("bottleneck", "DEFENDED", []),  # no citation
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),
        ("bottleneck", "DEFENDED", ["[Source: edgar.business_segments, filed 2026-02-25]"]),
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),
        ("moat", "DEFENDED", ["[Source: fred.GDPC1]"]),
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert result.outcomes[0].outcome == "FAILED"
    assert result.defended_ratio_pretty == "4.0/5"


def test_audit_unknown_citation_prefix_fails() -> None:
    """A made-up citation key the audit doesn't recognize → FAILED."""
    defender = _defender_5([
        ("bottleneck", "DEFENDED", ["[Source: random_invented_key]"]),
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),
        ("bottleneck", "DEFENDED", ["[Source: edgar.business_segments, filed 2026-02-25]"]),
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),
        ("moat", "DEFENDED", ["[Source: fred.GDPC1]"]),
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert result.outcomes[0].outcome == "FAILED"


def test_audit_forward_looking_defense_downgraded() -> None:
    """A DEFENDED that relies on management 'plans to' or 'expects to'
    is forward-looking; per RULE 4 weakness, audit DOWNGRADES.
    """
    defender = """\
## Defender Response

1. **[axis: bottleneck] DEFENDED**
   - Management plans to diversify suppliers by 2027 [Source: 10-K risk_factors, filed 2026-02-25].
2. **[axis: bottleneck] DEFENDED**
   - Concrete metric: revenue is 10B [Source: fetch.revenue].
3. **[axis: bottleneck] DEFENDED**
   - Concrete: 10-K [Source: edgar.business_segments, filed 2026-02-25].
4. **[axis: moat] DEFENDED**
   - Concrete [Source: calculate_roic].
5. **[axis: moat] DEFENDED**
   - Concrete [Source: fred.GDPC1].

**Tally:** 5 DEFENDED, 0 CONCEDED
"""
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    # First attack downgraded (forward-looking phrasing).
    assert result.outcomes[0].outcome == "DOWNGRADED"
    assert result.outcomes[0].score == 0.5
    # Total: 0.5 + 1 + 1 + 1 + 1 = 4.5
    assert result.defended_ratio_pretty == "4.5/5"


def test_audit_missing_defense_yields_failed_slot() -> None:
    """Skeptic produced 5 attacks but Defender only wrote 3 responses.
    The two missing slots become FAILED — penalize the candidate
    rather than skip silently.
    """
    defender = """\
## Defender Response

1. **[axis: bottleneck] DEFENDED**
   - [Source: fetch.revenue]
2. **[axis: bottleneck] DEFENDED**
   - [Source: edgar.business_segments, filed 2026-02-25]
3. **[axis: bottleneck] DEFENDED**
   - [Source: calculate_roic]

**Tally:** 3 DEFENDED, 0 CONCEDED
"""
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert result.outcomes[3].outcome == "FAILED"
    assert result.outcomes[4].outcome == "FAILED"
    # The first 3 PASS but slots 4 and 5 are missing → 3.0/5.
    assert result.defended_ratio_pretty == "3.0/5"


def test_audit_axis_mismatch_uses_skeptic_authoritative() -> None:
    """If Defender flips the axis tag (mistake or hallucination), the
    audit keeps the Skeptic's axis as authoritative for routing."""
    skeptic = """\
1. **[axis: bottleneck] Attack type: Substitution**
"""
    defender = """\
1. **[axis: moat] DEFENDED**
   - [Source: fetch.revenue]

**Tally:** 1 DEFENDED, 0 CONCEDED
"""
    result = audit_v2_attacks(skeptic, defender, n_expected_attacks=1)
    # Outcome is recorded under the Skeptic's axis (bottleneck).
    assert result.outcomes[0].axis == "bottleneck"
    # Reason flags the mismatch for audit trail.
    assert "axis tag mismatch" in result.outcomes[0].reason


def test_audit_summary_text_has_per_attack_lines() -> None:
    defender = _defender_5([
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),
        ("bottleneck", "DEFENDED", ["[Source: fetch.cogs]"]),
        ("bottleneck", "CONCEDED", []),
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),
        ("moat", "DEFENDED", ["[Source: fred.GDPC1]"]),
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    assert "Per-attack outcomes" in result.summary_text
    assert "#1" in result.summary_text
    assert "#5" in result.summary_text
    assert "Defended ratio" in result.summary_text


# ---------------------------------------------------------------------------
# Steward-facing properties — what the rule logic in §21 will consume
# ---------------------------------------------------------------------------


def test_audit_outcome_score_sums_to_defended_ratio_numerator() -> None:
    defender = _defender_5([
        ("bottleneck", "DEFENDED", ["[Source: fetch.revenue]"]),  # 1.0
        ("bottleneck", "DEFENDED", []),                            # 0.0 (no cite)
        ("bottleneck", "CONCEDED", []),                            # 0.0
        ("moat", "DEFENDED", ["[Source: calculate_roic]"]),        # 1.0
        ("moat", "DEFENDED", ["[Source: random_unknown_key]"]),    # 0.0
    ])
    result = audit_v2_attacks(_SKEPTIC_5, defender, n_expected_attacks=5)
    total = sum(o.score for o in result.outcomes)
    assert total == 2.0
    assert result.defended_ratio == 0.4  # 2.0 / 5.0
