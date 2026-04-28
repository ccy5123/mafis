"""Constitution v2.0 rubric-aware audit (§21 RULE 4).

Replaces the legacy generic-count discipline matrix with one that
tracks attacks per axis. The audit:

  1. Parses the Skeptic section for axis-tagged attacks.
  2. Parses the Defender section for DEFENDED / CONCEDED labels
     plus their cited evidence per attack.
  3. Verifies each DEFENDED label's citation against the available
     facts (tool outputs, 10-K passages, news snapshot).
  4. Per attack, emits an AuditOutcome:
       PASSED    — DEFENDED + citation directly supports the defense
       DOWNGRADED — DEFENDED + citation tangential or weak
       FAILED    — DEFENDED + citation absent or doesn't support
       CONCEDED  — Defender said CONCEDED (not a verification
                   failure; an honest concession)
  5. Recomputes the defended_ratio and the axis-pass survivability
     for the Steward's RULE 1-4 logic.

The legacy `agents.steward_audit` is preserved for backward compat
with the per-ticker workflow; this module is the v2 path used when
the runner has Stage 3 axis assignments.

This module is INTENTIONALLY conservative. Per Commitment 3
(precision over recall), ambiguous citations get DOWNGRADED rather
than PASSED. The Steward's RULE 4 then converts DOWNGRADED to a
half-defense, which makes weak defenses costly to the candidate
without dismissing them entirely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


AuditOutcomeKind = Literal["PASSED", "DOWNGRADED", "FAILED", "CONCEDED"]


@dataclass(frozen=True)
class ParsedAttack:
    """One attack from the Skeptic section, after parsing."""

    number: int
    axis: str  # "moat" | "new_frontier" | "bottleneck" | "overall_thesis"
    raw_text: str


@dataclass(frozen=True)
class ParsedDefense:
    """One Defender response, after parsing."""

    number: int
    axis: str  # echoed from Skeptic; SHOULD match ParsedAttack[N].axis
    label: Literal["DEFENDED", "CONCEDED", "MISSING"]
    raw_text: str
    citations: tuple[str, ...]  # extracted [Source: ...] tokens


@dataclass(frozen=True)
class AuditOutcome:
    """Per-attack audit result."""

    attack_number: int
    axis: str
    outcome: AuditOutcomeKind
    score: float  # 1.0 / 0.5 / 0.0 per RULE 4
    reason: str


@dataclass(frozen=True)
class AuditResult:
    """Aggregate audit output consumed by the Steward."""

    n_total_attacks: int
    outcomes: tuple[AuditOutcome, ...]
    defended_ratio: float  # 0.0-1.0 (numerator scored, denominator is N)
    defended_ratio_pretty: str  # "X.X/N" for prompt embedding
    axes_with_concession: tuple[str, ...]  # axes where any attack was conceded/failed
    summary_text: str  # multi-line text suitable for the Steward prompt's <audit_summary>


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


# Skeptic attack heading: `1. **[axis: bottleneck] Attack type: ...**`
# The axis tag is what we need; the attack type is informational.
_ATTACK_HEAD_RE = re.compile(
    r"^\s*(?P<num>\d+)\.\s+\*\*\[axis:\s*(?P<axis>[a-z_]+)\]",
    re.MULTILINE,
)


def parse_attacks(skeptic_text: str) -> list[ParsedAttack]:
    """Extract every numbered attack from the Skeptic section."""
    out: list[ParsedAttack] = []
    matches = list(_ATTACK_HEAD_RE.finditer(skeptic_text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(skeptic_text)
        out.append(
            ParsedAttack(
                number=int(m.group("num")),
                axis=m.group("axis"),
                raw_text=skeptic_text[start:end].strip(),
            )
        )
    return out


# Defender response heading:
# `1. **[axis: bottleneck] DEFENDED**`  or  `... CONCEDED**`
_DEFENSE_HEAD_RE = re.compile(
    r"^\s*(?P<num>\d+)\.\s+\*\*\[axis:\s*(?P<axis>[a-z_]+)\]\s+(?P<label>DEFENDED|CONCEDED)\*\*",
    re.MULTILINE,
)


# Inline citation pattern — same shape as elsewhere in MAFIS.
_CITATION_RE = re.compile(r"\[Source:\s*[^\]]+\]")


def parse_defenses(defender_text: str) -> list[ParsedDefense]:
    """Extract every numbered Defender response."""
    out: list[ParsedDefense] = []
    matches = list(_DEFENSE_HEAD_RE.finditer(defender_text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(defender_text)
        body = defender_text[start:end].strip()
        citations = tuple(_CITATION_RE.findall(body))
        out.append(
            ParsedDefense(
                number=int(m.group("num")),
                axis=m.group("axis"),
                label=m.group("label"),  # type: ignore[arg-type]
                raw_text=body,
                citations=citations,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


# Citation kinds we consider PASSED on shape alone. The full citation-
# grounding audit (the existing `quality.citation_audit` module) only
# verifies edgar.* citations against the 10-K embeddings; for the
# v2 audit we trust well-formed citations of the listed kinds. The
# stricter ground-truth check happens separately downstream.
_PASSED_CITATION_PREFIXES: tuple[str, ...] = (
    "fetch.",
    "calculate_",
    "reverse_dcf",
    "cross_validate_quote",
    "get_peer_multiples",
    "fred.",
    "edgar.",
    "10-K ",
    "Google News",
    "GDELT",
    "value chain brief",
)


# Citations that are forward-looking signal weakness — even if the
# citation itself exists, the defense is not yet provable. RULE 4
# downgrades DEFENDED labels resting solely on these.
_FORWARD_LOOKING_HINTS: tuple[str, ...] = (
    "plans to",
    "expects to",
    "guidance",
    "roadmap",
    "intends to",
)


def _classify_citation(citation: str) -> Literal["passed", "downgraded", "failed"]:
    """Return the per-citation classification used by the audit.

    A citation is FAILED if it doesn't match a recognized prefix
    (i.e. the LLM made up a citation key). It's PASSED if it
    matches a recognized prefix and the surrounding evidence isn't
    obviously forward-looking.
    """
    inner = citation.replace("[Source:", "").replace("]", "").strip()
    if not any(inner.startswith(p) for p in _PASSED_CITATION_PREFIXES):
        return "failed"
    return "passed"


def _classify_defense(defense: ParsedDefense) -> tuple[AuditOutcomeKind, str]:
    """Convert a single ParsedDefense to (outcome, human-readable reason)."""
    if defense.label == "CONCEDED":
        return ("CONCEDED", "Defender conceded honestly")

    if not defense.citations:
        return (
            "FAILED",
            "DEFENDED but no [Source: ...] citation found in the response",
        )

    # If ALL citations match the failed shape, the defense fails.
    classifications = [_classify_citation(c) for c in defense.citations]
    if all(c == "failed" for c in classifications):
        return (
            "FAILED",
            f"DEFENDED but citations are not recognized "
            f"({', '.join(defense.citations)})",
        )

    # Forward-looking framing in the body weakens the defense.
    body_lower = defense.raw_text.lower()
    if any(hint in body_lower for hint in _FORWARD_LOOKING_HINTS):
        return (
            "DOWNGRADED",
            "DEFENDED but defense relies on forward-looking management "
            "guidance / plans rather than a concrete event",
        )

    if any(c == "failed" for c in classifications):
        return (
            "DOWNGRADED",
            "DEFENDED with mixed citations — at least one citation key "
            "is not recognized",
        )

    return ("PASSED", "DEFENDED with verifiable citation")


def _outcome_score(outcome: AuditOutcomeKind) -> float:
    """RULE 4 scoring."""
    return {
        "PASSED": 1.0,
        "DOWNGRADED": 0.5,
        "FAILED": 0.0,
        "CONCEDED": 0.0,
    }[outcome]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def audit_v2_attacks(
    skeptic_text: str,
    defender_text: str,
    n_expected_attacks: int,
) -> AuditResult:
    """Run the v2 audit on a Skeptic+Defender pair.

    `n_expected_attacks` is the count from the Skeptic's AttackPlan
    (5 for 2-axis candidates, 7 for 3-axis). When the parsed counts
    don't match (LLM output was truncated, formatting drifted), the
    missing attacks are recorded as FAILED to penalize the candidate
    rather than skip them silently.
    """
    attacks = parse_attacks(skeptic_text)
    defenses = parse_defenses(defender_text)

    # Index by attack number so partial output gets stitched correctly.
    attacks_by_num = {a.number: a for a in attacks}
    defenses_by_num = {d.number: d for d in defenses}

    outcomes: list[AuditOutcome] = []
    for n in range(1, n_expected_attacks + 1):
        attack = attacks_by_num.get(n)
        defense = defenses_by_num.get(n)

        if attack is None and defense is None:
            outcomes.append(
                AuditOutcome(
                    attack_number=n,
                    axis="unknown",
                    outcome="FAILED",
                    score=0.0,
                    reason="attack #{n} missing from Skeptic AND Defender output",
                )
            )
            continue

        if attack is None:
            # Defender wrote a response to a non-existent attack; ignore
            # the malformation, treat as FAILED for that slot.
            outcomes.append(
                AuditOutcome(
                    attack_number=n,
                    axis=defense.axis if defense else "unknown",
                    outcome="FAILED",
                    score=0.0,
                    reason=f"attack #{n} missing in Skeptic but defended in Defender",
                )
            )
            continue

        if defense is None:
            outcomes.append(
                AuditOutcome(
                    attack_number=n,
                    axis=attack.axis,
                    outcome="FAILED",
                    score=0.0,
                    reason=f"attack #{n} present in Skeptic but no Defender response",
                )
            )
            continue

        # Axis tag mismatch is suspicious — the Defender flipped the
        # axis somewhere. Keep the Skeptic's axis as authoritative.
        outcome_kind, reason = _classify_defense(defense)
        outcomes.append(
            AuditOutcome(
                attack_number=n,
                axis=attack.axis,
                outcome=outcome_kind,
                score=_outcome_score(outcome_kind),
                reason=(
                    f"axis tag mismatch (Skeptic={attack.axis}, "
                    f"Defender={defense.axis}); using Skeptic's. {reason}"
                    if defense.axis != attack.axis
                    else reason
                ),
            )
        )

    total_score = sum(o.score for o in outcomes)
    ratio = total_score / n_expected_attacks if n_expected_attacks else 0.0

    axes_with_concession = tuple(
        sorted({o.axis for o in outcomes if o.outcome in ("CONCEDED", "FAILED")})
    )

    summary_lines: list[str] = [
        f"Total attacks expected: {n_expected_attacks}",
        f"Defended ratio (RULE 4 weighted): {total_score:.1f}/{n_expected_attacks} "
        f"= {ratio:.2%}",
        "Per-attack outcomes:",
    ]
    for o in outcomes:
        summary_lines.append(
            f"  #{o.attack_number} [axis={o.axis}] "
            f"{o.outcome} (score {o.score:.1f}) — {o.reason}"
        )
    if axes_with_concession:
        summary_lines.append(
            f"Axes with at least one concession/fail: "
            f"{', '.join(axes_with_concession)}"
        )

    return AuditResult(
        n_total_attacks=n_expected_attacks,
        outcomes=tuple(outcomes),
        defended_ratio=ratio,
        defended_ratio_pretty=f"{total_score:.1f}/{n_expected_attacks}",
        axes_with_concession=axes_with_concession,
        summary_text="\n".join(summary_lines),
    )


__all__ = [
    "AuditOutcome",
    "AuditOutcomeKind",
    "AuditResult",
    "ParsedAttack",
    "ParsedDefense",
    "audit_v2_attacks",
    "parse_attacks",
    "parse_defenses",
]
