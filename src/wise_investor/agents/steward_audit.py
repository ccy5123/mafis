"""Steward discipline audit — deterministic post-check on the Steward's output.

Phase 2-B defined the verdict-from-labels decision matrix explicitly:

    Both top-two rebuttals NEUTRALIZED  →  BUY allowed (C3-C5)
    One NEUTRALIZED, one SURVIVED       →  HOLD (C1-C2) or PASS (C1)
    Both SURVIVED                       →  PASS (C1) by default

Empirically, Qwen 2.5 7B follows the letter of the prompt (emits the
SURVIVED/NEUTRALIZED labels) but does NOT reliably apply the decision
matrix when choosing Verdict + Conviction. We observed a BUY C4 issued
with one SURVIVED label visible in the same section — a direct
violation of the rules the system prompt teaches.

This module parses the Steward's markdown, counts the labels, checks the
Verdict/Conviction against the matrix, and — on violation — appends a
System Audit note to the end of the Steward section that downgrades the
reported verdict. The original Steward text is NOT modified in place;
the audit is additive so the human reader can see both what the LLM
wrote and what the discipline-corrected decision is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Verdicts + mapping to max-allowed conviction given label counts.
_VERDICT_RE = re.compile(r"\bVerdict\b", re.IGNORECASE)
_VERDICT_VALUE_RE = re.compile(
    r"(?im)^\s*(?:\*\*)?(BUY|HOLD|PASS)(?:\*\*)?\s*$"
)
_CONVICTION_RE = re.compile(
    r"(?i)conviction\s*[:\s]\s*\**\s*(\d)\s*\**"
)
_NEUTRALIZED_RE = re.compile(r"\b(?:\*\*)?NEUTRALIZED(?:\*\*)?\b")
_SURVIVED_RE = re.compile(r"\b(?:\*\*)?SURVIVED(?:\*\*)?\b")


# Phrases that signal speculative Bull reasoning. The Phase 2-B prompt
# explicitly bans these as grounds for "NEUTRALIZATION". When a
# paragraph labelled NEUTRALIZED contains any of these AND carries no
# `[Source: ...]` citation to ground the claim, the audit reclassifies
# it as an invalid neutralization (effectively SURVIVED) before
# applying the verdict matrix.
_SPECULATIVE_MARKERS: list[re.Pattern[str]] = [
    re.compile(r"\bcould\b", re.IGNORECASE),
    re.compile(r"\bmay\b", re.IGNORECASE),
    re.compile(r"\bmight\b", re.IGNORECASE),
    re.compile(r"\bshould\b", re.IGNORECASE),
    re.compile(r"\bwould\b", re.IGNORECASE),
    re.compile(r"\blikely to\b", re.IGNORECASE),
    re.compile(r"\bexpected to\b", re.IGNORECASE),
    re.compile(r"\bis (?:working|developing) on\b", re.IGNORECASE),
    re.compile(r"\bare (?:working|developing) on\b", re.IGNORECASE),
    re.compile(r"\bis well[- ]positioned\b", re.IGNORECASE),
    re.compile(r"\bsupports?\s+(?:a|the)?\s*higher\b", re.IGNORECASE),
    re.compile(r"\breducing the likelihood\b", re.IGNORECASE),
    re.compile(r"\breasonable assumptions?\b", re.IGNORECASE),
    re.compile(r"\bcompetitive edge\b", re.IGNORECASE),
    re.compile(r"\bshould (?:be able to|remain|continue)\b", re.IGNORECASE),
]

# A `[Source: ...]` citation is our proxy for "concrete evidence". When
# present alongside speculative language the audit treats the
# neutralization as at-least-borderline valid (the reader can judge
# whether the cited source actually refutes the Skeptic's scenario).
_CITATION_RE = re.compile(r"\[Source:", re.IGNORECASE)


@dataclass
class StewardAuditResult:
    """Outcome of auditing a Steward section."""

    verdict: str | None       # parsed BUY / HOLD / PASS (None if unparseable)
    conviction: int | None    # parsed 1-5
    neutralized_count: int    # raw NEUTRALIZED label count
    survived_count: int       # raw SURVIVED label count
    invalid_neutralized_count: int = 0   # NEUTRALIZED paragraphs that are
                                         # pure speculation (no `[Source: ]`)
    effective_neutralized: int = 0  # raw - invalid
    effective_survived: int = 0     # raw + invalid
    violation: bool = False   # True if verdict breaks the discipline matrix
    corrected_verdict: str | None = None   # verdict the matrix requires
    corrected_conviction: int | None = None
    notes: list[str] = None   # type: ignore[assignment]
    reclassified_paragraphs: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []
        if self.reclassified_paragraphs is None:
            self.reclassified_paragraphs = []


def _parse_verdict(text: str) -> str | None:
    """Find the Verdict value under the `## Verdict` heading.

    Uses a heading-scoped search so BUY/HOLD/PASS mentions elsewhere in
    the Rationale don't interfere.
    """
    # Slice from "## Verdict" to the next H2 or end.
    heading = re.search(r"(?im)^\s*##\s*Verdict\s*$", text)
    if heading is None:
        return None
    start = heading.end()
    next_h2 = re.search(r"(?im)^\s*##\s", text[start:])
    end = start + (next_h2.start() if next_h2 else len(text) - start)
    section = text[start:end]
    m = _VERDICT_VALUE_RE.search(section)
    return m.group(1).upper() if m else None


def _parse_conviction(text: str) -> int | None:
    m = _CONVICTION_RE.search(text)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    if not 1 <= n <= 5:
        return None
    return n


def _rationale_scope(text: str) -> str:
    """Return the text inside the `## Rationale` heading (or the whole
    text if no explicit heading exists).
    """
    heading = re.search(r"(?im)^\s*##\s*Rationale\s*$", text)
    if heading is None:
        return text
    start = heading.end()
    next_h2 = re.search(r"(?im)^\s*##\s", text[start:])
    end = start + (next_h2.start() if next_h2 else len(text) - start)
    return text[start:end]


def _label_counts(text: str) -> tuple[int, int]:
    """Count SURVIVED / NEUTRALIZED labels inside the Rationale section.

    We do NOT count mentions outside Rationale (e.g. in Confidence
    Caveats) because those restate labels without originating them.
    """
    scope = _rationale_scope(text)
    return (
        len(_NEUTRALIZED_RE.findall(scope)),
        len(_SURVIVED_RE.findall(scope)),
    )


_BULLET_RE = re.compile(r"^\s*[-*]\s", re.MULTILINE)


def _split_paragraphs(scope: str) -> list[str]:
    """Split the Rationale scope into logical paragraphs.

    Blank lines are the primary separator. Consecutive bullet items
    (lines starting with `-` or `*`) are further split into one
    paragraph per bullet — each bullet is an independent piece of
    evidence in the Steward template, and we must validate them
    separately.
    """
    raw = [p.strip() for p in re.split(r"\n\s*\n", scope) if p.strip()]
    expanded: list[str] = []
    for p in raw:
        lines = p.splitlines()
        starts = [
            i for i, line in enumerate(lines) if re.match(r"^\s*[-*]\s", line)
        ]
        if len(starts) < 2:
            expanded.append(p)
            continue
        # Group each bullet's continuation lines with it.
        for idx, start in enumerate(starts):
            end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
            bullet = "\n".join(lines[start:end]).strip()
            if bullet:
                expanded.append(bullet)
    return expanded


def _extract_neutralized_paragraphs(text: str) -> list[str]:
    """Return every logical paragraph in Rationale containing a NEUTRALIZED
    label. Each bullet item is its own paragraph — see _split_paragraphs.
    """
    scope = _rationale_scope(text)
    return [p for p in _split_paragraphs(scope) if _NEUTRALIZED_RE.search(p)]


def _is_valid_neutralization(paragraph: str) -> tuple[bool, list[str]]:
    """Decide whether a paragraph labelled NEUTRALIZED actually presents
    concrete evidence rather than speculative Bull reasoning.

    Rule: if ANY speculative marker appears in the paragraph AND there
    is no `[Source: ...]` citation in the same paragraph, the
    neutralization is invalid — the Bull case is restated rather than
    refuted. Speculative language with a citation present passes
    (reader can judge whether the citation refutes the scenario).

    Returns (is_valid, matched_speculative_markers).
    """
    matched: list[str] = []
    for pat in _SPECULATIVE_MARKERS:
        m = pat.search(paragraph)
        if m:
            matched.append(m.group(0))
    if not matched:
        return (True, [])

    has_citation = bool(_CITATION_RE.search(paragraph))
    if has_citation:
        return (True, matched)

    return (False, matched)


def _assess_neutralizations(text: str) -> tuple[int, list[tuple[str, list[str]]]]:
    """Return (invalid_count, list of (excerpt, matched_markers)) for each
    NEUTRALIZED paragraph that fails the validity check.
    """
    invalid_details: list[tuple[str, list[str]]] = []
    for para in _extract_neutralized_paragraphs(text):
        valid, markers = _is_valid_neutralization(para)
        if not valid:
            excerpt = para.strip().splitlines()[0][:160]
            invalid_details.append((excerpt, markers))
    return (len(invalid_details), invalid_details)


def _required_verdict_ceiling(
    neutralized: int, survived: int
) -> tuple[str, int]:
    """Apply the discipline matrix to (N, S) label counts.

    Returns (max_verdict, max_conviction). BUY is only available when
    there are NO SURVIVED labels among the top-two rebuttals. We treat
    `survived == 0` as "both neutralized" regardless of whether two
    NEUTRALIZED labels were explicitly emitted, so that templates that
    only label the weak links still resolve correctly.
    """
    if survived == 0 and neutralized >= 1:
        return ("BUY", 5)
    if survived >= 1 and neutralized >= 1:
        return ("HOLD", 2)
    if survived >= 1 and neutralized == 0:
        return ("PASS", 1)
    # No labels emitted at all — degenerate case, allow the Steward's
    # own verdict to stand; audit only flags this as a warning later.
    return ("BUY", 5)


def _verdict_rank(v: str) -> int:
    return {"BUY": 2, "HOLD": 1, "PASS": 0}.get(v.upper(), 2)


def audit_steward_section(text: str) -> StewardAuditResult:
    """Parse the Steward markdown and determine whether the verdict and
    conviction comply with the discipline matrix.

    Two-stage check:
      1. Count raw NEUTRALIZED / SURVIVED labels in the Rationale.
      2. Inspect each NEUTRALIZED paragraph for speculative-only
         language without a `[Source: ...]` citation; reclassify such
         paragraphs as "effective SURVIVED" before applying the matrix.

    Non-destructive: callers are expected to APPEND an audit note when
    `result.violation` is True, not rewrite the Steward's words.
    """
    verdict = _parse_verdict(text)
    conviction = _parse_conviction(text)
    neutralized, survived = _label_counts(text)
    invalid_count, invalid_details = _assess_neutralizations(text)

    effective_neutralized = max(0, neutralized - invalid_count)
    effective_survived = survived + invalid_count

    notes: list[str] = []
    notes.append(
        f"Labels parsed: NEUTRALIZED={neutralized}, SURVIVED={survived}."
    )
    if invalid_count > 0:
        notes.append(
            f"Speculative-only NEUTRALIZATIONs: {invalid_count} "
            f"(reclassified as SURVIVED for matrix)."
        )
        for excerpt, markers in invalid_details:
            notes.append(
                f"  - reclassified: {excerpt!r} "
                f"(speculative markers: {', '.join(sorted(set(markers)))})"
            )
        notes.append(
            f"Effective counts: NEUTRALIZED={effective_neutralized}, "
            f"SURVIVED={effective_survived}."
        )

    max_verdict, max_conviction = _required_verdict_ceiling(
        effective_neutralized, effective_survived
    )

    violation = False
    corrected_verdict: str | None = None
    corrected_conviction: int | None = None

    if verdict is None:
        notes.append(
            "No parseable Verdict heading found — audit cannot enforce matrix."
        )
        return StewardAuditResult(
            verdict=None,
            conviction=conviction,
            neutralized_count=neutralized,
            survived_count=survived,
            invalid_neutralized_count=invalid_count,
            effective_neutralized=effective_neutralized,
            effective_survived=effective_survived,
            violation=False,
            corrected_verdict=None,
            corrected_conviction=None,
            notes=notes,
            reclassified_paragraphs=[e for e, _ in invalid_details],
        )

    # Verdict-level check: is the parsed verdict more optimistic than the
    # matrix ceiling allows?
    if _verdict_rank(verdict) > _verdict_rank(max_verdict):
        violation = True
        corrected_verdict = max_verdict
        corrected_conviction = max_conviction
        notes.append(
            f"VIOLATION: reported Verdict={verdict} exceeds matrix ceiling "
            f"{max_verdict} given effective label counts."
        )

    # Conviction-level check: even if verdict is allowed, is conviction too high?
    elif conviction is not None and conviction > max_conviction and verdict == max_verdict:
        violation = True
        corrected_verdict = verdict
        corrected_conviction = max_conviction
        notes.append(
            f"VIOLATION: Conviction={conviction} exceeds ceiling {max_conviction} "
            f"for Verdict={verdict} given effective label counts."
        )

    else:
        notes.append(
            f"OK: Verdict={verdict}, Conviction={conviction} consistent with matrix."
        )

    return StewardAuditResult(
        verdict=verdict,
        conviction=conviction,
        neutralized_count=neutralized,
        survived_count=survived,
        invalid_neutralized_count=invalid_count,
        effective_neutralized=effective_neutralized,
        effective_survived=effective_survived,
        violation=violation,
        corrected_verdict=corrected_verdict,
        corrected_conviction=corrected_conviction,
        notes=notes,
        reclassified_paragraphs=[e for e, _ in invalid_details],
    )


def apply_audit_to_section(steward_text: str, result: StewardAuditResult) -> str:
    """Append a System Audit note to the Steward section when a violation
    was detected. Leaves a clean section unchanged when no violation.
    """
    if not result.violation:
        return steward_text

    audit_lines = [
        "",
        "",
        "---",
        "",
        "### System Audit — Discipline Matrix Enforcement",
        "",
        f"- Raw labels: NEUTRALIZED={result.neutralized_count}, "
        f"SURVIVED={result.survived_count}.",
    ]
    if result.invalid_neutralized_count > 0:
        audit_lines.append(
            f"- Speculative-only NEUTRALIZATIONs (reclassified as SURVIVED): "
            f"{result.invalid_neutralized_count}."
        )
        audit_lines.append(
            f"- Effective labels used for matrix: "
            f"NEUTRALIZED={result.effective_neutralized}, "
            f"SURVIVED={result.effective_survived}."
        )
    audit_lines.extend(
        [
            f"- Reported Verdict: **{result.verdict}** / Conviction: "
            f"{result.conviction if result.conviction is not None else '?'}.",
            f"- Matrix ceiling for effective labels: "
            f"**{result.corrected_verdict}** / Conviction "
            f"{result.corrected_conviction}.",
            "- This is an automatic Python post-check; the Steward's narrative "
            "above is left verbatim for audit transparency. The verdict used "
            "downstream is the matrix-corrected value, not the one the "
            "language model wrote.",
        ]
    )
    for n in result.notes:
        audit_lines.append(f"- {n}")

    return steward_text.rstrip() + "\n" + "\n".join(audit_lines) + "\n"


__all__ = [
    "StewardAuditResult",
    "apply_audit_to_section",
    "audit_steward_section",
]
