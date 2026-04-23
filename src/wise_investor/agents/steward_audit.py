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


@dataclass
class StewardAuditResult:
    """Outcome of auditing a Steward section."""

    verdict: str | None       # parsed BUY / HOLD / PASS (None if unparseable)
    conviction: int | None    # parsed 1-5
    neutralized_count: int
    survived_count: int
    violation: bool           # True if parsed verdict breaks the discipline matrix
    corrected_verdict: str | None   # what verdict the matrix requires
    corrected_conviction: int | None
    notes: list[str]          # human-readable audit narrative


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


def _label_counts(text: str) -> tuple[int, int]:
    """Count SURVIVED / NEUTRALIZED labels inside the Rationale section.

    We do NOT count mentions outside Rationale (e.g. in Confidence
    Caveats) because those restate labels without originating them.
    """
    heading = re.search(r"(?im)^\s*##\s*Rationale\s*$", text)
    if heading is None:
        # No explicit Rationale heading — fall back to whole text.
        scope = text
    else:
        start = heading.end()
        next_h2 = re.search(r"(?im)^\s*##\s", text[start:])
        end = start + (next_h2.start() if next_h2 else len(text) - start)
        scope = text[start:end]
    return (
        len(_NEUTRALIZED_RE.findall(scope)),
        len(_SURVIVED_RE.findall(scope)),
    )


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

    Non-destructive: callers are expected to APPEND an audit note when
    `result.violation` is True, not rewrite the Steward's words.
    """
    verdict = _parse_verdict(text)
    conviction = _parse_conviction(text)
    neutralized, survived = _label_counts(text)

    notes: list[str] = []
    notes.append(
        f"Labels parsed: NEUTRALIZED={neutralized}, SURVIVED={survived}."
    )

    max_verdict, max_conviction = _required_verdict_ceiling(
        neutralized, survived
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
            violation=False,
            corrected_verdict=None,
            corrected_conviction=None,
            notes=notes,
        )

    # Verdict-level check: is the parsed verdict more optimistic than the
    # matrix ceiling allows?
    if _verdict_rank(verdict) > _verdict_rank(max_verdict):
        violation = True
        corrected_verdict = max_verdict
        corrected_conviction = max_conviction
        notes.append(
            f"VIOLATION: reported Verdict={verdict} exceeds matrix ceiling "
            f"{max_verdict} given label counts."
        )

    # Conviction-level check: even if verdict is allowed, is conviction too high?
    elif conviction is not None and conviction > max_conviction and verdict == max_verdict:
        violation = True
        corrected_verdict = verdict
        corrected_conviction = max_conviction
        notes.append(
            f"VIOLATION: Conviction={conviction} exceeds ceiling {max_conviction} "
            f"for Verdict={verdict} given label counts."
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
        violation=violation,
        corrected_verdict=corrected_verdict,
        corrected_conviction=corrected_conviction,
        notes=notes,
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
        f"- Labels parsed: NEUTRALIZED={result.neutralized_count}, "
        f"SURVIVED={result.survived_count}.",
        f"- Reported Verdict: **{result.verdict}** / Conviction: "
        f"{result.conviction if result.conviction is not None else '?'}.",
        f"- Matrix ceiling for these labels: "
        f"**{result.corrected_verdict}** / Conviction "
        f"{result.corrected_conviction}.",
        "- This is an automatic Python post-check; the Steward's narrative "
        "above is left verbatim for audit transparency. The verdict used "
        "downstream is the matrix-corrected value, not the one the "
        "language model wrote.",
    ]
    for n in result.notes:
        audit_lines.append(f"- {n}")

    return steward_text.rstrip() + "\n" + "\n".join(audit_lines) + "\n"


__all__ = [
    "StewardAuditResult",
    "apply_audit_to_section",
    "audit_steward_section",
]
