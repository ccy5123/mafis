"""Extract structured fields from a crew report and render Korean push text.

Deterministic (regex + templated text) so the Universal Citation Rule is
respected and no numeric hallucination can creep in at the translation
layer. If the Steward section is absent or malformed, produce a
best-effort summary with explicit "알 수 없음" (unknown) markers rather
than fabricating values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Korean translations of the fixed verdict vocabulary.
VERDICT_KR = {
    "BUY": "매수",
    "HOLD": "보유",
    "PASS": "관망",
}


@dataclass
class VerdictSummary:
    """Parsed Steward outcome for push-notification rendering."""

    symbol: str
    verdict: str | None  # BUY / HOLD / PASS
    conviction: int | None
    position_sizing: str | None
    bull_thesis: str | None
    top_rebuttal: str | None
    has_economist: bool
    has_skeptic: bool
    has_steward: bool


# ---------------------------------------------------------------------------
# Extraction from the combined markdown report
# ---------------------------------------------------------------------------


_PART_HEADER = re.compile(r"^#\s*Part\s+\d+\s*·\s*([A-Za-z]+)", re.MULTILINE)


def _section_slice(report: str, name: str) -> str | None:
    """Return the substring starting at `# Part N · <name>` up to the next
    `# Part` heading (or EOF). Returns None if no such header is present.
    """
    pattern = re.compile(
        rf"^#\s*Part\s+\d+\s*·\s*{re.escape(name)}\b.*?$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(report)
    if not m:
        return None
    start = m.end()
    # Find the next "# Part ..." header after this one.
    next_m = re.search(r"^#\s*Part\s+\d+\s*·", report[start:], re.MULTILINE)
    end = start + next_m.start() if next_m else len(report)
    return report[start:end]


def _extract_verdict_word(steward_section: str) -> str | None:
    """Find the single-word verdict (BUY/HOLD/PASS) inside the Steward section."""
    # Match after the "## Verdict" heading: the first non-empty line that is
    # exactly one of the known verdicts (optionally wrapped in **bold**).
    m = re.search(
        r"##\s*Verdict\s*\n+\s*\*{0,2}(BUY|HOLD|PASS)\*{0,2}",
        steward_section,
        re.IGNORECASE,
    )
    if m:
        return m.group(1).upper()
    # Fallback: any standalone BUY/HOLD/PASS line in the section.
    for line in steward_section.splitlines():
        stripped = line.strip().strip("*").strip()
        if stripped.upper() in ("BUY", "HOLD", "PASS"):
            return stripped.upper()
    return None


def _extract_conviction(steward_section: str) -> int | None:
    """Find the integer conviction level."""
    # "Conviction: N" anywhere in the section.
    m = re.search(
        r"Conviction\s*[:\-]\s*\*{0,2}(\d+)\*{0,2}",
        steward_section,
        re.IGNORECASE,
    )
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _extract_position_sizing(steward_section: str) -> str | None:
    """Grab the first non-empty line under the Position Sizing Guidance header."""
    m = re.search(
        r"##\s*Position\s*Sizing\s*Guidance\s*\n+(.+?)(?:\n\s*\n|\n##|\Z)",
        steward_section,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    body = m.group(1).strip()
    # Collapse whitespace and take first sentence-ish span (max ~140 chars).
    collapsed = " ".join(body.split())
    return collapsed[:140]


def _extract_rationale_first_paragraph(steward_section: str) -> str | None:
    """Pull the first paragraph of the Rationale block."""
    m = re.search(
        r"##\s*Rationale\s*\n+(.+?)(?:\n\s*\n|\n##|\Z)",
        steward_section,
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return None
    para = m.group(1).strip()
    # Trim to something pushable on Telegram.
    return " ".join(para.split())[:320]


def _split_bull_vs_skeptic(rationale_first: str | None) -> tuple[str | None, str | None]:
    """The Steward's first Rationale paragraph is scripted to contain a
    Bull-thesis sentence followed by the Skeptic's strongest rebuttal.

    We split on 'Skeptic' or 'rebuttal' keywords; imperfect, but deterministic.
    """
    if not rationale_first:
        return None, None
    # Try to locate the Skeptic/rebuttal marker.
    split_pattern = re.compile(
        r"(?:\bThe\s+Skeptic(?:'s|[^a-zA-Z])|\brebuttal\b|\bsurviving\b)",
        re.IGNORECASE,
    )
    m = split_pattern.search(rationale_first)
    if m:
        bull = rationale_first[: m.start()].strip().rstrip(".")
        rebuttal = rationale_first[m.start():].strip().rstrip(".")
        return (bull or None, rebuttal or None)
    # If we cannot split, treat the whole paragraph as the Bull thesis.
    return rationale_first.rstrip("."), None


def extract_verdict_summary(symbol: str, report: str) -> VerdictSummary:
    """Parse a combined crew report into a VerdictSummary."""
    parts = set(m.group(1).lower() for m in _PART_HEADER.finditer(report))
    has_economist = "economist" in parts
    has_skeptic = "skeptic" in parts
    has_steward = "steward" in parts

    steward_section = _section_slice(report, "Steward") or ""
    verdict = _extract_verdict_word(steward_section)
    conviction = _extract_conviction(steward_section)
    position = _extract_position_sizing(steward_section)
    rationale = _extract_rationale_first_paragraph(steward_section)
    bull, rebuttal = _split_bull_vs_skeptic(rationale)

    return VerdictSummary(
        symbol=symbol.upper(),
        verdict=verdict,
        conviction=conviction,
        position_sizing=position,
        bull_thesis=bull,
        top_rebuttal=rebuttal,
        has_economist=has_economist,
        has_skeptic=has_skeptic,
        has_steward=has_steward,
    )


# ---------------------------------------------------------------------------
# Korean summary rendering
# ---------------------------------------------------------------------------


def format_korean_summary(summary: VerdictSummary, report_path: str | None = None) -> str:
    """Render a compact Korean push message (<= ~800 chars)."""
    verdict_kr = VERDICT_KR.get(summary.verdict or "", "알 수 없음")
    conviction_display = (
        f"{summary.conviction}/5" if summary.conviction is not None else "—/5"
    )

    lines: list[str] = []
    lines.append(f"📊 {summary.symbol} 분석 완료")
    lines.append("")
    lines.append(f"판정: *{verdict_kr}* · 확신도 {conviction_display}")
    if summary.position_sizing:
        lines.append(f"포지션 제안: {summary.position_sizing}")
    lines.append("")

    if summary.bull_thesis:
        lines.append(f"불 논거: {summary.bull_thesis}")
    else:
        lines.append("불 논거: 리포트에서 추출 실패 — 본문 확인 필요")

    if summary.top_rebuttal:
        lines.append(f"회의론자 반박: {summary.top_rebuttal}")
    else:
        lines.append("회의론자 반박: 리포트에서 추출 실패")

    missing: list[str] = []
    if not summary.has_economist:
        missing.append("Economist")
    if not summary.has_skeptic:
        missing.append("Skeptic")
    if not summary.has_steward:
        missing.append("Steward")
    if missing:
        lines.append("")
        lines.append(f"⚠️ 리포트에 누락된 파트: {', '.join(missing)}")

    if report_path:
        lines.append("")
        lines.append(f"📄 전체: {report_path}")

    return "\n".join(lines)
