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


# Multi-language locale pack for the push-notification renderer.
#
# We render the Steward verdict in the user's preferred language
# (settings.user_language: ko / en / ja / zh). The LLM-generated
# bull_thesis / top_rebuttal text is kept as-is — translation of that
# narrative content is handled by the separate translator module that
# also produces the attached .md file. This split keeps the summary
# renderer deterministic (fixed vocabulary → fixed Korean/Japanese/
# Chinese labels) and restricts the LLM translation call to the long
# report, where the cost is worth paying.
LOCALE: dict[str, dict[str, object]] = {
    "ko": {
        "verdict_map": {"BUY": "매수", "HOLD": "보유", "PASS": "관망"},
        "unknown": "알 수 없음",
        "title": "📊 {symbol} 분석 완료",
        "verdict_line": "판정: *{verdict}* · 확신도 {conv}",
        "audit_downgrade": (
            "⚠️ 감사 자동 조정: LLM 원본 {orig_v} {orig_c} → "
            "매트릭스 기준 {new_v} {new_c}"
        ),
        "position_label": "포지션 제안: {text}",
        "position_override": {
            "PASS": "관망 — 포지션 없음. 재검토 트리거 발생 시까지 보류.",
            "HOLD": "기존 포지션 유지. 신규 매수 없음.",
        },
        "bull_label": "불 논거: {text}",
        "bull_fallback": "불 논거: 리포트에서 추출 실패 — 본문 확인 필요",
        "rebuttal_label": "회의론자 반박: {text}",
        "rebuttal_fallback": "회의론자 반박: 리포트에서 추출 실패",
        "missing_parts": "⚠️ 리포트에 누락된 파트: {parts}",
    },
    "en": {
        "verdict_map": {"BUY": "BUY", "HOLD": "HOLD", "PASS": "PASS"},
        "unknown": "UNKNOWN",
        "title": "📊 {symbol} — analysis complete",
        "verdict_line": "Verdict: *{verdict}* · conviction {conv}",
        "audit_downgrade": (
            "⚠️ Audit auto-adjustment: LLM raw {orig_v} {orig_c} → "
            "discipline matrix {new_v} {new_c}"
        ),
        "position_label": "Position sizing: {text}",
        "position_override": {
            "PASS": "Stand aside — no position until a re-examination trigger fires.",
            "HOLD": "Hold existing position; no new additions.",
        },
        "bull_label": "Bull thesis: {text}",
        "bull_fallback": "Bull thesis: extraction failed — check the report body",
        "rebuttal_label": "Skeptic rebuttal: {text}",
        "rebuttal_fallback": "Skeptic rebuttal: extraction failed",
        "missing_parts": "⚠️ Missing parts in report: {parts}",
    },
    "ja": {
        "verdict_map": {"BUY": "買い", "HOLD": "保有", "PASS": "見送り"},
        "unknown": "不明",
        "title": "📊 {symbol} 分析完了",
        "verdict_line": "判定: *{verdict}* · 確信度 {conv}",
        "audit_downgrade": (
            "⚠️ 監査自動調整: LLM原本 {orig_v} {orig_c} → "
            "規律マトリクス {new_v} {new_c}"
        ),
        "position_label": "ポジション提案: {text}",
        "position_override": {
            "PASS": "見送り — ポジションなし。再検討トリガー発生まで保留。",
            "HOLD": "既存ポジション維持。新規追加なし。",
        },
        "bull_label": "強気論拠: {text}",
        "bull_fallback": "強気論拠: レポートから抽出失敗 — 本文を確認",
        "rebuttal_label": "懐疑派反論: {text}",
        "rebuttal_fallback": "懐疑派反論: レポートから抽出失敗",
        "missing_parts": "⚠️ レポートに欠落パート: {parts}",
    },
    "zh": {
        "verdict_map": {"BUY": "买入", "HOLD": "持有", "PASS": "观望"},
        "unknown": "未知",
        "title": "📊 {symbol} 分析完成",
        "verdict_line": "判定: *{verdict}* · 信心度 {conv}",
        "audit_downgrade": (
            "⚠️ 审计自动调整: LLM 原始 {orig_v} {orig_c} → "
            "纪律矩阵 {new_v} {new_c}"
        ),
        "position_label": "仓位建议: {text}",
        "position_override": {
            "PASS": "观望 — 暂不持仓，等待重新审视触发条件。",
            "HOLD": "维持现有仓位，不新增。",
        },
        "bull_label": "多头论据: {text}",
        "bull_fallback": "多头论据: 报告提取失败 — 请查看正文",
        "rebuttal_label": "怀疑者反驳: {text}",
        "rebuttal_fallback": "怀疑者反驳: 报告提取失败",
        "missing_parts": "⚠️ 报告中缺失部分: {parts}",
    },
}

# Korean verdict mapping — kept as a module-level alias for
# backwards-compat with code that imported `VERDICT_KR` directly.
VERDICT_KR = LOCALE["ko"]["verdict_map"]  # type: ignore[assignment]


@dataclass
class VerdictSummary:
    """Parsed Steward outcome for push-notification rendering."""

    symbol: str
    verdict: str | None  # BUY / HOLD / PASS (audit-corrected when violation)
    conviction: int | None
    position_sizing: str | None
    bull_thesis: str | None
    top_rebuttal: str | None
    has_economist: bool
    has_skeptic: bool
    has_steward: bool
    # Audit flags — populated when the Python discipline audit
    # corrected the Steward's original verdict.
    audit_downgraded: bool = False
    original_verdict: str | None = None
    original_conviction: int | None = None


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


def _truncate_at_sentence(text: str, soft_max: int = 320, hard_max: int = 400) -> str:
    """Clip `text` at a sentence boundary near `soft_max`.

    Walks forward from soft_max looking for the first sentence terminator
    (period / question / exclamation followed by space or end). Falls back
    to a hard cut at `hard_max` if no boundary is found in the window —
    appending "..." so the reader sees the truncation was intentional.
    """
    if len(text) <= soft_max:
        return text
    # Look for a sentence terminator between soft_max and hard_max.
    for i in range(soft_max, min(hard_max, len(text))):
        ch = text[i]
        if ch in ".!?" and (i + 1 == len(text) or text[i + 1] in " \n\t"):
            return text[: i + 1]
    # Nothing found — hard cut + ellipsis so the truncation is visible.
    cut = min(hard_max, len(text))
    return text[:cut].rstrip() + "…"


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
    # Trim to something pushable on Telegram, but at sentence boundary.
    return _truncate_at_sentence(" ".join(para.split()))


def _ensure_terminator(text: str) -> str:
    """Guarantee `text` ends with a sentence terminator so Telegram
    readers don't see what looks like a mid-sentence truncation. The
    prior implementation used `.rstrip('.')` which produced fragments
    like `"... from TSMC's"` — fine structurally, but visually the
    user reads it as a truncated message and files a bug.
    """
    stripped = text.rstrip()
    if stripped and stripped[-1] not in ".!?":
        return stripped + "."
    return stripped


def _split_bull_vs_skeptic(rationale_first: str | None) -> tuple[str | None, str | None]:
    """The Steward's first Rationale paragraph is scripted to contain a
    Bull-thesis sentence followed by the Skeptic's strongest rebuttal.

    We split on 'Skeptic' or 'rebuttal' keywords; imperfect, but deterministic.
    Both fragments are normalized to end with a sentence terminator so the
    Telegram renderer doesn't look truncated at the tail.
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
        bull_raw = rationale_first[: m.start()].strip()
        rebuttal_raw = rationale_first[m.start():].strip()
        bull = _ensure_terminator(bull_raw) if bull_raw else None
        rebuttal = _ensure_terminator(rebuttal_raw) if rebuttal_raw else None
        return (bull, rebuttal)
    # If we cannot split, treat the whole paragraph as the Bull thesis.
    return _ensure_terminator(rationale_first), None


def extract_verdict_summary(symbol: str, report: str) -> VerdictSummary:
    """Parse a combined crew report into a VerdictSummary.

    Verdict + conviction are taken from the Python discipline audit
    (audit_steward_section) when it detects a violation, NOT from the
    Steward LLM's own text. This keeps Telegram push, paper-trades
    ledger, and the printed report all consistent on the corrected
    value — otherwise the push would announce "매수" while the ledger
    stored PASS, which happened in the NVDA_20260424_1137 run.

    Position / bull / rebuttal are still drawn from the Steward's
    narrative because the audit only corrects the verdict, not the
    surrounding prose.
    """
    parts = set(m.group(1).lower() for m in _PART_HEADER.finditer(report))
    has_economist = "economist" in parts
    has_skeptic = "skeptic" in parts
    has_steward = "steward" in parts

    steward_section = _section_slice(report, "Steward") or ""

    # Run the discipline audit on the FULL combined report (so the
    # Defender-aware path fires). Prefer the audit's corrected verdict
    # over the Steward's raw words when the audit flagged a violation.
    verdict = _extract_verdict_word(steward_section)
    conviction = _extract_conviction(steward_section)
    audit_downgraded = False
    original_verdict: str | None = None
    original_conviction: int | None = None
    try:
        from wise_investor.agents.steward_audit import audit_steward_section

        audit = audit_steward_section(report)
        if audit.violation and audit.corrected_verdict is not None:
            original_verdict = verdict
            original_conviction = conviction
            verdict = audit.corrected_verdict
            conviction = audit.corrected_conviction
            audit_downgraded = True
    except Exception:
        # Audit failures should never block the summary — fall back to
        # whatever the Steward's own text said.
        pass

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
        audit_downgraded=audit_downgraded,
        original_verdict=original_verdict,
        original_conviction=original_conviction,
    )


# ---------------------------------------------------------------------------
# Korean summary rendering
# ---------------------------------------------------------------------------


SUPPORTED_LANGUAGES: tuple[str, ...] = ("ko", "en", "ja", "zh")


def format_summary(
    summary: VerdictSummary,
    lang: str = "ko",
    report_path: str | None = None,
) -> str:
    """Render a compact push message in the requested language.

    `report_path` is accepted for backwards compat with call sites but
    NOT rendered into the message body — file paths on the Telegram
    push are unclickable on mobile and clutter the summary. The run
    script should send the .md file as a Telegram document instead.

    When the discipline audit downgrades the LLM's verdict (BUY → PASS,
    BUY → HOLD, etc.), we also override the Steward's original
    position-sizing text. The LLM writes sizing aligned to its raw
    verdict (e.g. "2-3% of equity allocation"), which is misleading
    once the audit flips the verdict to PASS. The override substitutes
    a canonical stand-aside / hold message in the target language.
    """
    loc = LOCALE.get(lang, LOCALE["ko"])
    verdict_map = loc["verdict_map"]  # type: ignore[index]
    unknown = loc["unknown"]  # type: ignore[assignment]

    verdict_display = verdict_map.get(summary.verdict or "", unknown)  # type: ignore[attr-defined]
    conviction_display = (
        f"{summary.conviction}/5" if summary.conviction is not None else "—/5"
    )

    lines: list[str] = []
    lines.append(loc["title"].format(symbol=summary.symbol))  # type: ignore[attr-defined]
    lines.append("")
    lines.append(
        loc["verdict_line"].format(  # type: ignore[attr-defined]
            verdict=verdict_display, conv=conviction_display
        )
    )

    # When the Python audit downgraded the LLM's verdict, surface it.
    # The user MUST see this — otherwise the Telegram banner looks
    # consistent with the LLM narrative while the paper-trades ledger
    # tells a different story.
    if summary.audit_downgraded and summary.original_verdict:
        original_localized = verdict_map.get(  # type: ignore[attr-defined]
            summary.original_verdict, summary.original_verdict
        )
        original_conv = (
            f"{summary.original_conviction}/5"
            if summary.original_conviction is not None
            else "—/5"
        )
        lines.append(
            loc["audit_downgrade"].format(  # type: ignore[attr-defined]
                orig_v=original_localized,
                orig_c=original_conv,
                new_v=verdict_display,
                new_c=conviction_display,
            )
        )

    # Fix A: on audit downgrade, replace the LLM-authored sizing text
    # with a verdict-aligned canonical string. Otherwise fall back to
    # whatever the Steward wrote.
    position_text = summary.position_sizing
    if summary.audit_downgraded and summary.verdict in ("PASS", "HOLD"):
        override_map = loc["position_override"]  # type: ignore[index]
        position_text = override_map.get(summary.verdict, summary.position_sizing)  # type: ignore[attr-defined]

    if position_text:
        lines.append(loc["position_label"].format(text=position_text))  # type: ignore[attr-defined]
    lines.append("")

    if summary.bull_thesis:
        lines.append(loc["bull_label"].format(text=summary.bull_thesis))  # type: ignore[attr-defined]
    else:
        lines.append(loc["bull_fallback"])  # type: ignore[arg-type]

    if summary.top_rebuttal:
        lines.append(loc["rebuttal_label"].format(text=summary.top_rebuttal))  # type: ignore[attr-defined]
    else:
        lines.append(loc["rebuttal_fallback"])  # type: ignore[arg-type]

    missing: list[str] = []
    if not summary.has_economist:
        missing.append("Economist")
    if not summary.has_skeptic:
        missing.append("Skeptic")
    if not summary.has_steward:
        missing.append("Steward")
    if missing:
        lines.append("")
        lines.append(
            loc["missing_parts"].format(parts=", ".join(missing))  # type: ignore[attr-defined]
        )

    # Note: report_path intentionally not rendered. The run script
    # follows up with a sendDocument call for mobile-clickable access.
    return "\n".join(lines)


def format_korean_summary(
    summary: VerdictSummary, report_path: str | None = None
) -> str:
    """Backwards-compat wrapper — delegates to `format_summary(lang="ko")`.

    Older call sites and the test suite still import this by name.
    New code should call `format_summary(summary, lang=...)` directly.
    """
    return format_summary(summary, lang="ko", report_path=report_path)
