"""Tests for the notification package.

Covers the extraction/regex behavior of summary.py against synthetic
crew reports, and the graceful-degrade path of telegram.py when the
bot token is absent.
"""

from __future__ import annotations

from wise_investor.notify.summary import (
    VERDICT_KR,
    VerdictSummary,
    extract_verdict_summary,
    format_korean_summary,
)
from wise_investor.notify.telegram import TelegramNotifier


# ---------------------------------------------------------------------------
# Summary extraction
# ---------------------------------------------------------------------------


_FIXTURE_FULL_REPORT = """\
# NVDA — Equity Research Note

_Models: A · B · C · D · E_

---

# Part 1 · Economist

## Rate Cycle
Fed Funds is 3.64% [Source: fred.FEDFUNDS].

---

# Part 2 · Analyst

## 1. Business Summary
NVIDIA makes AI accelerators.

---

# Part 4 · Skeptic

## Attack on the Bull Thesis

### 1. Target claim (Analyst): X

---

# Part 5 · Steward

## Verdict
**BUY**

## Conviction Level
Conviction: 4

Some explanation sentence.

## Rationale
The Bull thesis is that NVIDIA's strong AI accelerator leadership and ecosystem justify its premium multiples. The Skeptic's strongest surviving rebuttal is that the implied FCF growth rate of 20.27% is unusually high compared to historical semiconductor industry averages.

The top two Skeptic rebuttals are NEUTRALIZED by the Bull evidence.

## Position Sizing Guidance
3-5% of equity allocation for this conviction level

## Confidence Caveats
- Something unknown.
"""


def test_extract_verdict_summary_full_report() -> None:
    s = extract_verdict_summary("NVDA", _FIXTURE_FULL_REPORT)
    assert s.symbol == "NVDA"
    assert s.verdict == "BUY"
    assert s.conviction == 4
    assert s.position_sizing is not None
    assert "3-5%" in s.position_sizing
    assert s.bull_thesis is not None
    assert "AI accelerator" in s.bull_thesis
    assert s.top_rebuttal is not None
    assert "20.27%" in s.top_rebuttal or "unusually high" in s.top_rebuttal.lower()
    assert s.has_economist and s.has_skeptic and s.has_steward


def test_extract_verdict_missing_steward_section() -> None:
    report = """
# Part 1 · Economist

Some macro text.

# Part 2 · Analyst

Some analysis.
"""
    s = extract_verdict_summary("NVDA", report)
    assert s.verdict is None
    assert s.conviction is None
    assert s.position_sizing is None
    assert s.has_steward is False


def test_extract_verdict_handles_hold_verdict() -> None:
    report = """
# Part 5 · Steward

## Verdict
HOLD

## Conviction Level
Conviction: 2

## Rationale
Bull says X. Skeptic's rebuttal survives.

## Position Sizing Guidance
No addition at current price; existing position retained.
"""
    s = extract_verdict_summary("GEV", report)
    assert s.verdict == "HOLD"
    assert s.conviction == 2
    assert "No addition" in (s.position_sizing or "")


def test_extract_verdict_handles_pass_verdict_without_bold() -> None:
    report = """
# Part 5 · Steward

## Verdict
PASS

## Conviction Level
Conviction: 1
"""
    s = extract_verdict_summary("XYZ", report)
    assert s.verdict == "PASS"
    assert s.conviction == 1


def test_extract_verdict_conviction_with_stars() -> None:
    report = """
# Part 5 · Steward

## Verdict
**BUY**

## Conviction Level
**Conviction: 5**
"""
    s = extract_verdict_summary("FOO", report)
    assert s.conviction == 5


# ---------------------------------------------------------------------------
# Korean rendering
# ---------------------------------------------------------------------------


def test_format_korean_summary_buy_verdict() -> None:
    summary = VerdictSummary(
        symbol="NVDA",
        verdict="BUY",
        conviction=4,
        position_sizing="3-5% of equity allocation",
        bull_thesis="Strong AI ecosystem and high FCF",
        top_rebuttal="Implied growth unsustainable",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )
    text = format_korean_summary(summary)
    assert "NVDA" in text
    assert "매수" in text  # BUY → 매수
    assert "4/5" in text
    assert "3-5%" in text
    assert "불 논거" in text
    assert "회의론자 반박" in text
    assert "⚠️" not in text  # no missing parts


def test_format_korean_summary_hold_verdict() -> None:
    summary = VerdictSummary(
        symbol="GEV",
        verdict="HOLD",
        conviction=2,
        position_sizing="No addition at current price",
        bull_thesis=None,
        top_rebuttal=None,
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )
    text = format_korean_summary(summary)
    assert "보유" in text
    assert "2/5" in text
    # Missing bull/rebuttal -> fallback text.
    assert "추출 실패" in text


def test_format_korean_summary_flags_missing_parts() -> None:
    summary = VerdictSummary(
        symbol="FOO",
        verdict=None,
        conviction=None,
        position_sizing=None,
        bull_thesis=None,
        top_rebuttal=None,
        has_economist=False,
        has_skeptic=False,
        has_steward=False,
    )
    text = format_korean_summary(summary)
    assert "알 수 없음" in text
    assert "Economist" in text and "Skeptic" in text and "Steward" in text


def test_verdict_kr_mapping_complete() -> None:
    assert VERDICT_KR["BUY"] == "매수"
    assert VERDICT_KR["HOLD"] == "보유"
    assert VERDICT_KR["PASS"] == "관망"


def test_format_korean_summary_does_not_render_report_path() -> None:
    """Policy change: filesystem paths are unclickable on mobile and
    clutter the summary. The run script follows up with a
    sendDocument call instead. The `report_path` arg is still
    accepted for backwards compat but intentionally not rendered.
    """
    summary = VerdictSummary(
        symbol="NVDA",
        verdict="BUY",
        conviction=3,
        position_sizing=None,
        bull_thesis="x",
        top_rebuttal=None,
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )
    text = format_korean_summary(summary, report_path="/tmp/nvda.md")
    assert "/tmp/nvda.md" not in text
    assert "📄" not in text


# ---------------------------------------------------------------------------
# Telegram notifier (no-op when unconfigured)
# ---------------------------------------------------------------------------


def test_telegram_notifier_not_configured_by_default() -> None:
    # Explicitly override with empty values so this test does not depend on
    # the developer's real .env state.
    n = TelegramNotifier(bot_token="", chat_id="")
    assert n.configured is False


def test_telegram_notifier_configured_when_both_set() -> None:
    n = TelegramNotifier(bot_token="fake-token", chat_id="12345")
    assert n.configured is True


def test_telegram_notifier_send_silently_skips_when_unconfigured() -> None:
    n = TelegramNotifier(bot_token="", chat_id="")
    # Must return False and must NOT raise.
    assert n.send("anything") is False


def test_placeholder_token_treated_as_unconfigured() -> None:
    n = TelegramNotifier(bot_token="your_telegram_bot_token", chat_id="12345")
    assert n.configured is False


# ---------------------------------------------------------------------------
# Message chunking — 4096 char Telegram limit
# ---------------------------------------------------------------------------


def test_chunk_text_short_message_single_chunk() -> None:
    from wise_investor.notify.telegram import _chunk_text

    chunks = _chunk_text("short message")
    assert chunks == ["short message"]


def test_chunk_text_long_message_splits_at_paragraph() -> None:
    from wise_investor.notify.telegram import _chunk_text

    # Build a message with clear paragraph boundaries that forces a split.
    para = "A" * 1500
    msg = para + "\n\n" + para + "\n\n" + para  # ~4506 chars
    chunks = _chunk_text(msg)
    assert len(chunks) >= 2
    # Every chunk under the safe cap.
    assert all(len(c) <= 3900 for c in chunks)
    # Original content preserved (minus trimmed whitespace) when joined.
    joined = " ".join(chunks).replace(" ", "")
    assert joined.count("A") == 1500 * 3


def test_chunk_text_falls_back_to_single_newline() -> None:
    """Messages without \\n\\n but with \\n should still split cleanly."""
    from wise_investor.notify.telegram import _chunk_text

    line = "X" * 200
    msg = ("\n".join([line] * 25))  # ~5024 chars, no blank lines
    chunks = _chunk_text(msg)
    assert len(chunks) >= 2
    assert all(len(c) <= 3900 for c in chunks)


def test_chunk_text_hard_cut_when_no_newlines() -> None:
    """Worst case: one enormous line — hard-cut at the limit."""
    from wise_investor.notify.telegram import _chunk_text

    msg = "A" * 10000
    chunks = _chunk_text(msg)
    assert len(chunks) >= 2
    assert all(len(c) <= 3900 for c in chunks)


def test_chunk_text_respects_soft_boundary_at_sentence() -> None:
    """_truncate_at_sentence clips at a period near soft_max."""
    from wise_investor.notify.summary import _truncate_at_sentence

    text = ("This is sentence one. " * 20).strip()  # ~440 chars
    out = _truncate_at_sentence(text, soft_max=100, hard_max=200)
    # Output ends with a period, not mid-word.
    assert out.endswith(".")
    assert len(out) <= 200


def test_chunk_text_hard_cut_when_no_sentence_boundary() -> None:
    from wise_investor.notify.summary import _truncate_at_sentence

    text = "A" * 500
    out = _truncate_at_sentence(text, soft_max=100, hard_max=200)
    # No period anywhere → hard cut with ellipsis marker.
    assert out.endswith("…")
    assert len(out) <= 201  # 200 chars + ellipsis


def test_truncate_short_text_unchanged() -> None:
    from wise_investor.notify.summary import _truncate_at_sentence

    assert _truncate_at_sentence("short.", soft_max=100) == "short."


def test_extract_verdict_uses_audit_corrected_when_defender_downgrades() -> None:
    """When the Defender says 0 DEFENDED / 1 CONCEDED and the Steward
    issues BUY C4, the summary MUST show the audit-corrected PASS C1,
    not the LLM's raw words. Otherwise Telegram and paper_trades
    diverge — exact NVDA_20260424_1137 bug.
    """
    from wise_investor.notify.summary import extract_verdict_summary

    report = """\
# Part 5 · Defender

### Response to Skeptic #1
**Label:** CONCEDED

**Tally:** 0 DEFENDED, 1 CONCEDED

---

# Part 6 · Steward

## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
The Bull thesis is X.

**NEUTRALIZED**: Defender said so.
"""
    s = extract_verdict_summary("NVDA", report)
    # Corrected: 0D/1C → effective S > N → PASS C1.
    assert s.verdict == "PASS"
    assert s.conviction == 1
    assert s.audit_downgraded is True
    assert s.original_verdict == "BUY"
    assert s.original_conviction == 4


def test_extract_verdict_no_audit_action_leaves_steward_values() -> None:
    """Clean reports (no audit violation) keep the Steward's verdict."""
    from wise_investor.notify.summary import extract_verdict_summary

    report = """\
# Part 6 · Steward

## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
- **NEUTRALIZED**: A [Source: fetch.revenue].
- **NEUTRALIZED**: B [Source: fetch.net_income].
"""
    s = extract_verdict_summary("NVDA", report)
    assert s.verdict == "BUY"
    assert s.conviction == 4
    assert s.audit_downgraded is False


def test_format_korean_summary_shows_audit_downgrade_banner() -> None:
    from wise_investor.notify.summary import (
        VerdictSummary,
        format_korean_summary,
    )

    s = VerdictSummary(
        symbol="NVDA",
        verdict="PASS",
        conviction=1,
        position_sizing="No position.",
        bull_thesis="Bull X",
        top_rebuttal="Skeptic Y",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
        audit_downgraded=True,
        original_verdict="BUY",
        original_conviction=4,
    )
    md = format_korean_summary(s)
    assert "감사 자동 조정" in md
    assert "매수 4/5" in md   # original
    assert "관망 1/5" in md   # corrected
    # No filesystem path — user's mobile won't handle it.
    assert "/home" not in md
    assert ".crew.md" not in md


def test_format_korean_summary_omits_audit_line_when_clean() -> None:
    from wise_investor.notify.summary import (
        VerdictSummary,
        format_korean_summary,
    )

    s = VerdictSummary(
        symbol="NVDA",
        verdict="BUY",
        conviction=4,
        position_sizing="3-5%",
        bull_thesis="Strong cash flow",
        top_rebuttal="Supply chain risk",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )
    md = format_korean_summary(s)
    assert "감사 자동 조정" not in md


def test_chunk_text_limit_constant_matches_telegram() -> None:
    from wise_investor.notify.telegram import (
        TELEGRAM_MAX_MESSAGE_LEN,
        _CHUNK_SAFE_LEN,
    )

    assert TELEGRAM_MAX_MESSAGE_LEN == 4096
    # Safe cap leaves room for the "(part i/N)" suffix.
    assert _CHUNK_SAFE_LEN < TELEGRAM_MAX_MESSAGE_LEN


# ---------------------------------------------------------------------------
# Fix A — audit downgrade rewrites position sizing
# ---------------------------------------------------------------------------


def test_format_summary_overrides_position_when_audit_downgrades_to_pass() -> None:
    """When the audit downgrades BUY → PASS, the Steward's original
    BUY-flavored position text (e.g. '2-3% of equity allocation')
    would be misleading on a PASS verdict. The renderer auto-overrides
    with a canonical stand-aside message aligned to the PASS verdict.
    Exact NVDA_20260424_1137 bug scenario.
    """
    from wise_investor.notify.summary import (
        VerdictSummary,
        format_korean_summary,
    )

    s = VerdictSummary(
        symbol="NVDA",
        verdict="PASS",
        conviction=1,
        position_sizing="2-3% of equity allocation for this conviction level.",
        bull_thesis="Bull.",
        top_rebuttal="Rebuttal.",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
        audit_downgraded=True,
        original_verdict="BUY",
        original_conviction=4,
    )
    text = format_korean_summary(s)
    # The Steward's BUY-flavored sizing must NOT appear.
    assert "2-3%" not in text
    # The canonical PASS stand-aside message must appear.
    assert "관망" in text and "포지션 없음" in text


def test_format_summary_overrides_position_when_audit_downgrades_to_hold() -> None:
    from wise_investor.notify.summary import (
        VerdictSummary,
        format_korean_summary,
    )

    s = VerdictSummary(
        symbol="NVDA",
        verdict="HOLD",
        conviction=2,
        position_sizing="Add 2-3% at current prices.",
        bull_thesis="Bull.",
        top_rebuttal="Rebuttal.",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
        audit_downgraded=True,
        original_verdict="BUY",
        original_conviction=4,
    )
    text = format_korean_summary(s)
    assert "2-3%" not in text
    assert "기존 포지션 유지" in text


def test_format_summary_preserves_position_when_no_audit_downgrade() -> None:
    """Clean reports keep the Steward's original sizing text verbatim."""
    from wise_investor.notify.summary import (
        VerdictSummary,
        format_korean_summary,
    )

    s = VerdictSummary(
        symbol="NVDA",
        verdict="BUY",
        conviction=4,
        position_sizing="3-5% of equity allocation",
        bull_thesis=None,
        top_rebuttal=None,
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )
    text = format_korean_summary(s)
    assert "3-5%" in text


# ---------------------------------------------------------------------------
# Fix B — bull / rebuttal keep their trailing punctuation
# ---------------------------------------------------------------------------


def test_split_bull_vs_skeptic_preserves_or_adds_trailing_period() -> None:
    """The raw _split_bull_vs_skeptic used to .rstrip('.') which made
    the Telegram summary look mid-sentence truncated. We now preserve
    (or re-append) the terminator so both fragments read as complete
    sentences.
    """
    from wise_investor.notify.summary import _split_bull_vs_skeptic

    para = (
        "The Bull thesis is that NVIDIA's lead is durable. "
        "The Skeptic's surviving rebuttal is that compute is commoditizing."
    )
    bull, reb = _split_bull_vs_skeptic(para)
    assert bull is not None and reb is not None
    assert bull.endswith(".")
    assert reb.endswith(".")


def test_split_bull_vs_skeptic_appends_period_when_missing() -> None:
    """If the upstream Steward prose cut off without a period, we add one."""
    from wise_investor.notify.summary import _split_bull_vs_skeptic

    # No trailing period on either fragment.
    para = (
        "The Bull thesis is X, Y, and Z "
        "The Skeptic's strongest rebuttal survives on supply concerns"
    )
    bull, reb = _split_bull_vs_skeptic(para)
    assert bull is not None and reb is not None
    assert bull.endswith(".")
    assert reb.endswith(".")


# ---------------------------------------------------------------------------
# Fix C — multi-language summary rendering (ko / en / ja / zh)
# ---------------------------------------------------------------------------


def _make_clean_summary() -> "object":
    from wise_investor.notify.summary import VerdictSummary

    return VerdictSummary(
        symbol="NVDA",
        verdict="BUY",
        conviction=4,
        position_sizing="3-5% of equity allocation.",
        bull_thesis="AI accelerator leadership is durable.",
        top_rebuttal="Implied growth rate looks unsustainable.",
        has_economist=True,
        has_skeptic=True,
        has_steward=True,
    )


def test_format_summary_english_uses_english_labels() -> None:
    from wise_investor.notify.summary import format_summary

    text = format_summary(_make_clean_summary(), lang="en")
    assert "NVDA" in text
    assert "BUY" in text
    assert "conviction" in text.lower()
    assert "4/5" in text
    assert "Bull thesis" in text
    assert "Skeptic rebuttal" in text
    # No Korean labels should leak into the English output.
    assert "판정" not in text
    assert "확신도" not in text


def test_format_summary_japanese_uses_japanese_labels() -> None:
    from wise_investor.notify.summary import format_summary

    text = format_summary(_make_clean_summary(), lang="ja")
    assert "NVDA" in text
    assert "買い" in text
    assert "判定" in text
    assert "確信度" in text
    assert "4/5" in text
    # English verdict token shouldn't replace the Japanese one.
    assert "BUY" not in text


def test_format_summary_chinese_uses_chinese_labels() -> None:
    from wise_investor.notify.summary import format_summary

    text = format_summary(_make_clean_summary(), lang="zh")
    assert "NVDA" in text
    assert "买入" in text
    assert "判定" in text
    assert "信心度" in text
    assert "4/5" in text
    assert "BUY" not in text


def test_format_summary_unknown_lang_falls_back_to_korean() -> None:
    """A typo like 'kr' or 'jp' should not crash the push. Fall back to Korean."""
    from wise_investor.notify.summary import format_summary

    text = format_summary(_make_clean_summary(), lang="kr")  # typo
    # Korean label shows up in the fallback rendering.
    assert "판정" in text


def test_supported_languages_exposes_all_four() -> None:
    from wise_investor.notify.summary import SUPPORTED_LANGUAGES

    assert set(SUPPORTED_LANGUAGES) == {"ko", "en", "ja", "zh"}


def test_user_language_default_is_korean() -> None:
    """Default user_language in Settings is 'ko' so existing deployments
    see no behavior change after the upgrade.
    """
    from wise_investor.config import Settings

    s = Settings()
    assert s.user_language == "ko"



