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


def test_format_korean_summary_includes_report_path() -> None:
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
    assert "/tmp/nvda.md" in text


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
