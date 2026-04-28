"""Tests for the tip annotation surface (constitution Sec 7).

Verifies:
  - lookup_tip_annotations returns a per-ticker summary dict, omitting
    tickers with no mentions in the window
  - compute_gap_analysis computes the symmetric difference
  - Window filtering uses TipStore's `since` parameter correctly
  - All-uppercase normalization on ticker keys
  - Render method produces a single-line human-readable string
  - Sample text is truncated, not delivered as full body
"""

from __future__ import annotations

import datetime as dt

from wise_investor.ingest.tip_annotation import (
    DEFAULT_WINDOW_DAYS,
    SAMPLE_TEXT_MAX_CHARS,
    TipAnnotation,
    compute_gap_analysis,
    lookup_tip_annotations,
)
from wise_investor.ingest.tip_store import Tip

# ---------------------------------------------------------------------------
# Stub TipReader
# ---------------------------------------------------------------------------


class _StubStore:
    """In-memory TipReader stub. Records each list_tips call for assertions."""

    def __init__(self, tips: list[Tip] | None = None) -> None:
        self._tips = list(tips or [])
        self.calls: list[dict] = []

    def list_tips(
        self,
        ticker: str | None = None,
        category: str | None = None,
        categories: list[str] | None = None,
        topic: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Tip]:
        self.calls.append({
            "ticker": ticker,
            "category": category,
            "categories": categories,
            "topic": topic,
            "since": since,
            "limit": limit,
        })

        def _matches(tip: Tip) -> bool:
            if ticker and ticker.upper() not in [
                t.upper() for t in tip.detected_tickers
            ]:
                return False
            if category and tip.category != category:
                return False
            return not (since and tip.received_at < since)

        return [t for t in self._tips if _matches(t)]


def _tip(
    *,
    id_: int,
    received_at: str,
    tickers: list[str],
    text: str,
    category: str = "ticker",
) -> Tip:
    return Tip(
        id=id_,
        received_at=received_at,
        raw_text=text,
        category=category,
        detected_tickers=[t.upper() for t in tickers],
        topics=[],
        lang="ko",
        sender=None,
        source="telegram",
        consumed_by=[],
        created_at=received_at,
    )


# ---------------------------------------------------------------------------
# lookup_tip_annotations
# ---------------------------------------------------------------------------


def test_lookup_returns_annotation_for_mentioned_ticker() -> None:
    today = dt.date(2026, 4, 28)
    tips = [
        _tip(id_=1, received_at="2026-04-23", tickers=["NVDA"], text="NVDA looks great"),
        _tip(id_=2, received_at="2026-04-15", tickers=["NVDA"], text="NVDA earnings"),
    ]
    store = _StubStore(tips)
    out = lookup_tip_annotations(["NVDA", "AAPL"], store, today=today)

    assert "NVDA" in out
    assert "AAPL" not in out  # no mentions → omitted
    nvda = out["NVDA"]
    assert nvda.n_mentions == 2
    assert nvda.last_mention_days_ago == 5    # 2026-04-23 → 5 days ago
    assert nvda.first_mention_days_ago == 13  # 2026-04-15 → 13 days ago


def test_lookup_uppercases_ticker_keys() -> None:
    today = dt.date(2026, 4, 28)
    tips = [_tip(id_=1, received_at="2026-04-25", tickers=["nvda"], text="x")]
    store = _StubStore(tips)
    out = lookup_tip_annotations(["nvda"], store, today=today)
    assert "NVDA" in out
    assert "nvda" not in out


def test_lookup_skips_empty_or_whitespace_tickers() -> None:
    today = dt.date(2026, 4, 28)
    out = lookup_tip_annotations(["", "  ", "NVDA"], _StubStore([]), today=today)
    assert out == {}


def test_lookup_uses_window_days_for_since() -> None:
    today = dt.date(2026, 4, 28)
    store = _StubStore([])
    lookup_tip_annotations(["NVDA"], store, window_days=30, today=today)
    # The since= parameter should be (today - 30 days).
    assert store.calls[0]["since"] == "2026-03-29"


def test_lookup_default_window_is_90_days() -> None:
    assert DEFAULT_WINDOW_DAYS == 90


def test_lookup_handles_listing_failure_gracefully() -> None:
    class _FlakyStore(_StubStore):
        def list_tips(self, **kwargs):
            raise RuntimeError("DB locked")

    today = dt.date(2026, 4, 28)
    out = lookup_tip_annotations(["NVDA"], _FlakyStore([]), today=today)
    assert out == {}  # graceful degradation, no crash


def test_lookup_truncates_sample_text() -> None:
    today = dt.date(2026, 4, 28)
    long_text = "x" * (SAMPLE_TEXT_MAX_CHARS + 50)
    tips = [_tip(id_=1, received_at="2026-04-25", tickers=["NVDA"], text=long_text)]
    store = _StubStore(tips)
    out = lookup_tip_annotations(["NVDA"], store, today=today)
    assert out["NVDA"].sample_text.endswith("…")
    assert len(out["NVDA"].sample_text) <= SAMPLE_TEXT_MAX_CHARS + 1


def test_lookup_skips_unparseable_received_at() -> None:
    today = dt.date(2026, 4, 28)
    tips = [_tip(id_=1, received_at="GARBAGE", tickers=["NVDA"], text="x")]
    store = _StubStore(tips)
    out = lookup_tip_annotations(["NVDA"], store, today=today)
    # Unparseable → no annotation produced
    assert "NVDA" not in out


# ---------------------------------------------------------------------------
# TipAnnotation.render
# ---------------------------------------------------------------------------


def test_render_single_mention() -> None:
    a = TipAnnotation(
        ticker="NVDA",
        n_mentions=1,
        first_mention_days_ago=5,
        last_mention_days_ago=5,
        sample_text="x",
    )
    assert "5d ago" in a.render()
    assert "1×" not in a.render()  # single-mention form is different


def test_render_multiple_mentions() -> None:
    a = TipAnnotation(
        ticker="NVDA",
        n_mentions=4,
        first_mention_days_ago=60,
        last_mention_days_ago=2,
        sample_text="x",
    )
    text = a.render()
    assert "4×" in text
    assert "60d" in text
    assert "2d" in text


# ---------------------------------------------------------------------------
# compute_gap_analysis
# ---------------------------------------------------------------------------


def test_gap_analysis_partitions_correctly() -> None:
    today = dt.date(2026, 4, 28)
    tips = [
        _tip(id_=1, received_at="2026-04-25", tickers=["NVDA"], text="x"),
        _tip(id_=2, received_at="2026-04-25", tickers=["AAPL"], text="x"),
        _tip(id_=3, received_at="2026-04-20", tickers=["TSLA"], text="x"),
    ]
    store = _StubStore(tips)

    surfaced = ["NVDA", "MSFT", "GOOG"]  # NVDA in both, MSFT/GOOG only surfaced
    report = compute_gap_analysis(surfaced, store, today=today)

    assert "NVDA" in report.mentioned_and_surfaced
    assert "AAPL" in report.mentioned_only
    assert "TSLA" in report.mentioned_only
    assert "MSFT" in report.surfaced_only
    assert "GOOG" in report.surfaced_only


def test_gap_analysis_mention_counts() -> None:
    today = dt.date(2026, 4, 28)
    tips = [
        _tip(id_=1, received_at="2026-04-25", tickers=["NVDA"], text="x"),
        _tip(id_=2, received_at="2026-04-22", tickers=["NVDA"], text="x"),
        _tip(id_=3, received_at="2026-04-20", tickers=["NVDA", "AAPL"], text="x"),
    ]
    store = _StubStore(tips)
    report = compute_gap_analysis([], store, today=today)
    assert report.by_ticker_mentions["NVDA"] == 3
    assert report.by_ticker_mentions["AAPL"] == 1


def test_gap_analysis_n_mentioned_property() -> None:
    today = dt.date(2026, 4, 28)
    tips = [
        _tip(id_=1, received_at="2026-04-25", tickers=["NVDA", "AAPL"], text="x"),
    ]
    store = _StubStore(tips)
    report = compute_gap_analysis(["NVDA", "MSFT"], store, today=today)
    # NVDA + AAPL mentioned; NVDA + MSFT surfaced.
    # n_mentioned = 2 (NVDA, AAPL); n_surfaced = 2 (NVDA, MSFT).
    assert report.n_mentioned == 2
    assert report.n_surfaced == 2
    assert report.overlap_ratio == 0.5  # 1 in overlap / 2 mentioned


def test_gap_analysis_overlap_ratio_zero_when_no_mentions() -> None:
    today = dt.date(2026, 4, 28)
    store = _StubStore([])
    report = compute_gap_analysis(["NVDA", "MSFT"], store, today=today)
    assert report.overlap_ratio == 0.0


def test_gap_analysis_does_not_filter_by_category() -> None:
    """Gap analysis must include every tip with a detected ticker, not
    just category=ticker. A macro tip that incidentally mentions a
    name still counts as user attention. Keeping the lookup
    category-agnostic also ensures the gap report and the per-ticker
    annotation surface the same set of tickers — two reports that
    disagree on whether the user mentioned NVDA would be confusing."""
    today = dt.date(2026, 4, 28)
    store = _StubStore([])
    compute_gap_analysis(["NVDA"], store, today=today)
    # Stub records the call; verify category filter is NOT set
    assert store.calls[0]["category"] is None


def test_gap_analysis_handles_listing_failure_gracefully() -> None:
    class _FlakyStore(_StubStore):
        def list_tips(self, **kwargs):
            raise RuntimeError("DB error")

    today = dt.date(2026, 4, 28)
    report = compute_gap_analysis(["NVDA"], _FlakyStore([]), today=today)
    # No mentions → NVDA shows up in surfaced_only
    assert report.mentioned_only == ()
    assert "NVDA" in report.surfaced_only
    assert report.by_ticker_mentions == {}


def test_gap_analysis_window_recorded_in_report() -> None:
    today = dt.date(2026, 4, 28)
    store = _StubStore([])
    report = compute_gap_analysis(["NVDA"], store, today=today, window_days=30)
    assert report.window_days == 30


def test_gap_analysis_normalizes_surfaced_tickers_uppercase() -> None:
    today = dt.date(2026, 4, 28)
    tips = [_tip(id_=1, received_at="2026-04-25", tickers=["NVDA"], text="x")]
    store = _StubStore(tips)
    # Pass mixed case
    report = compute_gap_analysis(["nvda", "Msft"], store, today=today)
    assert "NVDA" in report.mentioned_and_surfaced
    assert "MSFT" in report.surfaced_only
