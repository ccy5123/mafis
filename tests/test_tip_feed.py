"""Tests for the tip_feed module — fetching, formatting, and consumption."""

from __future__ import annotations

import datetime as dt

import pytest

from wise_investor.data.tip_feed import (
    TipBundle,
    fetch_macro_tips,
    fetch_ticker_tips,
    fetch_tips_for_run,
    format_tips_block,
    mark_consumed_for_run,
)
from wise_investor.ingest.tip_store import TipStore


@pytest.fixture
def store(tmp_path) -> TipStore:
    return TipStore(db_path=tmp_path / "tips.sqlite")


def _seed(store, **overrides):
    """Create a tip with sane defaults that tests can override per-call."""
    defaults = dict(
        raw_text="placeholder",
        category="ticker",
        detected_tickers=["NVDA"],
        topics=[],
        sender="cyjoe",
    )
    defaults.update(overrides)
    return store.record_tip(**defaults)


# ---------------------------------------------------------------------------
# fetch_ticker_tips
# ---------------------------------------------------------------------------


def test_fetch_ticker_tips_returns_matching_unconsumed(store) -> None:
    _seed(store, raw_text="NVDA earnings strong", detected_tickers=["NVDA"])
    _seed(store, raw_text="TSM bullish", category="ticker", detected_tickers=["TSM"])

    tips = fetch_ticker_tips("NVDA", run_tag="run1", store=store)
    assert len(tips) == 1
    assert "NVDA earnings" in tips[0].raw_text


def test_fetch_ticker_tips_excludes_already_consumed(store) -> None:
    t1 = _seed(store, raw_text="first")
    _seed(store, raw_text="second")

    store.mark_consumed(t1.id, "run1")

    tips = fetch_ticker_tips("NVDA", run_tag="run1", store=store)
    assert [t.raw_text for t in tips] == ["second"]


def test_fetch_ticker_tips_empty_run_tag_returns_empty(store) -> None:
    _seed(store, raw_text="x")
    assert fetch_ticker_tips("NVDA", run_tag="", store=store) == []


def test_fetch_ticker_tips_empty_symbol_returns_empty(store) -> None:
    _seed(store, raw_text="x")
    assert fetch_ticker_tips("", run_tag="run1", store=store) == []


def test_fetch_ticker_tips_filters_macro_category(store) -> None:
    """Even if a macro tip happens to mention a ticker (it shouldn't,
    but defense in depth), we don't promote it to the ticker bucket.
    """
    _seed(
        store,
        raw_text="NVDA mention but macro context",
        category="macro",
        detected_tickers=["NVDA"],
        topics=["interest_rates"],
    )
    assert fetch_ticker_tips("NVDA", run_tag="run1", store=store) == []


def test_fetch_ticker_tips_respects_age_window(store) -> None:
    old = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%S")
    new = (dt.datetime.now() - dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")
    _seed(store, raw_text="old", received_at=old)
    _seed(store, raw_text="new", received_at=new)

    tips = fetch_ticker_tips("NVDA", run_tag="run1", store=store, max_age_days=7)
    assert [t.raw_text for t in tips] == ["new"]


def test_fetch_ticker_tips_respects_limit(store) -> None:
    for i in range(15):
        _seed(store, raw_text=f"tip {i}")
    tips = fetch_ticker_tips("NVDA", run_tag="run1", store=store, limit=5)
    assert len(tips) == 5


# ---------------------------------------------------------------------------
# fetch_macro_tips
# ---------------------------------------------------------------------------


def test_fetch_macro_tips_returns_macro_categories(store) -> None:
    _seed(store, raw_text="rates", category="macro", detected_tickers=[],
          topics=["interest_rates"])
    _seed(store, raw_text="fx", category="fx", detected_tickers=[], topics=["krw_usd"])
    _seed(store, raw_text="oil", category="commodity", detected_tickers=[], topics=["oil"])
    _seed(store, raw_text="china", category="geopolitics", detected_tickers=[],
          topics=["china"])
    _seed(store, raw_text="semi", category="sector", detected_tickers=[],
          topics=["semiconductor"])
    # These should NOT be returned:
    _seed(store, raw_text="ticker", category="ticker", detected_tickers=["NVDA"])
    _seed(store, raw_text="none", category="none", detected_tickers=[])

    tips = fetch_macro_tips(run_tag="run1", store=store)
    categories = {t.category for t in tips}
    assert categories == {"macro", "fx", "commodity", "geopolitics", "sector"}


def test_fetch_macro_tips_excludes_consumed(store) -> None:
    """Once a tip is consumed by ANY run, it stops re-injecting — even
    a fresh run_tag won't pull it back. Audit trail of who consumed it
    lives in `consumed_by`; the gating signal is just "any consumption".
    """
    t = _seed(store, raw_text="rates", category="macro", detected_tickers=[],
              topics=["interest_rates"])
    store.mark_consumed(t.id, "run1")

    assert fetch_macro_tips(run_tag="run1", store=store) == []
    # A different run also doesn't see it — once consumed, never again.
    assert fetch_macro_tips(run_tag="run2", store=store) == []


# ---------------------------------------------------------------------------
# fetch_tips_for_run — combines both buckets
# ---------------------------------------------------------------------------


def test_fetch_tips_for_run_returns_bundle(store) -> None:
    _seed(store, raw_text="t1", detected_tickers=["NVDA"])
    _seed(store, raw_text="m1", category="macro", detected_tickers=[],
          topics=["interest_rates"])

    bundle = fetch_tips_for_run("NVDA", run_tag="run1", store=store)
    assert isinstance(bundle, TipBundle)
    assert len(bundle.ticker) == 1
    assert len(bundle.macro) == 1
    assert not bundle.is_empty


def test_bundle_is_empty_when_no_tips(store) -> None:
    bundle = fetch_tips_for_run("NVDA", run_tag="run1", store=store)
    assert bundle.is_empty
    assert bundle.all_tips() == []


# ---------------------------------------------------------------------------
# format_tips_block
# ---------------------------------------------------------------------------


def test_format_tips_block_empty_returns_empty_string(store) -> None:
    bundle = TipBundle(ticker=[], macro=[])
    assert format_tips_block(bundle, symbol="NVDA") == ""


def test_format_tips_block_renders_disclaimer(store) -> None:
    t = _seed(store, raw_text="strong earnings")
    bundle = TipBundle(ticker=[t], macro=[])
    out = format_tips_block(bundle, symbol="NVDA")
    # Must carry the "do not cite as numeric source" rule embedded in
    # the block — the agent sees the constraint right next to the data.
    assert "user_provided_tips" in out
    assert "HUMAN-PROVIDED HYPOTHESES" in out
    assert "do NOT" in out and "user_tip.telegram" in out
    assert "tool_output" in out


def test_format_tips_block_separates_ticker_and_macro_sections(store) -> None:
    t1 = _seed(store, raw_text="ticker tip", detected_tickers=["NVDA"])
    t2 = _seed(
        store, raw_text="macro tip",
        category="macro", detected_tickers=[], topics=["interest_rates"],
    )
    bundle = TipBundle(ticker=[t1], macro=[t2])
    out = format_tips_block(bundle, symbol="NVDA")
    assert "Ticker-specific tips (NVDA)" in out
    assert "Macro / sector / geopolitics tips" in out
    assert "ticker tip" in out
    assert "macro tip" in out


def test_format_tips_block_truncates_long_tip(store) -> None:
    long = "X" * 500
    t = _seed(store, raw_text=long)
    out = format_tips_block(TipBundle(ticker=[t], macro=[]), symbol="NVDA")
    assert "…" in out
    # Block as a whole still bounded — no runaway prompt growth.
    assert len(out) < 1500


def test_format_tips_block_includes_topics_for_macro(store) -> None:
    t = _seed(
        store, raw_text="rates",
        category="macro", detected_tickers=[],
        topics=["interest_rates", "fed"],
    )
    out = format_tips_block(TipBundle(ticker=[], macro=[t]), symbol="NVDA")
    assert "interest_rates" in out and "fed" in out


# ---------------------------------------------------------------------------
# mark_consumed_for_run
# ---------------------------------------------------------------------------


def test_mark_consumed_for_run_appends_tag(store) -> None:
    t1 = _seed(store, raw_text="a")
    t2 = _seed(store, raw_text="b")

    n = mark_consumed_for_run([t1, t2], run_tag="run1", store=store)
    assert n == 2

    # Re-fetched tips reflect the consumption.
    f1 = store.get_tip(t1.id)
    f2 = store.get_tip(t2.id)
    assert "run1" in f1.consumed_by
    assert "run1" in f2.consumed_by


def test_mark_consumed_for_run_idempotent(store) -> None:
    t = _seed(store, raw_text="a")
    mark_consumed_for_run([t], run_tag="run1", store=store)
    n_again = mark_consumed_for_run([t], run_tag="run1", store=store)
    # Already consumed by run1 — mark_consumed returns True (idempotent
    # success), counting as 1 update from the helper's perspective.
    assert n_again == 1
    assert store.get_tip(t.id).consumed_by == ["run1"]


def test_mark_consumed_for_run_empty_tag_is_noop(store) -> None:
    t = _seed(store, raw_text="a")
    n = mark_consumed_for_run([t], run_tag="", store=store)
    assert n == 0
    assert store.get_tip(t.id).consumed_by == []


def test_mark_consumed_for_run_swallows_per_tip_error(
    store, monkeypatch
) -> None:
    """A per-tip mark_consumed failure must not abort the whole loop —
    the report has already been produced; we don't want a downstream
    SQLite hiccup to crash the whole crew run.
    """
    t1 = _seed(store, raw_text="a")
    t2 = _seed(store, raw_text="b")

    real_mark = store.mark_consumed
    call_count = {"n": 0}

    def _flaky(tip_id, run_tag):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("mock SQLite error")
        return real_mark(tip_id, run_tag)

    monkeypatch.setattr(store, "mark_consumed", _flaky)
    n = mark_consumed_for_run([t1, t2], run_tag="run1", store=store)
    # Only the second succeeded.
    assert n == 1


# ---------------------------------------------------------------------------
# End-to-end (TipStore + tip_feed) — confirms the dedup contract
# ---------------------------------------------------------------------------


def test_two_runs_same_ticker_dedupe_tips(store) -> None:
    """Same tip injected once on run1, then NOT injected on run2."""
    t = _seed(store, raw_text="strong earnings")

    # Run 1 fetches and consumes.
    bundle1 = fetch_tips_for_run("NVDA", run_tag="run1", store=store)
    assert len(bundle1.ticker) == 1
    mark_consumed_for_run(bundle1.all_tips(), run_tag="run1", store=store)

    # Run 2 (same ticker) should see no tips — already consumed.
    bundle2 = fetch_tips_for_run("NVDA", run_tag="run2", store=store)
    assert bundle2.is_empty


def test_back_to_back_fetches_without_consumption_both_see_the_tip(store) -> None:
    """fetch_tips_for_run is read-only — two consecutive fetches with
    no intervening mark_consumed both see the same tip. Consumption is
    explicit and one-directional.
    """
    _seed(store, raw_text="generic note")

    bundle_a = fetch_tips_for_run("NVDA", run_tag="run-a", store=store)
    bundle_b = fetch_tips_for_run("NVDA", run_tag="run-b", store=store)
    assert len(bundle_a.ticker) == 1
    assert len(bundle_b.ticker) == 1


def test_macro_tip_consumed_by_one_run_invisible_to_same_tag_again(store) -> None:
    t = _seed(
        store, raw_text="rates",
        category="macro", detected_tickers=[], topics=["interest_rates"],
    )
    b1 = fetch_tips_for_run("NVDA", run_tag="run1", store=store)
    assert len(b1.macro) == 1
    mark_consumed_for_run(b1.all_tips(), run_tag="run1", store=store)

    b2 = fetch_tips_for_run("NVDA", run_tag="run1", store=store)
    assert b2.is_empty
