"""Tests for the tips SQLite store.

Uses a per-test tmp_path sqlite file so tests don't touch the real
portfolio ledger.
"""

from __future__ import annotations

import json

import pytest

from wise_investor.ingest.tip_store import Tip, TipStore


@pytest.fixture
def store(tmp_path) -> TipStore:
    return TipStore(db_path=tmp_path / "tips.sqlite")


# ---------------------------------------------------------------------------
# Schema + basic insert/get
# ---------------------------------------------------------------------------


def test_schema_creates_tips_table_idempotently(tmp_path) -> None:
    # Two stores on the same file must not fight over schema creation.
    path = tmp_path / "tips.sqlite"
    TipStore(db_path=path)
    TipStore(db_path=path)  # no-op re-init
    # Smoke: an insert should succeed.
    s = TipStore(db_path=path)
    tip = s.record_tip("요즘 엔비디아 좋대", detected_tickers=["NVDA"])
    assert tip.id > 0


def test_record_tip_roundtrip(store: TipStore) -> None:
    tip = store.record_tip(
        raw_text="TSMC 실적 좋다던데",
        detected_tickers=["TSM"],
        lang="ko",
        sender="user_a",
    )
    assert tip.id > 0
    assert tip.raw_text == "TSMC 실적 좋다던데"
    assert tip.detected_tickers == ["TSM"]
    assert tip.lang == "ko"
    assert tip.sender == "user_a"
    assert tip.source == "telegram"
    assert tip.consumed_by == []

    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.id == tip.id
    assert fetched.raw_text == tip.raw_text


def test_record_tip_uppercases_tickers(store: TipStore) -> None:
    tip = store.record_tip("mixed case", detected_tickers=["nvda", "tsm"])
    assert tip.detected_tickers == ["NVDA", "TSM"]


def test_record_tip_strips_blank_tickers(store: TipStore) -> None:
    tip = store.record_tip(
        "text", detected_tickers=["NVDA", "", "  ", "TSM"]
    )
    assert tip.detected_tickers == ["NVDA", "TSM"]


def test_record_tip_with_no_tickers(store: TipStore) -> None:
    """Tip without any detected ticker is still recordable —
    ticker_extractor may return empty when it can't resolve names.
    """
    tip = store.record_tip("흥미로운 뉴스", detected_tickers=None)
    assert tip.detected_tickers == []
    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.detected_tickers == []


def test_record_tip_rejects_empty_text(store: TipStore) -> None:
    with pytest.raises(ValueError):
        store.record_tip("", detected_tickers=["NVDA"])
    with pytest.raises(ValueError):
        store.record_tip("   \n  ", detected_tickers=["NVDA"])


def test_record_tip_preserves_korean_text_in_json(store: TipStore) -> None:
    """detected_tickers is JSON-encoded; Korean characters in the
    tickers list (shouldn't happen normally but defensive) must
    roundtrip without mojibake.
    """
    tip = store.record_tip("엔비디아 오른다", detected_tickers=["NVDA"])
    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.raw_text == "엔비디아 오른다"


# ---------------------------------------------------------------------------
# List + filter
# ---------------------------------------------------------------------------


def test_list_tips_returns_newest_first(store: TipStore) -> None:
    store.record_tip("first", detected_tickers=["NVDA"], received_at="2026-04-20T10:00:00")
    store.record_tip("second", detected_tickers=["NVDA"], received_at="2026-04-22T10:00:00")
    store.record_tip("third", detected_tickers=["NVDA"], received_at="2026-04-21T10:00:00")

    rows = store.list_tips()
    assert [r.raw_text for r in rows] == ["second", "third", "first"]


def test_list_tips_filters_by_ticker(store: TipStore) -> None:
    store.record_tip("nvda news", detected_tickers=["NVDA"])
    store.record_tip("tsm news", detected_tickers=["TSM"])
    store.record_tip("nvda and amd", detected_tickers=["NVDA", "AMD"])

    nvda_tips = store.list_tips(ticker="NVDA")
    assert len(nvda_tips) == 2
    tsm_tips = store.list_tips(ticker="TSM")
    assert len(tsm_tips) == 1
    amd_tips = store.list_tips(ticker="AMD")
    assert len(amd_tips) == 1


def test_list_tips_ticker_filter_case_insensitive(store: TipStore) -> None:
    store.record_tip("news", detected_tickers=["NVDA"])
    assert len(store.list_tips(ticker="nvda")) == 1
    assert len(store.list_tips(ticker="Nvda")) == 1


def test_list_tips_filters_by_since(store: TipStore) -> None:
    store.record_tip("old", received_at="2026-04-01T10:00:00", detected_tickers=["NVDA"])
    store.record_tip("newer", received_at="2026-04-23T10:00:00", detected_tickers=["NVDA"])

    rows = store.list_tips(since="2026-04-15T00:00:00")
    assert [r.raw_text for r in rows] == ["newer"]


def test_list_tips_respects_limit(store: TipStore) -> None:
    for i in range(5):
        store.record_tip(f"tip {i}", detected_tickers=["NVDA"])
    rows = store.list_tips(limit=3)
    assert len(rows) == 3


# ---------------------------------------------------------------------------
# Consumption tracking
# ---------------------------------------------------------------------------


def test_mark_consumed_appends_run_tag(store: TipStore) -> None:
    tip = store.record_tip("x", detected_tickers=["NVDA"])
    assert store.mark_consumed(tip.id, "NVDA_20260424_1015") is True

    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.consumed_by == ["NVDA_20260424_1015"]


def test_mark_consumed_is_idempotent(store: TipStore) -> None:
    tip = store.record_tip("x", detected_tickers=["NVDA"])
    store.mark_consumed(tip.id, "run1")
    store.mark_consumed(tip.id, "run1")  # duplicate

    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.consumed_by == ["run1"]


def test_mark_consumed_supports_multiple_runs(store: TipStore) -> None:
    tip = store.record_tip("x", detected_tickers=["NVDA"])
    store.mark_consumed(tip.id, "run1")
    store.mark_consumed(tip.id, "run2")
    store.mark_consumed(tip.id, "run3")

    fetched = store.get_tip(tip.id)
    assert fetched is not None
    assert fetched.consumed_by == ["run1", "run2", "run3"]


def test_mark_consumed_returns_false_for_missing_tip(store: TipStore) -> None:
    assert store.mark_consumed(999, "run1") is False


def test_mark_consumed_rejects_empty_tag(store: TipStore) -> None:
    tip = store.record_tip("x", detected_tickers=["NVDA"])
    with pytest.raises(ValueError):
        store.mark_consumed(tip.id, "")


def test_unconsumed_for_run_excludes_already_consumed(store: TipStore) -> None:
    t1 = store.record_tip("first", detected_tickers=["NVDA"])
    t2 = store.record_tip("second", detected_tickers=["NVDA"])
    t3 = store.record_tip("third", detected_tickers=["NVDA"])

    store.mark_consumed(t1.id, "run1")
    store.mark_consumed(t3.id, "run1")  # consumed by run1 but not run2

    remaining = store.unconsumed_for_run("NVDA", run_tag="run1")
    assert {t.id for t in remaining} == {t2.id}

    # run2 hasn't consumed any tip yet — should see all three.
    remaining_run2 = store.unconsumed_for_run("NVDA", run_tag="run2")
    assert {t.id for t in remaining_run2} == {t1.id, t2.id, t3.id}


def test_unconsumed_for_run_respects_since_filter(store: TipStore) -> None:
    store.record_tip("old", detected_tickers=["NVDA"], received_at="2026-04-01T10:00:00")
    store.record_tip("new", detected_tickers=["NVDA"], received_at="2026-04-22T10:00:00")

    recent = store.unconsumed_for_run(
        "NVDA", run_tag="run1", since="2026-04-20T00:00:00"
    )
    assert [t.raw_text for t in recent] == ["new"]


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_tip_removes_row(store: TipStore) -> None:
    tip = store.record_tip("delete me", detected_tickers=["NVDA"])
    assert store.delete_tip(tip.id) is True
    assert store.get_tip(tip.id) is None


def test_delete_tip_returns_false_for_missing(store: TipStore) -> None:
    assert store.delete_tip(999) is False


# ---------------------------------------------------------------------------
# Store coexistence with paper_trades ledger (shared SQLite file)
# ---------------------------------------------------------------------------


def test_tip_store_coexists_with_paper_trades_ledger(tmp_path) -> None:
    """The design keeps tips + paper_trades + alert_history in one
    SQLite file for local-first simplicity. Creating both stores
    back-to-back on the same file must not break either schema.
    """
    from wise_investor.paper_trading.ledger import PaperTradeLedger

    path = tmp_path / "portfolio.sqlite"
    ledger = PaperTradeLedger(db_path=path)
    tips = TipStore(db_path=path)

    # Both tables must be writable independently.
    ledger.record_trade(
        symbol="NVDA", verdict="BUY", original_verdict="BUY",
        conviction=4, original_conviction=4,
    )
    tips.record_tip("NVDA looks strong", detected_tickers=["NVDA"])

    assert len(ledger.list_trades()) == 1
    assert len(tips.list_tips()) == 1


# ---------------------------------------------------------------------------
# Internal: JSON array is correctly persisted
# ---------------------------------------------------------------------------


def test_detected_tickers_stored_as_json_array(store: TipStore, tmp_path) -> None:
    """Direct SQL inspection — confirm we're storing a JSON array
    (so sqlite3 json_each containment queries work).
    """
    tip = store.record_tip("x", detected_tickers=["NVDA", "TSM"])

    import sqlite3
    with sqlite3.connect(store.db_path) as c:
        row = c.execute(
            "SELECT detected_tickers FROM tips WHERE id = ?", (tip.id,)
        ).fetchone()
        assert row is not None
        assert json.loads(row[0]) == ["NVDA", "TSM"]
