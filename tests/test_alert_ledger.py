"""Tests for the alert dedup ledger — cron-spam prevention."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from wise_investor.alerts.chain_alerts import ChainAlert
from wise_investor.alerts.ledger import (
    DEFAULT_COOLDOWN_HOURS,
    AlertLedger,
)


def _alert(
    target: str = "NVDA",
    node: str = "TSMC",
    title: str = "TSMC Kaohsiung outage",
    source: str = "Reuters",
) -> ChainAlert:
    return ChainAlert(
        target_symbol=target,
        matched_node=node,
        chain_path=[node, target],
        hops=1,
        relation="supplies",
        news_title=title,
        news_source=source,
        news_published="2026-04-24",
        news_kind="google_news",
    )


# ---------------------------------------------------------------------------
# Record + last_sent
# ---------------------------------------------------------------------------


def test_last_sent_none_for_unseen_alert(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    assert ledger.last_sent(_alert()) is None


def test_record_single_alert_populates_last_sent(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a = _alert()
    now = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([a], now=now)
    last = ledger.last_sent(a)
    assert last == now


def test_record_same_key_updates_timestamp(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a = _alert()
    t1 = dt.datetime(2026, 4, 24, 10, 0, 0)
    t2 = dt.datetime(2026, 4, 25, 10, 0, 0)
    ledger.record([a], now=t1)
    ledger.record([a], now=t2)
    # Timestamp bumped; UNIQUE index prevents duplicate rows.
    assert ledger.last_sent(a) == t2


def test_record_different_keys_coexist(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a1 = _alert(title="Story A")
    a2 = _alert(title="Story B")
    a3 = _alert(target="GEV", node="Siemens", title="Story A")  # same title, different (target,node)
    now = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([a1, a2, a3], now=now)
    recent = ledger.list_recent()
    assert len(recent) == 3


# ---------------------------------------------------------------------------
# Cooldown logic via is_new + filter_new
# ---------------------------------------------------------------------------


def test_is_new_true_for_unseen(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    assert ledger.is_new(_alert()) is True


def test_is_new_false_within_cooldown(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a = _alert()
    t0 = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([a], now=t0)
    # 10 hours later, still inside 48h cooldown.
    later = t0 + dt.timedelta(hours=10)
    assert ledger.is_new(a, cooldown_hours=48, now=later) is False


def test_is_new_true_after_cooldown_expires(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a = _alert()
    t0 = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([a], now=t0)
    # 49 hours later, cooldown lapsed.
    later = t0 + dt.timedelta(hours=49)
    assert ledger.is_new(a, cooldown_hours=48, now=later) is True


def test_filter_new_drops_already_sent(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a1 = _alert(title="Old news")
    a2 = _alert(title="Breaking news")
    t0 = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([a1], now=t0)
    later = t0 + dt.timedelta(hours=5)
    filtered = ledger.filter_new([a1, a2], cooldown_hours=48, now=later)
    assert [x.news_title for x in filtered] == ["Breaking news"]


def test_filter_new_preserves_order(tmp_path: Path) -> None:
    """Filtered list preserves input order; we don't sort alphabetically."""
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    alerts = [
        _alert(title="Z"),
        _alert(title="A"),
        _alert(title="M"),
    ]
    filtered = ledger.filter_new(alerts)
    assert [a.news_title for a in filtered] == ["Z", "A", "M"]


def test_filter_new_with_zero_cooldown_never_dedupes(tmp_path: Path) -> None:
    """cooldown_hours=0 disables dedup — everything comes through."""
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    a = _alert()
    ledger.record([a])
    filtered = ledger.filter_new([a], cooldown_hours=0)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# list_recent + prune
# ---------------------------------------------------------------------------


def test_list_recent_orders_by_sent_desc(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    t1 = dt.datetime(2026, 4, 20, 10, 0, 0)
    t2 = dt.datetime(2026, 4, 24, 10, 0, 0)
    ledger.record([_alert(title="Older")], now=t1)
    ledger.record([_alert(title="Newer")], now=t2)
    recent = ledger.list_recent(within_hours=24 * 30)
    assert [r.news_title for r in recent] == ["Newer", "Older"]


def test_list_recent_excludes_old_entries(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    old = dt.datetime.now() - dt.timedelta(days=10)
    ledger.record([_alert(title="Ancient")], now=old)
    recent = ledger.list_recent(within_hours=24)  # 1-day window
    assert recent == []


def test_prune_deletes_old_rows(tmp_path: Path) -> None:
    ledger = AlertLedger(tmp_path / "alerts.sqlite")
    now_dt = dt.datetime.now()
    ledger.record([_alert(title="Old")], now=now_dt - dt.timedelta(days=40))
    ledger.record([_alert(title="Fresh")], now=now_dt)
    deleted = ledger.prune(older_than_hours=24 * 30)  # 30-day cutoff
    assert deleted == 1
    remaining = ledger.list_recent(within_hours=24 * 60)
    assert [r.news_title for r in remaining] == ["Fresh"]


# ---------------------------------------------------------------------------
# Defaults sanity
# ---------------------------------------------------------------------------


def test_default_cooldown_is_reasonable() -> None:
    # 48h is the default; typical news cycle is ~24h, so 2x gives buffer
    # without hiding genuinely fresh re-reports.
    assert 24 <= DEFAULT_COOLDOWN_HOURS <= 96
