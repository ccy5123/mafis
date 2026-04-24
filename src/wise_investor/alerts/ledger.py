"""SQLite dedup ledger for chain alerts — prevents cron-spam.

Problem this solves: `scan_chain_alerts.py` is stateless. If the cron
fires hourly and a news item like "TSMC Kaohsiung outage" stays in
the Google News feed for 24+ hours, the same NVDA alert would fire
24+ times. This module records every (target, matched_node,
news_title) triple it has seen recently and filters alerts that
would repeat within a configurable cooldown window.

Uses the shared `settings.sqlite_path` so the human can inspect the
alert history alongside their positions + paper-trades without
juggling multiple database files.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from wise_investor.alerts.chain_alerts import ChainAlert
from wise_investor.config import settings


logger = logging.getLogger(__name__)


DEFAULT_COOLDOWN_HOURS = 48


SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_symbol TEXT NOT NULL,
    matched_node TEXT NOT NULL,
    news_title TEXT NOT NULL,
    news_source TEXT,
    news_published TEXT,
    sent_at TEXT NOT NULL,
    UNIQUE(target_symbol, matched_node, news_title)
);
CREATE INDEX IF NOT EXISTS idx_alert_history_sent_at ON alert_history(sent_at);
CREATE INDEX IF NOT EXISTS idx_alert_history_target ON alert_history(target_symbol);
"""


@dataclass
class AlertRecord:
    """One historical alert the ledger has seen."""

    target_symbol: str
    matched_node: str
    news_title: str
    news_source: str
    news_published: str
    sent_at: str  # ISO datetime


def _alert_key(alert: ChainAlert) -> tuple[str, str, str]:
    """Normalize a ChainAlert to the dedup key."""
    return (
        alert.target_symbol.upper(),
        alert.matched_node,
        alert.news_title.strip(),
    )


class AlertLedger:
    """SQLite-backed dedup store for chain alerts."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        if db_path is None:
            db_path = settings.sqlite_path
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)
            c.commit()

    # ---- Core API ---------------------------------------------------

    def last_sent(self, alert: ChainAlert) -> dt.datetime | None:
        """Return the ISO datetime of the last time this alert fired,
        or None if it's never been recorded.
        """
        sym, node, title = _alert_key(alert)
        with self._conn() as c:
            row = c.execute(
                """
                SELECT sent_at FROM alert_history
                WHERE target_symbol = ? AND matched_node = ? AND news_title = ?
                """,
                (sym, node, title),
            ).fetchone()
        if row is None:
            return None
        try:
            return dt.datetime.fromisoformat(row["sent_at"])
        except (ValueError, TypeError):
            return None

    def is_new(
        self,
        alert: ChainAlert,
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
        now: dt.datetime | None = None,
    ) -> bool:
        """True when the alert is either brand-new OR last sent longer
        than `cooldown_hours` ago.
        """
        last = self.last_sent(alert)
        if last is None:
            return True
        now = now or dt.datetime.now()
        elapsed_hours = (now - last).total_seconds() / 3600.0
        return elapsed_hours >= cooldown_hours

    def filter_new(
        self,
        alerts: Iterable[ChainAlert],
        cooldown_hours: float = DEFAULT_COOLDOWN_HOURS,
        now: dt.datetime | None = None,
    ) -> list[ChainAlert]:
        """Return only the alerts that pass the cooldown check."""
        return [
            a for a in alerts
            if self.is_new(a, cooldown_hours=cooldown_hours, now=now)
        ]

    def record(
        self,
        alerts: Iterable[ChainAlert],
        now: dt.datetime | None = None,
    ) -> int:
        """Insert or refresh the sent_at timestamp for each alert.

        Uses ON CONFLICT(target_symbol, matched_node, news_title) to
        keep the UNIQUE index clean — same key's sent_at gets bumped
        to the new `now`. Returns the number of rows affected.
        """
        iso = (now or dt.datetime.now()).strftime("%Y-%m-%dT%H:%M:%S")
        count = 0
        with self._conn() as c:
            for a in alerts:
                sym, node, title = _alert_key(a)
                c.execute(
                    """
                    INSERT INTO alert_history
                        (target_symbol, matched_node, news_title,
                         news_source, news_published, sent_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(target_symbol, matched_node, news_title)
                    DO UPDATE SET
                        sent_at = excluded.sent_at,
                        news_source = excluded.news_source,
                        news_published = excluded.news_published
                    """,
                    (sym, node, title, a.news_source, a.news_published, iso),
                )
                count += 1
            c.commit()
        return count

    # ---- Utility queries --------------------------------------------

    def list_recent(self, within_hours: float = 168.0) -> list[AlertRecord]:
        """Return every alert sent within the last `within_hours`,
        newest first. Default window is 1 week.
        """
        cutoff = (
            dt.datetime.now() - dt.timedelta(hours=within_hours)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as c:
            rows = c.execute(
                """
                SELECT * FROM alert_history
                WHERE sent_at >= ?
                ORDER BY sent_at DESC
                """,
                (cutoff,),
            ).fetchall()
        return [
            AlertRecord(
                target_symbol=r["target_symbol"],
                matched_node=r["matched_node"],
                news_title=r["news_title"],
                news_source=r["news_source"] or "",
                news_published=r["news_published"] or "",
                sent_at=r["sent_at"],
            )
            for r in rows
        ]

    def prune(self, older_than_hours: float = 720.0) -> int:
        """Delete rows older than `older_than_hours` (default 30 days).
        Returns the number of rows deleted.
        """
        cutoff = (
            dt.datetime.now() - dt.timedelta(hours=older_than_hours)
        ).strftime("%Y-%m-%dT%H:%M:%S")
        with self._conn() as c:
            cur = c.execute(
                "DELETE FROM alert_history WHERE sent_at < ?", (cutoff,)
            )
            c.commit()
            return cur.rowcount


__all__ = [
    "AlertLedger",
    "AlertRecord",
    "DEFAULT_COOLDOWN_HOURS",
]
