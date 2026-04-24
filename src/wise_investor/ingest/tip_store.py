"""SQLite-backed store for user-submitted stock tips.

A "tip" is a free-text message the user forwarded into the bot from
an external source (stock group chat, newsletter, personal note).
Each tip carries one or more detected tickers; Phase 2's tip_feed
pulls recent tips for a given ticker into the Analyst's context so
the crew sees what humans are saying about the name.

Stored in the shared settings.sqlite_path alongside paper_trades
and alert_history.

Schema:

    tips(
        id               INTEGER PRIMARY KEY,
        received_at      TEXT NOT NULL,  -- when Telegram delivered it
        raw_text         TEXT NOT NULL,  -- original message body
        detected_tickers TEXT NOT NULL,  -- JSON array, e.g. '["NVDA"]'
        lang             TEXT DEFAULT 'ko',
        sender           TEXT,           -- telegram username, optional
        source           TEXT DEFAULT 'telegram',
        consumed_by      TEXT,           -- comma-separated run tags that
                                         -- injected this tip into a crew run
        created_at       TEXT NOT NULL
    )

Notes:
  - detected_tickers is stored as a JSON array string so one tip can
    cite multiple names ("NVDA 내리면 AMD가 덕볼까?"). Queries use the
    sqlite3 json1 extension's `json_each` for containment lookups.
  - `consumed_by` is append-only (comma-separated tags) — re-running
    the same crew for the same ticker should NOT re-inject the same
    tip, so the injector checks this field to dedupe.
  - No dedup on (raw_text, received_at): the user may legitimately
    forward the same message twice; they're separate events.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from wise_investor.config import settings


logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS tips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    detected_tickers TEXT NOT NULL,
    lang TEXT NOT NULL DEFAULT 'ko',
    sender TEXT,
    source TEXT NOT NULL DEFAULT 'telegram',
    consumed_by TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tips_received_at ON tips(received_at);
"""


@dataclass
class Tip:
    """One stored tip row."""

    id: int
    received_at: str               # ISO datetime, e.g. 2026-04-24T11:37:00
    raw_text: str
    detected_tickers: list[str]    # normalized uppercase tickers
    lang: str
    sender: str | None
    source: str
    consumed_by: list[str]         # list of run tags that already used this tip
    created_at: str


class TipStore:
    """Thin sync wrapper around the tips SQLite table."""

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

    # ---- CRUD -------------------------------------------------------

    def record_tip(
        self,
        raw_text: str,
        detected_tickers: list[str] | None = None,
        lang: str = "ko",
        sender: str | None = None,
        source: str = "telegram",
        received_at: str | None = None,
    ) -> Tip:
        """Insert a new tip row and return the populated object.

        `detected_tickers` is normalized to uppercase; `raw_text` is
        stored verbatim. Blank text raises ValueError so an empty
        Telegram message can't pollute the store.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("raw_text must not be empty")

        tickers = [t.strip().upper() for t in (detected_tickers or []) if t.strip()]
        tickers_json = json.dumps(tickers, ensure_ascii=False)

        ra = received_at or dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        now = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO tips
                    (received_at, raw_text, detected_tickers, lang,
                     sender, source, consumed_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ra, raw_text, tickers_json, lang, sender, source, None, now),
            )
            c.commit()
            tip_id = cur.lastrowid or -1

        tip = self.get_tip(tip_id)
        assert tip is not None
        return tip

    def get_tip(self, tip_id: int) -> Tip | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM tips WHERE id = ?", (tip_id,)
            ).fetchone()
            return _row_to_tip(row) if row else None

    def list_tips(
        self,
        ticker: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Tip]:
        """Return tips ordered newest-first.

        `ticker` filters to rows whose detected_tickers JSON array
        contains an exact match (case-insensitive; input is uppercased).
        `since` is an ISO timestamp lower bound on received_at.
        """
        params: list = []
        q = "SELECT t.* FROM tips t WHERE 1=1"

        if ticker:
            # json1 extension's json_each iterates the array elements.
            q += (
                " AND EXISTS ("
                " SELECT 1 FROM json_each(t.detected_tickers) "
                " WHERE UPPER(value) = ?"
                " )"
            )
            params.append(ticker.upper())

        if since:
            q += " AND t.received_at >= ?"
            params.append(since)

        q += " ORDER BY t.received_at DESC, t.id DESC"
        if limit is not None and limit > 0:
            q += " LIMIT ?"
            params.append(int(limit))

        with self._conn() as c:
            rows = c.execute(q, tuple(params)).fetchall()
            return [_row_to_tip(r) for r in rows]

    def delete_tip(self, tip_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM tips WHERE id = ?", (tip_id,))
            c.commit()
            return cur.rowcount > 0

    # ---- Consumption tracking --------------------------------------

    def mark_consumed(self, tip_id: int, run_tag: str) -> bool:
        """Append `run_tag` to the consumed_by list for this tip.

        Used by the crew's Analyst-context injector so the same tip
        is not fed to the same run twice. Idempotent: re-marking with
        an existing tag is a no-op.
        """
        if not run_tag:
            raise ValueError("run_tag must not be empty")

        with self._conn() as c:
            row = c.execute(
                "SELECT consumed_by FROM tips WHERE id = ?", (tip_id,)
            ).fetchone()
            if row is None:
                return False
            existing = [t for t in (row["consumed_by"] or "").split(",") if t]
            if run_tag in existing:
                return True
            existing.append(run_tag)
            c.execute(
                "UPDATE tips SET consumed_by = ? WHERE id = ?",
                (",".join(existing), tip_id),
            )
            c.commit()
            return True

    def unconsumed_for_run(
        self,
        ticker: str,
        run_tag: str,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Tip]:
        """Return tips matching `ticker` that have NOT yet been
        consumed by `run_tag`. Used by the Phase 2 tip_feed injector.
        """
        matches = self.list_tips(ticker=ticker, since=since, limit=limit)
        return [t for t in matches if run_tag not in t.consumed_by]


def _row_to_tip(row: sqlite3.Row) -> Tip:
    try:
        tickers = json.loads(row["detected_tickers"] or "[]")
        if not isinstance(tickers, list):
            tickers = []
    except (json.JSONDecodeError, TypeError):
        tickers = []
    consumed = [t for t in (row["consumed_by"] or "").split(",") if t]
    return Tip(
        id=int(row["id"]),
        received_at=row["received_at"],
        raw_text=row["raw_text"],
        detected_tickers=[str(t).upper() for t in tickers],
        lang=row["lang"] or "ko",
        sender=row["sender"],
        source=row["source"] or "telegram",
        consumed_by=consumed,
        created_at=row["created_at"],
    )


__all__ = ["Tip", "TipStore"]
