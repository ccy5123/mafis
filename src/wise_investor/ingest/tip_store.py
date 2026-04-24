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
        category         TEXT NOT NULL,  -- ticker/macro/fx/sector/
                                         -- geopolitics/commodity/none
        detected_tickers TEXT NOT NULL,  -- JSON array, populated for
                                         -- category=ticker
        topics           TEXT,           -- JSON array of macro topics
                                         -- (e.g. ["interest_rates"])
        lang             TEXT DEFAULT 'ko',
        sender           TEXT,           -- telegram username, optional
        source           TEXT DEFAULT 'telegram',
        consumed_by      TEXT,           -- comma-separated run tags that
                                         -- injected this tip into a crew run
        created_at       TEXT NOT NULL
    )

Notes:
  - detected_tickers and topics are JSON arrays so one tip can cite
    multiple names or topics. Queries use the sqlite3 json1 extension's
    `json_each` for containment lookups.
  - `consumed_by` is append-only (comma-separated tags) — re-running
    the same crew for the same ticker should NOT re-inject the same
    tip, so the injector checks this field to dedupe.
  - No dedup on (raw_text, received_at): the user may legitimately
    forward the same message twice; they're separate events.
  - Schema migration: when opening an older database lacking the
    `category` / `topics` columns, _ensure_schema ALTERs them in.
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


_SCHEMA_TABLE = """
CREATE TABLE IF NOT EXISTS tips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'unknown',
    detected_tickers TEXT NOT NULL,
    topics TEXT,
    lang TEXT NOT NULL DEFAULT 'ko',
    sender TEXT,
    source TEXT NOT NULL DEFAULT 'telegram',
    consumed_by TEXT,
    created_at TEXT NOT NULL
);
"""

# Indexes are created AFTER the ALTER-TABLE migration so they can
# reference columns that might not exist on a legacy database.
_SCHEMA_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_tips_received_at ON tips(received_at);
CREATE INDEX IF NOT EXISTS idx_tips_category ON tips(category);
"""


# Fixed classification vocabulary. Kept in the store module (not
# classifier) so anywhere that queries the table has access to the
# canonical set without importing the LLM-adjacent classifier.
CATEGORIES: frozenset[str] = frozenset(
    {
        "ticker",
        "macro",
        "fx",
        "sector",
        "geopolitics",
        "commodity",
        "none",
        "unknown",  # transient value used before classification completes
    }
)


@dataclass
class Tip:
    """One stored tip row."""

    id: int
    received_at: str               # ISO datetime, e.g. 2026-04-24T11:37:00
    raw_text: str
    category: str                  # ticker/macro/fx/sector/geopolitics/commodity/none
    detected_tickers: list[str]    # normalized uppercase tickers (category=ticker)
    topics: list[str]              # macro/sector/etc. topic slugs
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
            # 1) CREATE TABLE IF NOT EXISTS — no-op on legacy databases.
            c.executescript(_SCHEMA_TABLE)
            # 2) ALTER TABLE ADD COLUMN for missing pieces. PRAGMA
            # table_info returns (cid, name, type, notnull, dflt, pk).
            existing_cols = {
                r[1] for r in c.execute("PRAGMA table_info(tips)").fetchall()
            }
            if "category" not in existing_cols:
                c.execute(
                    "ALTER TABLE tips ADD COLUMN category TEXT NOT NULL "
                    "DEFAULT 'unknown'"
                )
            if "topics" not in existing_cols:
                c.execute("ALTER TABLE tips ADD COLUMN topics TEXT")
            # 3) CREATE INDEX — now safe because all referenced
            # columns exist.
            c.executescript(_SCHEMA_INDEXES)
            c.commit()

    # ---- CRUD -------------------------------------------------------

    def record_tip(
        self,
        raw_text: str,
        category: str = "unknown",
        detected_tickers: list[str] | None = None,
        topics: list[str] | None = None,
        lang: str = "ko",
        sender: str | None = None,
        source: str = "telegram",
        received_at: str | None = None,
    ) -> Tip:
        """Insert a new tip row and return the populated object.

        `detected_tickers` is normalized to uppercase (populated when
        `category == "ticker"`). `topics` is a list of macro topic
        slugs (populated for macro/fx/sector/geopolitics/commodity).
        `raw_text` is stored verbatim; blank text raises ValueError.
        Unknown categories fall back to 'unknown' so classifier
        failures don't block the insert.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("raw_text must not be empty")

        normalized_category = (category or "unknown").strip().lower()
        if normalized_category not in CATEGORIES:
            logger.warning(
                "Unknown tip category %r; persisting as 'unknown'.",
                normalized_category,
            )
            normalized_category = "unknown"

        tickers = [t.strip().upper() for t in (detected_tickers or []) if t.strip()]
        tickers_json = json.dumps(tickers, ensure_ascii=False)

        topic_list = [t.strip().lower() for t in (topics or []) if t.strip()]
        topics_json = (
            json.dumps(topic_list, ensure_ascii=False) if topic_list else None
        )

        ra = received_at or dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        now = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO tips
                    (received_at, raw_text, category, detected_tickers,
                     topics, lang, sender, source, consumed_by, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ra, raw_text, normalized_category, tickers_json,
                    topics_json, lang, sender, source, None, now,
                ),
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
        category: str | None = None,
        categories: list[str] | None = None,
        topic: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Tip]:
        """Return tips ordered newest-first.

        Filters (combinable, all AND):
          - `ticker`: detected_tickers contains an exact match
            (case-insensitive).
          - `category`: exact category match. Pass one of CATEGORIES.
          - `categories`: filter to any in the list (macro pooling
            for Economist context: e.g. ["macro","fx","commodity"]).
          - `topic`: topics JSON array contains the slug (lowercased).
          - `since`: ISO timestamp lower bound on received_at.
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

        if category:
            q += " AND t.category = ?"
            params.append(category.strip().lower())

        if categories:
            placeholders = ",".join(["?"] * len(categories))
            q += f" AND t.category IN ({placeholders})"
            params.extend(c.strip().lower() for c in categories)

        if topic:
            q += (
                " AND t.topics IS NOT NULL AND EXISTS ("
                " SELECT 1 FROM json_each(t.topics) "
                " WHERE LOWER(value) = ?"
                " )"
            )
            params.append(topic.strip().lower())

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

    # topics is NULL-able (older rows pre-migration).
    raw_topics = None
    try:
        raw_topics = row["topics"]
    except IndexError:
        raw_topics = None
    topics: list[str] = []
    if raw_topics:
        try:
            parsed = json.loads(raw_topics)
            if isinstance(parsed, list):
                topics = [str(t).strip().lower() for t in parsed if str(t).strip()]
        except (json.JSONDecodeError, TypeError):
            topics = []

    try:
        category = row["category"] or "unknown"
    except IndexError:
        category = "unknown"

    consumed = [t for t in (row["consumed_by"] or "").split(",") if t]
    return Tip(
        id=int(row["id"]),
        received_at=row["received_at"],
        raw_text=row["raw_text"],
        category=str(category).strip().lower(),
        detected_tickers=[str(t).upper() for t in tickers],
        topics=topics,
        lang=row["lang"] or "ko",
        sender=row["sender"],
        source=row["source"] or "telegram",
        consumed_by=consumed,
        created_at=row["created_at"],
    )


__all__ = ["CATEGORIES", "Tip", "TipStore"]
