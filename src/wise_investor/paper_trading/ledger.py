"""SQLite-backed ledger of Steward verdicts + paper-trade P&L.

Tables (created idempotently in the shared settings.sqlite_path):

    paper_trades(
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        verdict_date TEXT,          -- ISO date, usually report creation date
        verdict TEXT,               -- audit-corrected BUY/HOLD/PASS
        original_verdict TEXT,      -- LLM's raw verdict
        conviction INTEGER,         -- audit-corrected 1..5
        original_conviction INTEGER,
        audit_downgraded INTEGER,   -- 0/1
        price_at_verdict REAL,      -- entry price for the paper trade
        report_path TEXT,           -- filesystem pointer for traceability
        created_at TEXT             -- ISO datetime of the ledger insert
    )

Notes:
  - A report CAN be recorded multiple times (e.g. re-run with fresh
    prices); each insert creates a new row. Dedup by (symbol,
    verdict_date, report_path) is NOT enforced at the schema level so
    historical re-issues remain inspectable.
  - `verdict_date` is when the verdict was FORMED, `created_at` is
    when the ledger row was WRITTEN. They differ when backfilling old
    reports.
  - P&L is computed on demand from a live-price dict; never stored.
"""

from __future__ import annotations

import datetime as dt
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from wise_investor.config import settings


logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    verdict_date TEXT NOT NULL,
    verdict TEXT NOT NULL,
    original_verdict TEXT NOT NULL,
    conviction INTEGER,
    original_conviction INTEGER,
    audit_downgraded INTEGER NOT NULL DEFAULT 0,
    price_at_verdict REAL,
    report_path TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_paper_trades_symbol ON paper_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_paper_trades_verdict_date ON paper_trades(verdict_date);
"""


@dataclass
class PaperTrade:
    """One recorded Steward verdict + entry price."""

    id: int
    symbol: str
    verdict_date: str            # ISO YYYY-MM-DD
    verdict: str                 # BUY / HOLD / PASS (audit-corrected)
    original_verdict: str
    conviction: int | None
    original_conviction: int | None
    audit_downgraded: bool
    price_at_verdict: float | None
    report_path: str | None
    created_at: str              # ISO YYYY-MM-DDTHH:MM:SS


@dataclass
class TradeReturn:
    """Live mark-to-market for one open paper trade."""

    trade: PaperTrade
    current_price: float | None
    return_pct: float | None    # percent change from entry to current


@dataclass
class PerformanceSummary:
    """Aggregate metrics over a set of trades with returns attached."""

    n_trades: int
    by_verdict: dict[str, dict[str, float]]       # {BUY/HOLD/PASS: {n, avg_return, win_rate}}
    by_conviction: dict[int, dict[str, float]]    # {1..5: {n, avg_return}}
    audit_effect: dict[str, float]                # avg_return for downgraded vs not-downgraded BUYs


class PaperTradeLedger:
    """Thin sync wrapper around the paper_trades SQLite table."""

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

    def record_trade(
        self,
        symbol: str,
        verdict: str,
        original_verdict: str,
        verdict_date: str | None = None,
        conviction: int | None = None,
        original_conviction: int | None = None,
        audit_downgraded: bool = False,
        price_at_verdict: float | None = None,
        report_path: str | None = None,
    ) -> PaperTrade:
        """Insert a new paper-trade row and return the populated object."""
        symbol = symbol.upper()
        if verdict.upper() not in {"BUY", "HOLD", "PASS"}:
            raise ValueError(f"verdict must be BUY/HOLD/PASS, got {verdict!r}")
        if conviction is not None and not 1 <= int(conviction) <= 5:
            raise ValueError(f"conviction must be 1..5, got {conviction}")

        vd = verdict_date or dt.date.today().isoformat()
        now = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        with self._conn() as c:
            cur = c.execute(
                """
                INSERT INTO paper_trades
                    (symbol, verdict_date, verdict, original_verdict,
                     conviction, original_conviction, audit_downgraded,
                     price_at_verdict, report_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    vd,
                    verdict.upper(),
                    original_verdict.upper(),
                    conviction,
                    original_conviction,
                    1 if audit_downgraded else 0,
                    price_at_verdict,
                    report_path,
                    now,
                ),
            )
            c.commit()
            trade_id = cur.lastrowid or -1

        trade = self.get_trade(trade_id)
        assert trade is not None
        return trade

    def get_trade(self, trade_id: int) -> PaperTrade | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM paper_trades WHERE id = ?", (trade_id,)
            ).fetchone()
            return _row_to_trade(row) if row else None

    def list_trades(
        self, symbol: str | None = None, verdict: str | None = None
    ) -> list[PaperTrade]:
        q = "SELECT * FROM paper_trades WHERE 1=1"
        params: list = []
        if symbol:
            q += " AND symbol = ?"
            params.append(symbol.upper())
        if verdict:
            q += " AND verdict = ?"
            params.append(verdict.upper())
        q += " ORDER BY verdict_date ASC, id ASC"
        with self._conn() as c:
            rows = c.execute(q, tuple(params)).fetchall()
            return [_row_to_trade(r) for r in rows]

    def delete_trade(self, trade_id: int) -> bool:
        with self._conn() as c:
            cur = c.execute("DELETE FROM paper_trades WHERE id = ?", (trade_id,))
            c.commit()
            return cur.rowcount > 0

    # ---- Derived views ----------------------------------------------

    def current_returns(
        self, prices: dict[str, float | None]
    ) -> list[TradeReturn]:
        """For every open trade, compute return vs entry price using
        the provided live quotes.
        """
        trades = self.list_trades()
        out: list[TradeReturn] = []
        for t in trades:
            price = prices.get(t.symbol.upper())
            if (
                price is None
                or t.price_at_verdict is None
                or t.price_at_verdict <= 0
            ):
                ret = None
            else:
                ret = round((price - t.price_at_verdict) / t.price_at_verdict * 100.0, 3)
            out.append(
                TradeReturn(trade=t, current_price=price, return_pct=ret)
            )
        return out

    def performance_summary(
        self, prices: dict[str, float | None]
    ) -> PerformanceSummary:
        """Aggregate win rate + avg return by verdict, by conviction,
        and by audit-downgrade flag. Trades with no current price are
        excluded from the aggregates (but counted in n_trades).
        """
        returns = self.current_returns(prices)
        n_trades = len(returns)
        priced = [r for r in returns if r.return_pct is not None]

        by_verdict: dict[str, dict[str, float]] = {}
        for verdict in ("BUY", "HOLD", "PASS"):
            subset = [r for r in priced if r.trade.verdict == verdict]
            if not subset:
                continue
            wins = sum(1 for r in subset if (r.return_pct or 0) > 0)
            avg = sum((r.return_pct or 0) for r in subset) / len(subset)
            by_verdict[verdict] = {
                "n": float(len(subset)),
                "avg_return_pct": round(avg, 3),
                "win_rate": round(wins / len(subset), 3),
            }

        by_conviction: dict[int, dict[str, float]] = {}
        for conv in (1, 2, 3, 4, 5):
            subset = [r for r in priced if r.trade.conviction == conv]
            if not subset:
                continue
            avg = sum((r.return_pct or 0) for r in subset) / len(subset)
            by_conviction[conv] = {
                "n": float(len(subset)),
                "avg_return_pct": round(avg, 3),
            }

        audit_effect: dict[str, float] = {}
        buys = [r for r in priced if r.trade.original_verdict == "BUY"]
        if buys:
            downgraded = [r for r in buys if r.trade.audit_downgraded]
            not_downgraded = [r for r in buys if not r.trade.audit_downgraded]
            if downgraded:
                audit_effect["downgraded_avg_return_pct"] = round(
                    sum((r.return_pct or 0) for r in downgraded) / len(downgraded),
                    3,
                )
            if not_downgraded:
                audit_effect["clean_avg_return_pct"] = round(
                    sum((r.return_pct or 0) for r in not_downgraded) / len(not_downgraded),
                    3,
                )

        return PerformanceSummary(
            n_trades=n_trades,
            by_verdict=by_verdict,
            by_conviction=by_conviction,
            audit_effect=audit_effect,
        )


def _row_to_trade(row: sqlite3.Row) -> PaperTrade:
    return PaperTrade(
        id=int(row["id"]),
        symbol=row["symbol"],
        verdict_date=row["verdict_date"],
        verdict=row["verdict"],
        original_verdict=row["original_verdict"],
        conviction=int(row["conviction"]) if row["conviction"] is not None else None,
        original_conviction=(
            int(row["original_conviction"])
            if row["original_conviction"] is not None
            else None
        ),
        audit_downgraded=bool(row["audit_downgraded"]),
        price_at_verdict=(
            float(row["price_at_verdict"])
            if row["price_at_verdict"] is not None
            else None
        ),
        report_path=row["report_path"],
        created_at=row["created_at"],
    )


__all__ = [
    "PaperTrade",
    "PaperTradeLedger",
    "PerformanceSummary",
    "TradeReturn",
]
