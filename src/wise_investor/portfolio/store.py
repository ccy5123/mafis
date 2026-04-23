"""SQLite-backed portfolio position ledger (design-v2.2 §5.3).

Schema (single table, human-inspectable):

    CREATE TABLE positions (
        symbol           TEXT PRIMARY KEY,
        shares           REAL NOT NULL,
        cost_basis_usd   REAL NOT NULL,      -- total cost paid, USD
        first_bought     TEXT NOT NULL,      -- ISO date of initial entry
        last_updated     TEXT NOT NULL,      -- ISO datetime of last edit
        tier             INTEGER NOT NULL,   -- 1, 2, or 3 per ticker registry
        notes            TEXT DEFAULT ''
    );

Deliberately flat. No lot-level history — adding an existing holding
overwrites shares/cost_basis with the weighted-average new state. If
lot tracking matters later (Phase 4 audit), migrate up from this.

Market value + weight calculations are in-memory: snapshot_weights()
takes a live-quote dict and returns per-symbol market value and
percent-of-portfolio. Steward sizing comparisons call this right
before producing their sizing band.
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
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY,
    shares REAL NOT NULL,
    cost_basis_usd REAL NOT NULL,
    first_bought TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    tier INTEGER NOT NULL,
    notes TEXT NOT NULL DEFAULT ''
);
"""


@dataclass
class Position:
    """One row in the positions table."""

    symbol: str
    shares: float
    cost_basis_usd: float
    first_bought: str  # ISO YYYY-MM-DD
    last_updated: str  # ISO YYYY-MM-DDTHH:MM:SS
    tier: int  # 1, 2, or 3
    notes: str = ""

    @property
    def avg_cost_per_share(self) -> float | None:
        if self.shares <= 0:
            return None
        return self.cost_basis_usd / self.shares


@dataclass
class WeightSnapshot:
    """Per-symbol market value + weight, derived from live quotes."""

    symbol: str
    shares: float
    cost_basis_usd: float
    price: float | None
    market_value_usd: float | None
    weight_pct: float | None  # percent of total portfolio value
    unrealized_pnl_usd: float | None


class PortfolioStore:
    """Thin synchronous wrapper around the positions SQLite DB."""

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

    def upsert_position(
        self,
        symbol: str,
        shares: float,
        cost_basis_usd: float,
        tier: int,
        first_bought: str | None = None,
        notes: str = "",
    ) -> Position:
        """Insert or replace a position row.

        When upserting an existing symbol, callers are expected to pass
        the CUMULATIVE shares + cost_basis (weighted average); this
        function does not merge lots. The `first_bought` column is
        preserved from the original row if one exists.
        """
        symbol = symbol.upper()
        if shares < 0:
            raise ValueError(f"shares must be non-negative, got {shares}")
        if cost_basis_usd < 0:
            raise ValueError(
                f"cost_basis_usd must be non-negative, got {cost_basis_usd}"
            )
        if tier not in {1, 2, 3}:
            raise ValueError(f"tier must be 1, 2, or 3; got {tier}")

        now_iso = dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        existing = self.get_position(symbol)
        fb = (
            first_bought
            if first_bought is not None
            else (existing.first_bought if existing else dt.date.today().isoformat())
        )

        with self._conn() as c:
            c.execute(
                """
                INSERT INTO positions
                    (symbol, shares, cost_basis_usd, first_bought,
                     last_updated, tier, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    shares=excluded.shares,
                    cost_basis_usd=excluded.cost_basis_usd,
                    last_updated=excluded.last_updated,
                    tier=excluded.tier,
                    notes=excluded.notes
                """,
                (symbol, shares, cost_basis_usd, fb, now_iso, tier, notes),
            )
            c.commit()

        result = self.get_position(symbol)
        assert result is not None  # just inserted
        return result

    def delete_position(self, symbol: str) -> bool:
        """Remove a position row. Returns True if a row was deleted."""
        symbol = symbol.upper()
        with self._conn() as c:
            cur = c.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            c.commit()
            return cur.rowcount > 0

    def get_position(self, symbol: str) -> Position | None:
        symbol = symbol.upper()
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM positions WHERE symbol = ?", (symbol,)
            ).fetchone()
            if row is None:
                return None
            return _row_to_position(row)

    def list_positions(self) -> list[Position]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM positions ORDER BY tier ASC, symbol ASC"
            ).fetchall()
            return [_row_to_position(r) for r in rows]

    # ---- derived views ----------------------------------------------

    def snapshot_weights(
        self, prices: dict[str, float | None]
    ) -> list[WeightSnapshot]:
        """Compute per-symbol market value + portfolio weight.

        `prices` maps SYMBOL (uppercase) to latest price in USD. Symbols
        with missing or None prices land in the snapshot with
        market_value_usd = None and weight_pct = None. The total
        portfolio value used for weight_pct is computed ONLY from
        symbols with a known price, so a missing quote for one name
        doesn't inflate the others' denominators.
        """
        positions = self.list_positions()
        mv_by_symbol: dict[str, float | None] = {}
        for p in positions:
            price = prices.get(p.symbol.upper())
            mv = p.shares * price if price is not None else None
            mv_by_symbol[p.symbol] = mv

        total = sum(v for v in mv_by_symbol.values() if v is not None)
        snapshots: list[WeightSnapshot] = []
        for p in positions:
            mv = mv_by_symbol[p.symbol]
            weight = (mv / total * 100.0) if (mv is not None and total > 0) else None
            price = prices.get(p.symbol.upper())
            pnl = (mv - p.cost_basis_usd) if mv is not None else None
            snapshots.append(
                WeightSnapshot(
                    symbol=p.symbol,
                    shares=p.shares,
                    cost_basis_usd=p.cost_basis_usd,
                    price=price,
                    market_value_usd=mv,
                    weight_pct=round(weight, 3) if weight is not None else None,
                    unrealized_pnl_usd=round(pnl, 2) if pnl is not None else None,
                )
            )
        return snapshots

    # ---- Steward sizing-gap helper ----------------------------------

    def sizing_gap(
        self,
        symbol: str,
        suggested_low_pct: float,
        suggested_high_pct: float,
        prices: dict[str, float | None],
    ) -> str:
        """Return a one-line verdict comparing the current weight to
        Steward's suggested sizing band.

        Output shape (for direct paste into a report):
            "Already at 4.2% (suggestion 3.0-5.0% — within band, no action)"
            "Currently 1.5% (suggestion 3.0-5.0% — add 1.5-3.5pp)"
            "Currently 6.0% (suggestion 3.0-5.0% — trim 1.0-3.0pp)"
            "No position (suggestion 3.0-5.0% — new entry opportunity)"
        """
        symbol = symbol.upper()
        if suggested_low_pct > suggested_high_pct:
            raise ValueError("low pct must be <= high pct")
        snaps = self.snapshot_weights(prices)
        match = next((s for s in snaps if s.symbol == symbol), None)
        if match is None or match.weight_pct is None:
            return (
                f"No position (suggestion "
                f"{suggested_low_pct:.1f}-{suggested_high_pct:.1f}% — "
                f"new entry opportunity)"
            )
        w = match.weight_pct
        band = f"{suggested_low_pct:.1f}-{suggested_high_pct:.1f}%"
        if suggested_low_pct <= w <= suggested_high_pct:
            return f"Already at {w:.1f}% (suggestion {band} — within band, no action)"
        if w < suggested_low_pct:
            delta_low = suggested_low_pct - w
            delta_high = suggested_high_pct - w
            return (
                f"Currently {w:.1f}% (suggestion {band} — "
                f"add {delta_low:.1f}-{delta_high:.1f}pp)"
            )
        # w > suggested_high_pct
        delta_low = w - suggested_high_pct
        delta_high = w - suggested_low_pct
        return (
            f"Currently {w:.1f}% (suggestion {band} — "
            f"trim {delta_low:.1f}-{delta_high:.1f}pp)"
        )


def _row_to_position(row: sqlite3.Row) -> Position:
    return Position(
        symbol=row["symbol"],
        shares=row["shares"],
        cost_basis_usd=row["cost_basis_usd"],
        first_bought=row["first_bought"],
        last_updated=row["last_updated"],
        tier=row["tier"],
        notes=row["notes"] or "",
    )


__all__ = ["PortfolioStore", "Position", "WeightSnapshot"]
