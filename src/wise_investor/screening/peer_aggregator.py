"""Peer-aggregate computation for industry comparison fields.

The Stage 2 prefilter compares a ticker's ROIC against
`industry_roic_3y_median` to decide moat-axis verdicts. Without that
median the moat axis is forced to NEED_LLM. This module fills the gap
by pulling Finnhub's per-symbol peer list and computing aggregates
across the peer group.

Two outputs feed back into `IndustryAggregates`:
  - `industry_roic_3y_median`: median (across peers) of each peer's
    3-year average ROIC.
  - `industry_gross_margin_3y_std`: standard deviation across the
    pooled (peer × year) gross-margin observations.

Cost discipline is intentional. Per peer we fetch one annual
financials response (~1 Finnhub call). A 5-peer aggregator is 5
calls; on Finnhub's 60/min free tier that's tractable for batch
screening of 30 tickers. Results are cached on disk for 24h so a
re-run in the same day pays nothing.

Korean tickers are NOT supported here in v1 — Finnhub's `peers`
endpoint returns sparse / empty lists for KRX symbols. The `for_kr`
helper returns IndustryAggregates(None, None) so the prefilter handles
KR moat as NEED_LLM, identically to today.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wise_investor.screening.live_adapter import (
    DEFAULT_EFFECTIVE_TAX_RATE,
    IndustryAggregates,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "peer_aggregate_cache"
CACHE_TTL_HOURS: int = 24
DEFAULT_PEER_LIMIT: int = 5  # cap API cost per ticker


# ---------------------------------------------------------------------------
# Client interface
# ---------------------------------------------------------------------------


class PeerCapableClient(Protocol):
    """Finnhub-shape client interface this module needs.

    The real `FinnhubClient` satisfies it; tests pass a stub.
    """

    def peers(self, symbol: str) -> list[str]: ...

    def financials(self, symbol: str, freq: str = "annual") -> Any: ...


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PeerAggregateDetail:
    """Per-peer breakdown returned alongside the aggregate.

    Useful for audit: shows which peers contributed and how many years
    of data each one had. Callers that just want the IndustryAggregates
    can ignore this.
    """

    symbol: str
    n_years: int
    avg_roic_3y: float | None
    gm_observations: tuple[float, ...]


@dataclass(frozen=True)
class PeerAggregateResult:
    """Aggregate + the peer breakdown that produced it."""

    industry_aggregates: IndustryAggregates
    peers_used: tuple[PeerAggregateDetail, ...]
    n_peers_attempted: int
    n_peers_with_data: int


def compute_industry_aggregates(
    symbol: str,
    *,
    client: PeerCapableClient | None = None,
    peer_limit: int = DEFAULT_PEER_LIMIT,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE,
    cache: bool = True,
    cache_dir: Path | None = None,
    today: dt.date | None = None,
) -> PeerAggregateResult:
    """Pull peers and compute aggregates.

    Args:
        symbol: Ticker to compute aggregates for. The ticker itself is
            excluded from the peer set (Finnhub sometimes self-includes).
        client: Anything satisfying PeerCapableClient. None constructs
            a default FinnhubClient (lazy import).
        peer_limit: Cap on how many peers we'll actually fetch
            financials for. Finnhub returns up to ~10; we trim to
            keep API budget bounded.
        effective_tax_rate: Used to derive NOPAT from each peer's
            operating income. 21% default mirrors the US-adapter
            convention (peer aggregation is US-only in v1).
        cache: When True, results are cached on disk for 24h keyed by
            (symbol, today) so a re-run pays no API cost.
        cache_dir: Override for the cache directory (tests use tmp_path).
        today: Override for "today" (cache key + freshness check).
    """
    today = today or dt.date.today()
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR

    if cache:
        cached = _load_cached(symbol, today, cache_dir)
        if cached is not None:
            return cached

    if client is None:
        from wise_investor.data.finnhub import FinnhubClient
        client = FinnhubClient()

    sym = symbol.upper()
    try:
        raw_peers = client.peers(sym)
    except Exception as e:
        logger.warning("peers() failed for %s: %s — returning empty aggregates", sym, e)
        return _empty_result()

    # Drop self-references and dedupe while preserving order.
    seen: set[str] = set()
    peer_list: list[str] = []
    for p in raw_peers:
        if not p or not isinstance(p, str):
            continue
        u = p.upper()
        if u == sym or u in seen:
            continue
        seen.add(u)
        peer_list.append(u)
        if len(peer_list) >= peer_limit:
            break

    n_attempted = len(peer_list)
    if n_attempted == 0:
        return _empty_result()

    per_peer: list[PeerAggregateDetail] = []
    for peer in peer_list:
        try:
            detail = _build_peer_detail(client, peer, effective_tax_rate)
        except Exception as e:
            logger.warning("peer %s aggregation failed: %s", peer, e)
            continue
        per_peer.append(detail)

    n_with_data = sum(1 for d in per_peer if d.n_years > 0)

    # Aggregate computation.
    roic_values = [d.avg_roic_3y for d in per_peer if d.avg_roic_3y is not None]
    if roic_values:
        roic_median: float | None = statistics.median(roic_values)
    else:
        roic_median = None

    pooled_gms = [
        gm
        for d in per_peer
        for gm in d.gm_observations
    ]
    if len(pooled_gms) >= 2:
        gm_std: float | None = statistics.stdev(pooled_gms)
    else:
        gm_std = None

    result = PeerAggregateResult(
        industry_aggregates=IndustryAggregates(
            industry_roic_3y_median=roic_median,
            industry_gross_margin_3y_std=gm_std,
        ),
        peers_used=tuple(per_peer),
        n_peers_attempted=n_attempted,
        n_peers_with_data=n_with_data,
    )

    if cache:
        _write_cache(symbol, today, cache_dir, result)
    return result


# ---------------------------------------------------------------------------
# Per-peer extraction
# ---------------------------------------------------------------------------


def _build_peer_detail(
    client: PeerCapableClient,
    peer_symbol: str,
    tax_rate: float,
) -> PeerAggregateDetail:
    """Compute one peer's 3y avg ROIC + per-year gross margins."""
    from wise_investor.data.finnhub import extract_field, total_debt

    resp = client.financials(peer_symbol, freq="annual")
    entries = list(getattr(resp, "data", []) or [])

    yearly_roic: list[float] = []
    yearly_gm: list[float] = []
    for entry in entries:
        year = getattr(entry, "year", None)
        if year is None:
            continue

        revenue = extract_field(entry, "revenue")
        gross_profit = extract_field(entry, "gross_profit")
        operating_income = extract_field(entry, "operating_income")

        if revenue is not None and revenue != 0 and gross_profit is not None:
            yearly_gm.append(gross_profit / revenue)

        if operating_income is None:
            continue
        debt = total_debt(entry)
        equity = extract_field(entry, "total_stockholders_equity")
        cash = extract_field(entry, "cash_and_cash_equivalents")
        if equity is None:
            continue
        ic = (debt or 0.0) + equity - (cash or 0.0)
        if ic <= 0:
            continue
        nopat = operating_income * (1.0 - tax_rate)
        yearly_roic.append(nopat / ic)

    # Take the most recent 3 years for the ROIC average. Finnhub returns
    # newest entries first, but be defensive about ordering.
    yearly_roic_sorted = yearly_roic[:3]  # rely on newest-first; alternative below if you need to sort by year.
    avg_roic_3y: float | None = (
        sum(yearly_roic_sorted) / len(yearly_roic_sorted)
        if yearly_roic_sorted
        else None
    )

    return PeerAggregateDetail(
        symbol=peer_symbol,
        n_years=len(entries),
        avg_roic_3y=avg_roic_3y,
        gm_observations=tuple(yearly_gm[:3]),  # also cap at most-recent 3
    )


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def _cache_path(symbol: str, today: dt.date, cache_dir: Path) -> Path:
    return cache_dir / f"{symbol.upper()}_{today.isoformat()}.json"


def _load_cached(
    symbol: str, today: dt.date, cache_dir: Path
) -> PeerAggregateResult | None:
    """Look for a cache entry within TTL. Stale entries return None."""
    if not cache_dir.exists():
        return None
    # Try today first; if missing, look back up to TTL/24 days.
    days_to_check = max(1, CACHE_TTL_HOURS // 24)
    for delta in range(days_to_check):
        d = today - dt.timedelta(days=delta)
        path = _cache_path(symbol, d, cache_dir)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as e:
            logger.warning("peer cache read failed for %s: %s", symbol, e)
            continue
        return _deserialize(payload)
    return None


def _write_cache(
    symbol: str, today: dt.date, cache_dir: Path, result: PeerAggregateResult
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(symbol, today, cache_dir)
    payload = _serialize(result)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _serialize(result: PeerAggregateResult) -> dict:
    return {
        "industry_aggregates": {
            "industry_roic_3y_median": result.industry_aggregates.industry_roic_3y_median,
            "industry_gross_margin_3y_std": result.industry_aggregates.industry_gross_margin_3y_std,
        },
        "peers_used": [
            {
                "symbol": d.symbol,
                "n_years": d.n_years,
                "avg_roic_3y": d.avg_roic_3y,
                "gm_observations": list(d.gm_observations),
            }
            for d in result.peers_used
        ],
        "n_peers_attempted": result.n_peers_attempted,
        "n_peers_with_data": result.n_peers_with_data,
    }


def _deserialize(payload: dict) -> PeerAggregateResult:
    aggs = payload["industry_aggregates"]
    peers = tuple(
        PeerAggregateDetail(
            symbol=p["symbol"],
            n_years=p["n_years"],
            avg_roic_3y=p["avg_roic_3y"],
            gm_observations=tuple(p["gm_observations"]),
        )
        for p in payload.get("peers_used", [])
    )
    return PeerAggregateResult(
        industry_aggregates=IndustryAggregates(
            industry_roic_3y_median=aggs["industry_roic_3y_median"],
            industry_gross_margin_3y_std=aggs["industry_gross_margin_3y_std"],
        ),
        peers_used=peers,
        n_peers_attempted=payload["n_peers_attempted"],
        n_peers_with_data=payload["n_peers_with_data"],
    )


def _empty_result() -> PeerAggregateResult:
    return PeerAggregateResult(
        industry_aggregates=IndustryAggregates(
            industry_roic_3y_median=None,
            industry_gross_margin_3y_std=None,
        ),
        peers_used=(),
        n_peers_attempted=0,
        n_peers_with_data=0,
    )


__all__ = [
    "CACHE_TTL_HOURS",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_PEER_LIMIT",
    "PeerAggregateDetail",
    "PeerAggregateResult",
    "PeerCapableClient",
    "compute_industry_aggregates",
]
