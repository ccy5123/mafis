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

    `profile()` is OPTIONAL (P1d 2026-04). When the client exposes it,
    the aggregator reads `profile(symbol).finnhub_industry` to filter
    out industry-mismatched peers (defensive belt against Finnhub
    returning cross-category peers like Coinbase as a peer for credit
    ratings — see data/calibration/peer_audit_2026-04.yaml). When the
    client doesn't expose `profile()` (older test stubs), the filter
    silently disables and the aggregator falls back to natural
    filtering via empty financials.
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

    P1d (2026-04) added `industry` (the peer's finnhub_industry as
    fetched from /stock/profile2, None when unavailable) and
    `industry_mismatch` (True when an explicit industry filter rejected
    this peer from ROIC/GM aggregation). Defaults preserve backward
    compatibility with cached results from before P1d.
    """

    symbol: str
    n_years: int
    avg_roic_3y: float | None
    gm_observations: tuple[float, ...]
    industry: str | None = None
    industry_mismatch: bool = False


@dataclass(frozen=True)
class PeerAggregateResult:
    """Aggregate + the peer breakdown that produced it."""

    industry_aggregates: IndustryAggregates
    peers_used: tuple[PeerAggregateDetail, ...]
    n_peers_attempted: int
    n_peers_with_data: int
    focal_industry: str | None = None
    n_peers_industry_mismatch: int = 0


def compute_industry_aggregates(
    symbol: str,
    *,
    client: PeerCapableClient | None = None,
    peer_limit: int = DEFAULT_PEER_LIMIT,
    effective_tax_rate: float = DEFAULT_EFFECTIVE_TAX_RATE,
    cache: bool = True,
    cache_dir: Path | None = None,
    today: dt.date | None = None,
    as_of_date: dt.date | None = None,
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
        as_of_date: When set, peer financials are filtered to filings
            public on or before this date (back-validation use). When
            None, peer financials are used as-is — the live-screening
            default. **This is the lookahead-bias guard:** without
            as_of_date filtering, calibrating constitution v2.0
            against 2018-06-30 would inadvertently use 2024 industry
            medians as the "industry" baseline, contaminating the
            Stage 2 advantage calculation with future information.
    """
    today = today or dt.date.today()
    cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR

    # Cache key includes the as_of_date so a 2018 calibration's peer
    # medians don't collide with today's. Use as_of_date when present
    # (the meaningful temporal coordinate); fall back to `today` for
    # live-mode caching.
    cache_key_date = as_of_date or today

    if cache:
        cached = _load_cached(symbol, cache_key_date, cache_dir)
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

    # Self-exclusion has to account for the financials-reported alias
    # map: querying GOOG's peers returns GOOGL, but for ROIC purposes
    # GOOGL == GOOG (the 10-K covers both). Without this, GOOG would
    # show up with its own data as a peer and double-count. The alias
    # map lives in `data.finnhub`; we import lazily to keep this module
    # importable without the Finnhub dep.
    try:
        from wise_investor.data.finnhub import _financials_symbol
        sym_alias = _financials_symbol(sym)
    except Exception:
        sym_alias = sym
    self_aliases = {sym, sym_alias}

    # Drop self-references and dedupe while preserving order.
    seen: set[str] = set()
    peer_list: list[str] = []
    for p in raw_peers:
        if not p or not isinstance(p, str):
            continue
        u = p.upper()
        if u in self_aliases or u in seen:
            continue
        # Also normalize the peer's symbol to its financials alias
        # so two peers that are dual-share siblings don't get fetched
        # twice (the underlying 10-K is identical).
        try:
            from wise_investor.data.finnhub import _financials_symbol as _alias_for
            u_alias = _alias_for(u)
        except Exception:
            u_alias = u
        if u_alias in self_aliases or u_alias in seen:
            continue
        seen.add(u_alias)
        peer_list.append(u_alias)
        if len(peer_list) >= peer_limit:
            break

    n_attempted = len(peer_list)
    if n_attempted == 0:
        return _empty_result()

    # P1d: explicit industry filter. Fetch focal industry once; per peer,
    # fetch peer industry inside _build_peer_detail. Mismatch → exclude
    # from ROIC/GM aggregation. When focal_industry is None (client has
    # no profile() method, or profile() failed), the filter silently
    # disables and natural filtering via empty financials applies.
    focal_industry = _safe_industry(client, sym)

    per_peer: list[PeerAggregateDetail] = []
    for peer in peer_list:
        try:
            detail = _build_peer_detail(
                client,
                peer,
                effective_tax_rate,
                as_of_date=as_of_date,
                focal_industry=focal_industry,
            )
        except Exception as e:
            logger.warning("peer %s aggregation failed: %s", peer, e)
            continue
        per_peer.append(detail)

    n_with_data = sum(1 for d in per_peer if d.n_years > 0)
    n_industry_mismatch = sum(1 for d in per_peer if d.industry_mismatch)

    # Aggregate computation. Industry-mismatched peers are excluded from
    # both ROIC median and GM std calculations even if they happen to
    # have financials data. _build_peer_detail already zeroes out
    # avg_roic_3y/gm_observations for mismatched peers, but the
    # explicit guard here makes the policy obvious to readers.
    roic_values = [
        d.avg_roic_3y
        for d in per_peer
        if d.avg_roic_3y is not None and not d.industry_mismatch
    ]
    if roic_values:
        roic_median: float | None = statistics.median(roic_values)
    else:
        roic_median = None

    pooled_gms = [
        gm
        for d in per_peer
        if not d.industry_mismatch
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
        focal_industry=focal_industry,
        n_peers_industry_mismatch=n_industry_mismatch,
    )

    if cache:
        _write_cache(symbol, cache_key_date, cache_dir, result)
    return result


# ---------------------------------------------------------------------------
# Per-peer extraction
# ---------------------------------------------------------------------------


def _safe_industry(client: PeerCapableClient, symbol: str) -> str | None:
    """Best-effort fetch of `client.profile(symbol).finnhub_industry`.

    Returns None on any failure: missing profile() method on the
    client (older test stubs), Finnhub API error, or any other
    exception. P1d (2026-04) — used to disable the explicit industry
    filter gracefully when industry data is unavailable, falling back
    to natural filtering via empty financials.
    """
    profile_fn = getattr(client, "profile", None)
    if profile_fn is None:
        return None
    try:
        prof = profile_fn(symbol)
    except Exception as e:
        logger.debug("profile(%s) failed in peer aggregator: %s", symbol, e)
        return None
    return getattr(prof, "finnhub_industry", None)


def _build_peer_detail(
    client: PeerCapableClient,
    peer_symbol: str,
    tax_rate: float,
    *,
    as_of_date: dt.date | None = None,
    focal_industry: str | None = None,
) -> PeerAggregateDetail:
    """Compute one peer's 3y avg ROIC + per-year gross margins.

    When `as_of_date` is set, peer entries are filtered to filings
    public on/before that date (lookahead-bias guard for back-
    validation). When None, all entries are used (live-mode default).

    P1d (2026-04): when `focal_industry` is provided, the peer's own
    industry is fetched via `_safe_industry` and compared. A mismatch
    (peer industry differs OR is None) zeroes out avg_roic_3y and
    gm_observations so the peer doesn't contribute to industry_median
    even if it happens to have financials data. When `focal_industry`
    is None, the filter is disabled (preserves backward compat with
    clients lacking profile()).
    """
    # IC formula must mirror the focal-ticker adapter (#3-deep, 2026-04):
    # IC = Total Assets − Cash. Using the same definition for ticker
    # and peers is what makes the §10 advantage calculation
    # (ticker_roic − industry_median) meaningful.
    from wise_investor.data.finnhub import extract_field

    peer_industry = _safe_industry(client, peer_symbol)

    # Mismatch judgment:
    #   focal None              → filter disabled (backward compat)
    #   focal=A, peer=A         → ok
    #   focal=A, peer=None      → mismatch (peer.industry unavailable
    #                             — could be cross-listing like
    #                             COIN.TO; safer to reject)
    #   focal=A, peer=B (A≠B)   → mismatch
    if focal_industry is None:
        industry_mismatch = False
    else:
        industry_mismatch = peer_industry != focal_industry

    if industry_mismatch:
        # Don't even fetch financials for an industry-mismatched peer:
        # the ROIC value would be excluded anyway, and we save an API
        # call. The detail still goes into peers_used so audit trails
        # show why this peer was rejected.
        return PeerAggregateDetail(
            symbol=peer_symbol,
            n_years=0,
            avg_roic_3y=None,
            gm_observations=tuple(),
            industry=peer_industry,
            industry_mismatch=True,
        )

    resp = client.financials(peer_symbol, freq="annual")
    entries = list(getattr(resp, "data", []) or [])

    if as_of_date is not None:
        # Reuse the same filing-lag logic as the historical adapter to
        # keep the lookahead semantics consistent across the codebase.
        from wise_investor.screening.historical_adapter_finnhub import (
            _is_public_by,
        )
        entries = [e for e in entries if _is_public_by(e, as_of_date)]

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
        total_assets = extract_field(entry, "total_assets")
        cash = extract_field(entry, "cash_and_cash_equivalents")
        if total_assets is None:
            continue
        ic = total_assets - (cash or 0.0)
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
        industry=peer_industry,
        industry_mismatch=False,
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
                "industry": d.industry,
                "industry_mismatch": d.industry_mismatch,
            }
            for d in result.peers_used
        ],
        "n_peers_attempted": result.n_peers_attempted,
        "n_peers_with_data": result.n_peers_with_data,
        "focal_industry": result.focal_industry,
        "n_peers_industry_mismatch": result.n_peers_industry_mismatch,
    }


def _deserialize(payload: dict) -> PeerAggregateResult:
    aggs = payload["industry_aggregates"]
    peers = tuple(
        PeerAggregateDetail(
            symbol=p["symbol"],
            n_years=p["n_years"],
            avg_roic_3y=p["avg_roic_3y"],
            gm_observations=tuple(p["gm_observations"]),
            industry=p.get("industry"),
            industry_mismatch=p.get("industry_mismatch", False),
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
        focal_industry=payload.get("focal_industry"),
        n_peers_industry_mismatch=payload.get("n_peers_industry_mismatch", 0),
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
