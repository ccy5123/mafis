"""SEC EDGAR XBRL companyfacts client for Finnhub-format fundamentals.

Background (P1b 2026-04): Finnhub's `/stock/financials-reported` endpoint
indexes only SEC US-GAAP 10-K XBRL filings. Foreign issuers filing 20-F
(TSM, NVO via IFRS; ASML via US-GAAP-on-20-F) appear in the manifest
universe but get zero entries from Finnhub, blocking quantitative
evaluation. This module fills the gap by hitting EDGAR's companyfacts
API directly and reshaping the response into the same `FinancialsEntry`
DTOs the screening adapters already consume.

API: GET https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
  Returns a concept-major dict: facts.{ns}.{concept}.units.{unit}.[items],
  where each item carries (end, val, fy, fp, form, filed, accn).
  Free, no API key, governed by SEC's fair-use policy (User-Agent must
  identify the project + contact email; ~10 req/s soft cap).

We pivot concept-major → year-major to produce `FinancialsEntry` rows
keyed by fiscal year, mirroring Finnhub's shape. For each (concept, fy)
pair, the most-recently-filed item wins (handles restatements). Concept
labels in the resulting `FinancialLineItem.concept` use the underscore-
separated `{namespace}_{ConceptName}` convention so that
`finnhub.extract_field` matches both us-gaap and ifrs-full variants
listed in `CONCEPT_CANDIDATES` without further renaming. The form
("10-K", "20-F") is preserved on each entry for audit trails.

Cache: companyfacts JSONs are large (multi-MB). Cached on disk for 24h
keyed by (cik, today.isoformat()). Re-runs in the same calendar day
are free.

Used by:
  - live_adapter (focal-ticker fallback when Finnhub returns 0 entries)
  - peer_aggregator (peer fallback for the same reason)

Korean tickers (DART path) and other non-SEC-listed issuers are out of
scope here; they have their own adapters.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from wise_investor.data.finnhub import (
    FinancialLineItem,
    FinancialReport,
    FinancialsEntry,
    FinancialsResponse,
)

logger = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "edgar_facts_cache"
CACHE_TTL_HOURS: int = 24

USER_AGENT = "MAFIS research ccy5123ccy@gmail.com"
BASE_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# SEC fair-use rate limit: stay well under 10 req/s.
_RATE_LIMIT_SLEEP_SEC = 0.2


class EdgarFactsError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Concept layout — which logical field lives where in a FinancialReport
# ---------------------------------------------------------------------------

# `extract_field` searches `entry.report.{ic|bs|cf}` based on the field
# name; we replicate that bucketing when assembling EDGAR-derived entries.
_FIELD_TO_BUCKET: dict[str, str] = {
    # Income statement
    "revenue": "ic",
    "gross_profit": "ic",
    "operating_income": "ic",
    "net_income": "ic",
    "eps_diluted": "ic",
    "eps_basic": "ic",
    "depreciation_and_amortization": "ic",
    # Balance sheet
    "total_assets": "bs",
    "total_stockholders_equity": "bs",
    "cash_and_cash_equivalents": "bs",
    "long_term_debt": "bs",
    "short_term_debt": "bs",
    # Cash flow
    "operating_cash_flow": "cf",
    "capital_expenditure": "cf",
}


def _logical_fields_for_extract() -> list[str]:
    """The logical field names whose `CONCEPT_CANDIDATES` we'll mine
    out of an EDGAR companyfacts response."""
    # Mirror finnhub.CONCEPT_CANDIDATES — but pull dynamically so adding
    # a new logical field there flows here automatically.
    from wise_investor.data.finnhub import CONCEPT_CANDIDATES
    return list(CONCEPT_CANDIDATES.keys())


# ---------------------------------------------------------------------------
# Network: companyfacts fetch with disk cache
# ---------------------------------------------------------------------------


def _companyfacts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _cache_path(cik: str, today: dt.date, cache_dir: Path) -> Path:
    return cache_dir / f"CIK{cik}_{today.isoformat()}.json"


def _load_cached(cik: str, today: dt.date, cache_dir: Path) -> dict | None:
    """Look for a cache entry within TTL. Stale entries return None."""
    if not cache_dir.exists():
        return None
    days_to_check = max(1, CACHE_TTL_HOURS // 24)
    for delta in range(days_to_check):
        d = today - dt.timedelta(days=delta)
        path = _cache_path(cik, d, cache_dir)
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as e:
            logger.warning("edgar_facts cache read failed for CIK%s: %s", cik, e)
    return None


def _write_cache(cik: str, today: dt.date, cache_dir: Path, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cik, today, cache_dir)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def fetch_company_facts(
    cik: str,
    *,
    cache: bool = True,
    cache_dir: Path | None = None,
    today: dt.date | None = None,
    http_get: Any | None = None,
) -> dict:
    """Fetch the EDGAR companyfacts JSON for a CIK.

    `cik` is zero-padded to 10 digits. `http_get` is an injection point
    for tests (defaults to `httpx.get`).
    """
    today = today or dt.date.today()
    cache_dir = cache_dir if cache_dir is not None else CACHE_DIR
    cik_padded = str(cik).zfill(10)

    if cache:
        cached = _load_cached(cik_padded, today, cache_dir)
        if cached is not None:
            return cached

    if http_get is None:
        http_get = httpx.get

    time.sleep(_RATE_LIMIT_SLEEP_SEC)
    try:
        r = http_get(
            _companyfacts_url(cik_padded), headers=BASE_HEADERS, timeout=30.0
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        raise EdgarFactsError(
            f"companyfacts fetch failed for CIK{cik_padded}: {e}"
        ) from e

    if cache:
        _write_cache(cik_padded, today, cache_dir, payload)
    return payload


# ---------------------------------------------------------------------------
# Pivot: concept-major → year-major (Finnhub-shape entries)
# ---------------------------------------------------------------------------


def _iter_fy_items(facts: dict) -> list[tuple[str, str, dict]]:
    """Flatten facts → list of (qualified_concept, unit, item_dict)
    for FY items only (drops quarterlies). qualified_concept is
    `{ns}_{ConceptName}`.
    """
    out: list[tuple[str, str, dict]] = []
    ns_map = facts.get("facts", {}) or {}
    for ns, concepts in ns_map.items():
        if not isinstance(concepts, dict):
            continue
        for concept, block in concepts.items():
            units = (block or {}).get("units", {}) or {}
            for unit_name, items in units.items():
                if not isinstance(items, list):
                    continue
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    if it.get("fp") != "FY":
                        continue
                    if it.get("fy") is None:
                        continue
                    qualified = f"{ns}_{concept}"
                    out.append((qualified, unit_name, it))
    return out


def _decide_primary_currency(facts: dict) -> str:
    """Decide a single reporting currency for a ticker's EDGAR facts.

    SEC's companyfacts API exposes IFRS/foreign filers in their native
    currency: TSM in TWD (with optional USD parallel), ASML in EUR,
    NVO in DKK, etc. Quantitative ratios (ROIC, gross margin) are
    self-cancelling within a single currency, so we just need to pick
    ONE currency and use it consistently across every concept and
    every fiscal year for this ticker.

    Strategy: look at the Assets concept (every solvent filer reports
    it). If USD is available (TSM/dual-reported case, or any us-gaap
    domestic filer), prefer USD for cross-Finnhub-comparability.
    Otherwise pick the first plausible currency unit seen.

    Returns "USD" as a safe fallback when no currency could be inferred
    (the entry-builder will then yield zero rows because no concept has
    USD data — graceful degradation, not a crash).
    """
    ns_map = facts.get("facts", {}) or {}
    units_seen: list[str] = []
    for ns, concept in (("us-gaap", "Assets"), ("ifrs-full", "Assets")):
        block = (ns_map.get(ns, {}) or {}).get(concept)
        if not block:
            continue
        units_seen.extend((block.get("units") or {}).keys())
    if "USD" in units_seen:
        return "USD"
    # Common ISO 4217 codes likely to appear on SEC filers.
    known_currencies = {
        "EUR", "DKK", "TWD", "JPY", "GBP", "CNY", "KRW", "CHF", "AUD",
        "CAD", "HKD", "SEK", "NOK", "SGD", "BRL", "INR",
    }
    for u in units_seen:
        if u.upper() in known_currencies:
            return u
    return "USD"


def _pick_best_unit(unit: str, field: str, primary_currency: str) -> bool:
    """Accept the value if its unit matches the ticker's primary currency.

    EPS is special-cased: it's reported per-share, and SEC tags those as
    "USD/shares" or analogous "{currency}/shares". Match anything that
    looks per-share and uses the primary currency stem.
    """
    u = unit.upper()
    pc = primary_currency.upper()
    if field in ("eps_diluted", "eps_basic"):
        return u.startswith(pc) and "SHARES" in u or u.startswith(f"{pc}/SHARE")
    return u == pc


def companyfacts_to_response(
    facts: dict,
    symbol: str,
) -> FinancialsResponse:
    """Pivot a companyfacts JSON into a Finnhub-shape FinancialsResponse.

    For each fiscal year present, emits one FinancialsEntry. For each
    `(concept, fy)` we pick the most recently filed item to handle
    restatements correctly. The entry's form/filed_date come from the
    most-recently-filed item across all concepts in that year.
    """
    from wise_investor.data.finnhub import CONCEPT_CANDIDATES

    cik = str(facts.get("cik") or "").zfill(10) if facts.get("cik") else None

    primary_currency = _decide_primary_currency(facts)

    # We only care about FY annual rows.
    flat = _iter_fy_items(facts)

    # Build a map: fy -> {(qualified_concept, unit) -> (filed, item)}.
    # Key includes unit because a single concept can have multiple
    # currency expressions (TSM reports Assets in both TWD and USD).
    # Lookup at extraction time picks the (concept, primary_currency)
    # tuple — see _pick_best_unit.
    by_year_concept: dict[int, dict[tuple[str, str], tuple[str, dict]]] = {}
    for qualified, unit, it in flat:
        fy = int(it["fy"])
        filed = str(it.get("filed") or "")
        cur = by_year_concept.setdefault(fy, {})
        key = (qualified, unit)
        existing = cur.get(key)
        if existing is None or filed > existing[0]:
            cur[key] = (filed, it)

    entries: list[FinancialsEntry] = []
    for fy in sorted(by_year_concept.keys()):
        per_concept = by_year_concept[fy]

        # Decide entry-level metadata: pick the filing with the most
        # recent `filed` date (this is typically the original 10-K/20-F
        # for that fiscal year, occasionally a restating amendment).
        latest_filed = ""
        latest_form: str | None = None
        latest_end: str | None = None
        latest_start: str | None = None
        latest_accn: str | None = None
        for (filed, it) in per_concept.values():
            if filed > latest_filed:
                latest_filed = filed
                latest_form = it.get("form")
                latest_end = it.get("end")
                latest_start = it.get("start")
                latest_accn = it.get("accn")

        ic_items: list[FinancialLineItem] = []
        bs_items: list[FinancialLineItem] = []
        cf_items: list[FinancialLineItem] = []

        # For each logical field, scan its candidates and pick the first
        # candidate that has a usable value (in primary_currency) for
        # this year. CONCEPT_CANDIDATES is ordered by preference, so
        # this preserves the same priority the Finnhub adapter uses
        # (e.g., RevenueFromContractWithCustomer* before SalesRevenueNet
        # for ASML's ASC-606 era).
        for field, concepts in CONCEPT_CANDIDATES.items():
            picked_value: float | None = None
            picked_concept: str | None = None
            for cand in concepts:
                # Try the (concept, primary_currency) tuple first; if
                # the concept exists under another unit only, the lookup
                # fails here and we fall through to the next candidate.
                pair = per_concept.get((cand, primary_currency))
                if pair is None:
                    # Some EPS items may be tagged with locale-specific
                    # per-share units; scan unit-agnostic for those.
                    matches = [
                        (u, fld_it)
                        for (q, u), fld_it in per_concept.items()
                        if q == cand and _pick_best_unit(u, field, primary_currency)
                    ]
                    if not matches:
                        continue
                    _u, pair = matches[0]
                _filed, item = pair
                val = item.get("val")
                if val is None:
                    continue
                try:
                    picked_value = float(val)
                except (TypeError, ValueError):
                    continue
                picked_concept = cand
                break
            if picked_concept is None or picked_value is None:
                continue
            line = FinancialLineItem(
                concept=picked_concept,
                value=picked_value,
                unit=primary_currency,
                label=picked_concept,
            )
            bucket = _FIELD_TO_BUCKET.get(field, "ic")
            if bucket == "ic":
                ic_items.append(line)
            elif bucket == "bs":
                bs_items.append(line)
            elif bucket == "cf":
                cf_items.append(line)

        if not (ic_items or bs_items or cf_items):
            continue

        entry = FinancialsEntry(
            access_number=latest_accn,
            cik=cik,
            end_date=latest_end,
            filed_date=latest_filed or None,
            form=latest_form,
            start_date=latest_start,
            symbol=symbol.upper(),
            year=fy,
            quarter=None,
            report=FinancialReport(ic=ic_items, bs=bs_items, cf=cf_items),
        )
        entries.append(entry)

    # Finnhub returns newest-first; mirror that ordering for downstream
    # code that relies on `entries[:3]` to mean "last three years".
    entries.sort(key=lambda e: (e.year or 0), reverse=True)

    return FinancialsResponse(cik=cik, symbol=symbol.upper(), data=entries)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def fetch_financials_via_edgar(
    symbol: str,
    *,
    cik: str | None = None,
    cache: bool = True,
    cache_dir: Path | None = None,
    today: dt.date | None = None,
    http_get: Any | None = None,
) -> FinancialsResponse:
    """End-to-end: ticker → CIK → companyfacts → FinancialsResponse.

    `cik` overrides the SEC company_tickers.json lookup (useful when the
    ticker isn't in that map but we know the CIK from elsewhere).
    """
    if cik is None:
        from wise_investor.rag.edgar import ticker_to_cik
        try:
            cik = ticker_to_cik(symbol)
        except Exception as e:
            raise EdgarFactsError(
                f"CIK lookup failed for {symbol}: {e}"
            ) from e

    facts = fetch_company_facts(
        cik,
        cache=cache,
        cache_dir=cache_dir,
        today=today,
        http_get=http_get,
    )
    return companyfacts_to_response(facts, symbol)


__all__ = [
    "CACHE_DIR",
    "CACHE_TTL_HOURS",
    "EdgarFactsError",
    "companyfacts_to_response",
    "fetch_company_facts",
    "fetch_financials_via_edgar",
]
