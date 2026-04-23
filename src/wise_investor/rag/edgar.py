"""SEC EDGAR client for fetching 10-K filings.

Minimal pipeline:
  ticker -> CIK via SEC's company_tickers.json (cached locally)
  CIK -> latest 10-K filing URL via https://data.sec.gov/submissions/CIK{cik}.json
  URL -> primary 10-K HTML document

Respects SEC's fair-use policy:
  - Polite User-Agent identifying the project.
  - Rate limited to ~5 req/s via an explicit time.sleep(0.2) between calls.

No API key required — SEC EDGAR is free and public.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


logger = logging.getLogger(__name__)


# SEC fair-use policy REQUIRES the User-Agent to include company-or-project
# name AND a contact vector (email). Without an email SEC returns 403 on
# every request (verified empirically 2026-04-23).
# Sample format per SEC: "Sample Company Name AdminContact@samplecompany.com"
USER_AGENT = "MAFIS research ccy5123ccy@gmail.com"
BASE_HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
HTML_HEADERS = {"User-Agent": USER_AGENT}

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO_ROOT / "data" / "edgar_cache"


class EdgarError(RuntimeError):
    pass


@dataclass
class FilingRef:
    """Everything we need to download one filing."""

    cik: str
    symbol: str
    form: str
    filing_date: str  # YYYY-MM-DD
    accession_number: str  # e.g. "0001045810-25-000008"
    primary_document: str  # e.g. "nvda-20250126.htm"

    @property
    def filing_index_url(self) -> str:
        acc_nodash = self.accession_number.replace("-", "")
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{acc_nodash}/"
        )

    @property
    def primary_url(self) -> str:
        return self.filing_index_url + self.primary_document


# ---------------------------------------------------------------------------
# CIK lookup (cached)
# ---------------------------------------------------------------------------


def _cik_map_path() -> Path:
    return CACHE_DIR / "company_tickers.json"


def fetch_cik_map(force: bool = False) -> dict[str, str]:
    """Return `{TICKER: CIK}` as strings, with CIK as 10-digit zero-padded.

    Downloads the mapping from sec.gov and caches it locally so repeated
    lookups are free.
    """
    cache = _cik_map_path()
    if not force and cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("CIK map cache unreadable, refetching: %s", e)

    url = "https://www.sec.gov/files/company_tickers.json"
    r = httpx.get(url, headers=BASE_HEADERS, timeout=15.0)
    r.raise_for_status()
    raw = r.json()

    # company_tickers.json is {index: {cik_str, ticker, title}}.
    mapping: dict[str, str] = {}
    for row in raw.values():
        ticker = row.get("ticker")
        cik = row.get("cik_str")
        if ticker and cik is not None:
            mapping[str(ticker).upper()] = str(int(cik)).zfill(10)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def ticker_to_cik(symbol: str, force_refresh: bool = False) -> str:
    """Resolve ticker -> zero-padded CIK string."""
    mapping = fetch_cik_map(force=force_refresh)
    cik = mapping.get(symbol.upper())
    if not cik:
        raise EdgarError(
            f"Ticker {symbol} not found in SEC company_tickers.json"
        )
    return cik


# ---------------------------------------------------------------------------
# Filing lookup
# ---------------------------------------------------------------------------


def _submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def get_latest_10k_ref(symbol: str) -> FilingRef:
    """Find the most recent 10-K filing metadata for a ticker."""
    cik = ticker_to_cik(symbol)
    r = httpx.get(_submissions_url(cik), headers=BASE_HEADERS, timeout=15.0)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms: list[str] = recent.get("form", [])
    dates: list[str] = recent.get("filingDate", [])
    accessions: list[str] = recent.get("accessionNumber", [])
    primary_docs: list[str] = recent.get("primaryDocument", [])

    for form, date, acc, doc in zip(forms, dates, accessions, primary_docs):
        if form.upper() == "10-K":
            return FilingRef(
                cik=cik,
                symbol=symbol.upper(),
                form=form,
                filing_date=date,
                accession_number=acc,
                primary_document=doc,
            )
    raise EdgarError(f"No 10-K found in recent filings for {symbol}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _rate_limit_sleep() -> None:
    """SEC asks for <= ~10 req/s; we stay under by sleeping 0.2s per call."""
    time.sleep(0.2)


def download_10k_html(ref: FilingRef) -> str:
    """Fetch the primary 10-K document as raw HTML text."""
    _rate_limit_sleep()
    r = httpx.get(ref.primary_url, headers=HTML_HEADERS, timeout=30.0)
    r.raise_for_status()
    return r.text


def cached_10k_html_path(ref: FilingRef) -> Path:
    return CACHE_DIR / f"{ref.symbol}_10K_{ref.filing_date}.html"


def download_10k(symbol: str, use_cache: bool = True) -> tuple[FilingRef, str]:
    """End-to-end: resolve symbol, fetch the latest 10-K HTML, cache on disk.

    Returns (FilingRef, html_text). Subsequent calls within the same
    filing cycle read from the on-disk cache.
    """
    ref = get_latest_10k_ref(symbol)
    cache = cached_10k_html_path(ref)

    if use_cache and cache.exists():
        logger.info("Reading cached 10-K from %s", cache)
        return ref, cache.read_text(encoding="utf-8")

    html = download_10k_html(ref)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(html, encoding="utf-8")
    return ref, html
