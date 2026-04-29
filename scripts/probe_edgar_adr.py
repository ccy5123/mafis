"""P1b pre-flight: probe EDGAR coverage for the manifest's US-ADR tickers.

For each of TSM, ASML, NVO:
  1. Resolve ticker -> CIK via company_tickers.json (cached helper)
  2. Hit submissions API to list recent forms (look for 20-F vs 10-K)
  3. Hit companyfacts API and report concept namespaces present
     (us-gaap vs ifrs-full vs dei) and years covered for the 5
     concepts we care about (Revenue, OperatingIncome, GrossProfit,
     Assets, Cash).

Output is printed; nothing persisted. Goal: verify before writing
real adapter code that EDGAR actually has data for these symbols.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.rag.edgar import (  # noqa: E402
    BASE_HEADERS,
    EdgarError,
    _submissions_url,
    fetch_cik_map,
    ticker_to_cik,
)

ADR_SYMBOLS = ["TSM", "ASML", "NVO"]
# 5 fundamentals concepts we need for the IC formula + GM
CONCEPTS_OF_INTEREST = {
    "Revenue / Revenues": ["us-gaap:Revenues", "ifrs-full:Revenue"],
    "Gross profit": ["us-gaap:GrossProfit", "ifrs-full:GrossProfit"],
    "Operating income": [
        "us-gaap:OperatingIncomeLoss",
        "ifrs-full:ProfitLossFromOperatingActivities",
    ],
    "Total assets": ["us-gaap:Assets", "ifrs-full:Assets"],
    "Cash & equivalents": [
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "ifrs-full:CashAndCashEquivalents",
    ],
}


def _companyfacts_url(cik: str) -> str:
    return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _get_json(url: str) -> Any:
    r = httpx.get(url, headers=BASE_HEADERS, timeout=30.0)
    r.raise_for_status()
    return r.json()


def _years_for_concept(facts: dict, namespace: str, concept: str) -> list[int]:
    """Return distinct fiscal years where this concept has at least one
    USD value (FY periods only, ignoring quarterlies)."""
    block = (facts.get("facts", {}).get(namespace, {}) or {}).get(concept)
    if not block:
        return []
    units = block.get("units", {})
    # Pull from any unit (USD, USD/share, etc.). Most fundamentals are USD.
    years: set[int] = set()
    for unit_name, items in units.items():
        for item in items:
            fp = item.get("fp")  # Q1/Q2/Q3/FY
            fy = item.get("fy")
            if fp == "FY" and fy is not None:
                years.add(int(fy))
    return sorted(years)


def main() -> int:
    # Warm up the CIK cache (one network call shared across symbols).
    cik_map = fetch_cik_map()
    print(f"company_tickers.json has {len(cik_map):,} entries\n")

    for symbol in ADR_SYMBOLS:
        print(f"=== {symbol} ===")
        try:
            cik = ticker_to_cik(symbol)
        except EdgarError as e:
            print(f"  CIK lookup failed: {e}\n")
            continue
        print(f"  CIK: {cik}")

        # Submissions: which forms?
        try:
            subs = _get_json(_submissions_url(cik))
        except Exception as e:
            print(f"  submissions failed: {e}\n")
            continue

        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        form_counts: dict[str, int] = {}
        for f in forms:
            form_counts[f] = form_counts.get(f, 0) + 1
        print(f"  Recent form counts (top 8):")
        for f, n in sorted(form_counts.items(), key=lambda kv: -kv[1])[:8]:
            print(f"    {f}: {n}")
        # Most recent annual
        annuals = [
            (date, form)
            for date, form in zip(dates, forms)
            if form.upper() in ("10-K", "20-F")
        ]
        if annuals:
            print(f"  Most recent annual filings (up to 3):")
            for date, form in annuals[:3]:
                print(f"    {form} {date}")

        # Company facts: what namespaces?
        try:
            facts = _get_json(_companyfacts_url(cik))
        except Exception as e:
            print(f"  companyfacts failed: {e}\n")
            continue

        ns_map = facts.get("facts", {})
        print(f"  facts namespaces: {sorted(ns_map.keys())}")
        for ns, block in ns_map.items():
            print(f"    {ns}: {len(block):,} concepts")

        # Concept availability matrix
        print(f"  Concept availability (FY years where data exists):")
        for label, candidates in CONCEPTS_OF_INTEREST.items():
            best_ns: str | None = None
            best_years: list[int] = []
            for cand in candidates:
                ns, concept = cand.split(":", 1)
                yrs = _years_for_concept(facts, ns, concept)
                if yrs and (not best_years or len(yrs) > len(best_years)):
                    best_ns = cand
                    best_years = yrs
            if best_ns:
                year_range = (
                    f"{best_years[0]}–{best_years[-1]}, n={len(best_years)}"
                    if best_years
                    else "none"
                )
                # Highlight whether 2018 (calibration year) is in there.
                has_2018 = 2018 in best_years
                marker = "✓" if has_2018 else "✗"
                print(f"    {label}: {best_ns} [{year_range}] 2018:{marker}")
            else:
                print(f"    {label}: NOT FOUND in any candidate namespace")

        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
