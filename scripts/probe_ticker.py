"""Quick Finnhub coverage probe for any ticker. Usage:
    python scripts/probe_ticker.py GEV
    python scripts/probe_ticker.py 005930.KS
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.finnhub import (  # noqa: E402
    FinnhubClient,
    FinnhubError,
    derive_ebitda,
    derive_free_cash_flow,
    extract_field,
    total_debt,
)


def dollars(v: float | None) -> str:
    if v is None:
        return "N/A"
    if abs(v) >= 1e9:
        return f"${v/1e9:,.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:,.2f}M"
    return f"${v:,.2f}"


def main(symbol: str) -> int:
    print(f"=== Finnhub probe: {symbol} ===")
    with FinnhubClient() as c:
        # /quote
        try:
            q = c.quote(symbol)
            print(f"quote.c (current price): {q.price}")
        except FinnhubError as e:
            print(f"quote FAILED: {e}")

        # /profile
        try:
            p = c.profile(symbol)
            print(f"profile.name: {p.name}")
            print(f"profile.country: {p.country}")
            print(f"profile.currency: {p.currency}")
            print(f"profile.finnhub_industry: {p.finnhub_industry}")
            print(f"profile.exchange: {p.exchange}")
            print(f"market_cap_usd: {dollars(p.market_cap_usd)}")
            print(f"shares_outstanding: {p.shares_outstanding_abs}")
        except FinnhubError as e:
            print(f"profile FAILED: {e}")

        # /peers
        try:
            peers = c.peers(symbol)
            print(f"peers ({len(peers)}): {peers}")
        except FinnhubError as e:
            print(f"peers FAILED: {e}")

        # /metric
        try:
            m = c.metric(symbol)
            print(f"metric.pe_ttm: {m.metric.pe_ttm}")
            print(f"metric.pe_annual: {m.metric.pe_annual}")
            print(f"metric.ev_ebitda_ttm: {m.metric.ev_ebitda_ttm}")
            print(f"metric.enterprise_value (USD): {dollars(m.metric.enterprise_value_usd)}")
        except FinnhubError as e:
            print(f"metric FAILED: {e}")

        # /financials-reported annual
        try:
            latest = c.latest_annual_financials(symbol)
            if latest is None:
                print("financials-reported: NO ANNUAL FILING FOUND")
            else:
                print(f"latest filing: {latest.form} end_date={latest.end_date} year={latest.year}")
                print(f"line items in report.ic: {len(latest.report.ic)}")
                print(f"line items in report.bs: {len(latest.report.bs)}")
                print(f"line items in report.cf: {len(latest.report.cf)}")
                print(f"revenue: {dollars(extract_field(latest, 'revenue'))}")
                print(f"net_income: {dollars(extract_field(latest, 'net_income'))}")
                print(f"operating_income: {dollars(extract_field(latest, 'operating_income'))}")
                print(f"gross_profit: {dollars(extract_field(latest, 'gross_profit'))}")
                print(f"eps_diluted: {extract_field(latest, 'eps_diluted')}")
                print(f"ebitda (derived): {dollars(derive_ebitda(latest))}")
                print(f"FCF (derived): {dollars(derive_free_cash_flow(latest))}")
                print(f"total_debt: {dollars(total_debt(latest))}")
                print(f"operating_cash_flow: {dollars(extract_field(latest, 'operating_cash_flow'))}")
                print(f"capital_expenditure: {dollars(extract_field(latest, 'capital_expenditure'))}")
        except FinnhubError as e:
            print(f"financials FAILED: {e}")

    return 0


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "GEV"
    sys.exit(main(sym))
