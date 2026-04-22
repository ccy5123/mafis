"""Smoke test the FinnhubClient end-to-end for AAPL.

Verifies: quote, profile, metric, financials (concept extraction), peers.
Prints every derived value so we can eyeball it against reality before wiring
the client into the calculation tools.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.finnhub import (  # noqa: E402
    DOLLAR_MILLIONS,
    FinnhubClient,
    derive_ebitda,
    derive_free_cash_flow,
    extract_field,
    total_debt,
)


console = Console()


def fmt_b(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"${v / 1e9:,.2f}B"


def main() -> int:
    symbol = "AAPL"
    console.rule(f"[bold]FinnhubClient smoke — {symbol}[/bold]")

    with FinnhubClient() as c:
        q = c.quote(symbol)
        p = c.profile(symbol)
        m = c.metric(symbol)
        fin = c.latest_annual_financials(symbol)
        peers = c.peers(symbol)

    # ---- quote + profile
    t = Table(title="Market data")
    t.add_column("Field")
    t.add_column("Value")
    t.add_row("quote.c (current price)", f"${q.price:,.2f}")
    t.add_row("profile.name", p.name or "—")
    t.add_row("profile.finnhub_industry", p.finnhub_industry or "—")
    t.add_row("market cap (from profile, USD)", fmt_b(p.market_cap_usd))
    t.add_row("shares outstanding", f"{(p.shares_outstanding_abs or 0):,.0f}" if p.shares_outstanding_abs else "N/A")
    console.print(t)

    # ---- metric
    mt = Table(title="Pre-computed metrics (/stock/metric)")
    mt.add_column("Metric")
    mt.add_column("Value")
    mt.add_row("PE Annual", str(m.metric.pe_annual))
    mt.add_row("PE TTM", str(m.metric.pe_ttm))
    mt.add_row("Forward PE", str(m.metric.forward_pe))
    mt.add_row("EV/EBITDA TTM", str(m.metric.ev_ebitda_ttm))
    mt.add_row("EV/Revenue TTM", str(m.metric.ev_revenue_ttm))
    mt.add_row("PEG TTM", str(m.metric.peg_ttm))
    mt.add_row("Enterprise Value (USD)", fmt_b(m.metric.enterprise_value_usd))
    mt.add_row("Revenue growth TTM YoY (%)", str(m.metric.revenue_growth_ttm_yoy))
    mt.add_row("Operating margin TTM (%)", str(m.metric.operating_margin_ttm))
    mt.add_row("Total debt / total equity (annual)", str(m.metric.total_debt_to_total_equity_annual))
    console.print(mt)

    # ---- financials-reported concept extraction
    ft = Table(title="Latest annual financials (XBRL concept extraction)")
    ft.add_column("Logical field")
    ft.add_column("Extracted value")
    if fin is None:
        console.print("[red]No annual financials available[/red]")
    else:
        ft.add_row("form / end_date / year", f"{fin.form} / {fin.end_date} / {fin.year}")
        for field in [
            "revenue", "gross_profit", "operating_income", "net_income",
            "eps_diluted", "depreciation_and_amortization",
            "total_assets", "total_stockholders_equity",
            "cash_and_cash_equivalents", "long_term_debt", "short_term_debt",
            "operating_cash_flow", "capital_expenditure",
        ]:
            v = extract_field(fin, field)
            if v is None:
                display = "N/A"
            elif abs(v) >= 1e6:
                display = fmt_b(v)
            else:
                display = f"{v:,.4f}"
            ft.add_row(field, display)

        ft.add_row("[bold]derived EBITDA[/bold]", fmt_b(derive_ebitda(fin)))
        ft.add_row("[bold]derived FCF[/bold]", fmt_b(derive_free_cash_flow(fin)))
        ft.add_row("[bold]total_debt (lt+st)[/bold]", fmt_b(total_debt(fin)))
    console.print(ft)

    # ---- peers
    console.print(f"\n[bold]Peers[/bold]: {peers}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
