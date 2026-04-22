"""Phase 1A integration smoke — exercise all 6 calculation tools against one symbol.

This is the Phase 1A exit checkpoint: if all six numbers render correctly here
against a real FMP account, the calculation layer is ready to be wired into
Phase 1B's Analyst/Valuer agent prompts.

Run: python scripts/smoke_phase1a.py [TICKER]   (default: AAPL)
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import settings  # noqa: E402
from wise_investor.data.cross_validate import cross_validate_quote  # noqa: E402
from wise_investor.data.finnhub import FinnhubClient as FMPClient  # noqa: E402
from wise_investor.tools.dcf import reverse_dcf  # noqa: E402
from wise_investor.tools.valuation import (  # noqa: E402
    calculate_ev_ebitda,
    calculate_per,
    get_peer_multiples,
)
from wise_investor.tools.verify import verify_number  # noqa: E402


console = Console()


def fmt(v: float | None, digits: int = 2) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.{digits}f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.{digits}f}M"
    return f"{v:,.{digits}f}"


def fmt_pct(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def run(symbol: str) -> int:
    if not settings.finnhub_api_key or settings.finnhub_api_key == "your_finnhub_api_key_here":
        console.print("[red]FINNHUB_API_KEY not set in .env[/red]")
        return 1

    console.rule(f"[bold]Phase 1A smoke — {symbol}[/bold]")

    with FMPClient() as fmp:
        # 1. Cross-validate FMP vs yfinance
        console.print("\n[bold]1. FMP ↔ yfinance cross-validation[/bold]")
        xv = cross_validate_quote(symbol, fmp=fmp)
        xv_table = Table(show_header=True)
        xv_table.add_column("Field")
        xv_table.add_column("FMP", justify="right")
        xv_table.add_column("yfinance", justify="right")
        xv_table.add_column("Δ %", justify="right")
        xv_table.add_column("Flag")
        for c in xv.comparisons:
            flag = (
                "[green]OK[/green]"
                if c.within_threshold
                else "[red]DIVERGES[/red]"
                if c.within_threshold is False
                else "[yellow]—[/yellow]"
            )
            xv_table.add_row(
                c.field,
                fmt(c.fmp_value),
                fmt(c.yf_value),
                "—" if c.diff_pct is None else f"{c.diff_pct:.2f}%",
                flag,
            )
        console.print(xv_table)
        if xv.any_flagged:
            console.print(
                "[yellow]⚠ At least one field exceeds threshold — investigate before trusting report.[/yellow]"
            )

        # 2. PER
        console.print("\n[bold]2. PER (Python-computed vs FMP-reported)[/bold]")
        per_res = calculate_per(symbol, client=fmp)
        per_table = Table(show_header=False, box=None)
        per_table.add_row("as_of", per_res.as_of or "—")
        per_table.add_row("price (input)", fmt(per_res.inputs.get("price")))
        per_table.add_row("EPS diluted (input)", fmt(per_res.inputs.get("eps_diluted_latest_annual")))
        per_table.add_row("[bold]PER (our calc)[/bold]", fmt(per_res.computed))
        per_table.add_row("PER (FMP reported)", fmt(per_res.fmp_reported))
        per_table.add_row(
            "diff vs FMP",
            "—" if per_res.diff_pct_vs_fmp is None else f"{per_res.diff_pct_vs_fmp:.2f}%",
        )
        if per_res.warnings:
            per_table.add_row("warnings", "; ".join(per_res.warnings))
        console.print(per_table)

        # 3. EV/EBITDA
        console.print("\n[bold]3. EV/EBITDA[/bold]")
        ev_res = calculate_ev_ebitda(symbol, client=fmp)
        ev_table = Table(show_header=False, box=None)
        ev_table.add_row("as_of", ev_res.as_of or "—")
        ev_table.add_row("EV (input)", fmt(ev_res.inputs.get("enterprise_value")))
        ev_table.add_row("EBITDA (input)", fmt(ev_res.inputs.get("ebitda_latest_annual")))
        ev_table.add_row("[bold]EV/EBITDA (our calc)[/bold]", fmt(ev_res.computed))
        ev_table.add_row("EV/EBITDA (FMP reported)", fmt(ev_res.fmp_reported))
        ev_table.add_row(
            "diff vs FMP",
            "—" if ev_res.diff_pct_vs_fmp is None else f"{ev_res.diff_pct_vs_fmp:.2f}%",
        )
        if ev_res.warnings:
            ev_table.add_row("warnings", "; ".join(ev_res.warnings))
        console.print(ev_table)

        # 4. Peer multiples
        console.print("\n[bold]4. Peer multiples table[/bold]")
        peers = get_peer_multiples(symbol, client=fmp, max_peers=5)
        peer_table = Table(show_header=True)
        peer_table.add_column("Symbol")
        peer_table.add_column("Name")
        peer_table.add_column("Market cap", justify="right")
        peer_table.add_column("PER", justify="right")
        peer_table.add_column("EV/EBITDA", justify="right")
        for row in peers.rows:
            peer_table.add_row(
                row.symbol,
                (row.name or "")[:28],
                fmt(row.market_cap),
                fmt(row.per),
                fmt(row.ev_ebitda),
            )
        console.print(peer_table)

        # 5. Reverse DCF
        console.print("\n[bold]5. Reverse DCF — implied growth[/bold]")
        dcf_res = reverse_dcf(symbol, client=fmp)
        dcf_table = Table(show_header=False, box=None)
        dcf_table.add_row("as_of", dcf_res.as_of or "—")
        dcf_table.add_row("market cap", fmt(dcf_res.current_market_cap))
        dcf_table.add_row("FCF (input)", fmt(dcf_res.inputs.get("fcf_latest_annual")))
        dcf_table.add_row("FCF source", str(dcf_res.inputs.get("fcf_source", "—")))
        dcf_table.add_row("discount rate", fmt_pct(dcf_res.inputs.get("discount_rate")))
        dcf_table.add_row("terminal growth", fmt_pct(dcf_res.inputs.get("terminal_growth")))
        dcf_table.add_row("high-growth years", str(dcf_res.inputs.get("high_growth_years")))
        dcf_table.add_row(
            "[bold]implied growth[/bold]",
            fmt_pct(dcf_res.implied_growth_rate),
        )
        if dcf_res.warnings:
            dcf_table.add_row("warnings", "; ".join(dcf_res.warnings))
        console.print(dcf_table)

        # 6. verify_number round-trip (Bull claims X, Skeptic verifies)
        console.print("\n[bold]6. verify_number round-trip[/bold]")
        console.print(
            "[dim]Simulating: Bull cites our computed PER. Skeptic verifies "
            "against the same source — they should agree within 0.5%.[/dim]"
        )
        verify_rows = []
        if per_res.computed is not None:
            v_per = verify_number(
                claim=per_res.computed, field="per", symbol=symbol, client=fmp, tolerance_pct=0.5
            )
            verify_rows.append(("PER", v_per))
        else:
            console.print("[yellow]PER unavailable → skipping verify_number(per)[/yellow]")

        # Also verify a raw FMP field (revenue)
        rev_claim = 391_000_000_000.0  # approx AAPL FY24; will likely mismatch for other tickers
        v_rev = verify_number(
            claim=rev_claim,
            field="revenue",
            symbol=symbol,
            client=fmp,
            tolerance_pct=1.0,
        )
        verify_rows.append(("revenue (claim=$391B)", v_rev))

        v_table = Table(show_header=True)
        v_table.add_column("What")
        v_table.add_column("Claim", justify="right")
        v_table.add_column("Source", justify="right")
        v_table.add_column("Δ %", justify="right")
        v_table.add_column("Match")
        for label, v in verify_rows:
            match = (
                "[green]✓[/green]"
                if v.matches is True
                else "[red]✗[/red]"
                if v.matches is False
                else "[yellow]?[/yellow]"
            )
            v_table.add_row(
                label,
                fmt(v.claim),
                fmt(v.source_value),
                "—" if v.diff_pct is None else f"{v.diff_pct:.2f}%",
                match,
            )
        console.print(v_table)

    console.print(
        Panel(
            "Phase 1A complete. All 6 calculation tools produced output above.\n"
            "Next: Phase 1B — wire these as CrewAI Tools into the Analyst agent.",
            style="green",
        )
    )
    return 0


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    sys.exit(run(ticker))
