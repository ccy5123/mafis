"""Fetch a ticker's latest 10-K from SEC EDGAR, parse its sections, index.

Usage:
    python scripts/index_10k.py NVDA
    python scripts/index_10k.py NVDA --no-cache     # force re-download

First run for a ticker downloads ~10-20 MB of HTML from sec.gov (polite
rate limits respected). Subsequent runs reuse data/edgar_cache/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.rag.edgar import EdgarError, download_10k  # noqa: E402
from wise_investor.rag.index import stats, upsert_10k_sections  # noqa: E402
from wise_investor.rag.sections import extract_sections  # noqa: E402


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol (e.g. NVDA)")
    parser.add_argument(
        "--no-cache", action="store_true", help="Force a fresh EDGAR download."
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    console.rule(f"[bold]Index 10-K for {symbol}[/bold]")

    try:
        ref, html = download_10k(symbol, use_cache=not args.no_cache)
    except EdgarError as e:
        console.print(f"[red]EDGAR error: {e}[/red]")
        return 1

    console.print(f"Filing: {ref.form} {ref.filing_date}  accession={ref.accession_number}")
    console.print(f"HTML: {len(html):,} chars")

    sections = extract_sections(html)
    present = {k: len(v or "") for k, v in sections.as_dict().items()}
    console.print(f"Sections extracted: {present}")

    if not sections.as_dict():
        console.print("[red]No sections extracted — regex may need widening.[/red]")
        return 2

    total = upsert_10k_sections(
        symbol=symbol,
        filing_date=ref.filing_date,
        sections=sections.as_dict(),
    )
    console.print(f"[green]Indexed {total} chunks[/green]")
    console.print(f"[dim]Collection stats: {stats()}[/dim]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
