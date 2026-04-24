"""Ad-hoc OpenDART inspector.

Usage:
    python scripts/probe_dart.py 005930                  # Samsung by stock code
    python scripts/probe_dart.py 00126380 --corp-code    # by corp_code directly
    python scripts/probe_dart.py 005930 --year 2024 --report q3
    python scripts/probe_dart.py 005930 --fs-div OFS     # separate statements

Prints:
  - Corp mapping row (name + stock_code + corp_code + last modify date)
  - Headline financials extracted from the /fnlttSinglAcntAll.json payload
    (Revenue / Operating Income / Net Income / Total Assets / ...)

Requires DART_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.data.dart import (  # noqa: E402
    DartClient,
    DartError,
    FS_CONSOLIDATED,
    FS_SEPARATE,
    REPORT_ANNUAL,
    REPORT_H1,
    REPORT_Q1,
    REPORT_Q3,
    extract_account_value,
)


console = Console()


_REPORT_CHOICES = {
    "annual": REPORT_ANNUAL,
    "h1": REPORT_H1,
    "q1": REPORT_Q1,
    "q3": REPORT_Q3,
}

# Korean account names to pull out for the headline summary. DART
# sometimes emits slightly different display names across filings; we
# probe both the most common variants.
_HEADLINE_ACCOUNTS: list[tuple[str, list[str]]] = [
    ("Revenue", ["매출액", "수익(매출액)", "영업수익"]),
    ("Operating Income", ["영업이익", "영업이익(손실)"]),
    ("Net Income", ["당기순이익", "당기순이익(손실)"]),
    ("Total Assets", ["자산총계"]),
    ("Total Liabilities", ["부채총계"]),
    ("Total Equity", ["자본총계"]),
    ("Cash & Equivalents", ["현금및현금성자산"]),
]


def _fmt_krw(value: float | None) -> str:
    if value is None:
        return "—"
    # DART reports in units of 1 KRW (not millions). For display,
    # render as trillions / billions when large.
    abs_v = abs(value)
    if abs_v >= 1e12:
        return f"₩{value/1e12:,.2f}T"
    if abs_v >= 1e8:
        return f"₩{value/1e8:,.2f}억"  # 억 = 100M
    return f"₩{value:,.0f}"


def run(args: argparse.Namespace) -> int:
    try:
        client = DartClient()
    except DartError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    with client:
        # Resolve to corp_code.
        if args.corp_code:
            corp_code = args.symbol.zfill(8)
            mapping_name = None
        else:
            console.print(f"Resolving stock code [cyan]{args.symbol}[/cyan] → corp_code...")
            corp_code = client.corp_code_from_stock_code(args.symbol)
            if corp_code is None:
                console.print(
                    f"[yellow]Stock code {args.symbol} not found in DART. Pass a "
                    f"corp_code directly with --corp-code.[/yellow]"
                )
                return 2
            # Grab the name for display.
            mappings = client.load_corp_mapping()
            entry = next(
                (m for m in mappings if m.stock_code == args.symbol.zfill(6)), None
            )
            mapping_name = entry.corp_name if entry else None

        console.print(
            f"[bold]DART probe[/bold]: "
            f"{mapping_name or '(direct corp_code)'} "
            f"([cyan]{corp_code}[/cyan])  year={args.year}  "
            f"report={args.report}  fs_div={args.fs_div}"
        )

        report_code = _REPORT_CHOICES[args.report]
        fs_div = FS_SEPARATE if args.fs_div.upper() == "OFS" else FS_CONSOLIDATED

        resp = client.financials(
            corp_code=corp_code,
            year=args.year,
            reprt_code=report_code,
            fs_div=fs_div,
        )

    if not resp.ok:
        console.print(f"[red]DART error {resp.status}: {resp.message}[/red]")
        return 3

    console.print(f"[dim]{len(resp.rows)} row(s) returned.[/dim]")

    # Headline summary.
    currency = resp.rows[0].currency if resp.rows else "KRW"
    table = Table(title=f"Headline financials ({currency})")
    table.add_column("Metric", style="cyan")
    table.add_column("Current period", justify="right")
    table.add_column("Raw value", justify="right", style="dim")
    for label, candidate_names in _HEADLINE_ACCOUNTS:
        value = None
        for name in candidate_names:
            value = extract_account_value(resp, account_nm=name)
            if value is not None:
                break
        table.add_row(label, _fmt_krw(value), f"{value or '':,.0f}" if value else "—")
    console.print(table)

    # Raw dump option.
    if args.dump:
        console.rule("[bold]All rows[/bold]")
        for row in resp.rows:
            console.print(
                f"  [dim]{row.sj_div}[/dim] "
                f"[cyan]{row.account_nm}[/cyan] "
                f"(id={row.account_id or '-'}): "
                f"{row.thstrm_amount or '-'} {row.currency or ''}"
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbol", help="6-digit stock code (e.g. 005930) or 8-digit corp code"
    )
    parser.add_argument(
        "--corp-code",
        action="store_true",
        help="Interpret `symbol` as an 8-digit corp_code directly (skip lookup)",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument(
        "--report", choices=list(_REPORT_CHOICES.keys()), default="annual"
    )
    parser.add_argument(
        "--fs-div", choices=["CFS", "OFS"], default="CFS",
        help="Consolidated (CFS) or separate (OFS)",
    )
    parser.add_argument(
        "--dump", action="store_true", help="Print every account row"
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
