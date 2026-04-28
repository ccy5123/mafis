"""Live screening — Stage 2 prefilter (+ optional Stage 3) on real-time fundamentals.

Unlike `scripts/run_prefilter.py` (which loads hand-curated JSON
fixtures for calibration) and `scripts/run_back_validation.py` (which
runs the rubric against historical fundamentals at a calibration date),
this CLI fetches **current** fundamentals from Finnhub and runs the
constitution v2.0 Stage 2 → Stage 3 pipeline against the live state.

Usage:
    python scripts/run_screening.py NVDA
    python scripts/run_screening.py NVDA --with-stage3
    python scripts/run_screening.py --universe data/calibration/manifest.yaml
    python scripts/run_screening.py --universe data/calibration/manifest.yaml --json

Output:
  - Default: rich table with axis verdicts + decision per ticker.
  - --json: full prefilter (and optionally stage3) result as JSON.

Constitution alignment: this is the live runner — Commitments 1-3 apply
unchanged. Tickers come from the manifest (no user-preference channel)
and the rubric never quietly upgrades a data-missing ticker (live mode
exposes top5_customer_share=None and segments=single-segment-default,
both of which the prefilter routes through NEED_LLM, not PASS).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.screening.calibration_ledger import load_manifest  # noqa: E402
from wise_investor.screening.live_adapter import fetch_live_fundamentals  # noqa: E402
from wise_investor.screening.prefilter import evaluate_ticker  # noqa: E402

console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "symbol",
        nargs="?",
        help="Single ticker to screen (mutually exclusive with --universe).",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        help=(
            "Path to a manifest YAML to screen as a batch. "
            "Mutually exclusive with the positional symbol."
        ),
    )
    parser.add_argument(
        "--with-stage3",
        action="store_true",
        help="After Stage 2, run Stage 3 LLM screening on advancing tickers.",
    )
    parser.add_argument(
        "--with-peers",
        action="store_true",
        help=(
            "Compute industry_roic_3y_median and industry_gross_margin_3y_std "
            "via Finnhub peer aggregation before Stage 2. Adds ~5 API calls "
            "per US ticker (cached 24h on disk). Korean tickers skip — "
            "Finnhub peers are sparse for KRX."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of formatted tables.",
    )
    args = parser.parse_args()

    if args.universe and args.symbol:
        console.print("[red]--universe and a symbol are mutually exclusive.[/red]")
        return 2
    if not args.universe and not args.symbol:
        console.print("[red]Provide either a symbol or --universe.[/red]")
        return 2

    if args.universe:
        if not args.universe.exists():
            console.print(f"[red]Manifest not found: {args.universe}[/red]")
            return 1
        manifest = load_manifest(args.universe)
        symbols = manifest.symbols
    else:
        symbols = [args.symbol]

    results = _run(symbols, with_stage3=args.with_stage3, with_peers=args.with_peers)

    if args.json:
        print(json.dumps(_to_json(results), indent=2, default=str))
        return 0

    _print_table(results, with_stage3=args.with_stage3)
    return 0


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run(
    symbols: list[str],
    *,
    with_stage3: bool,
    with_peers: bool = False,
) -> list[dict]:
    """Screen each symbol; collect (symbol, prefilter, stage3, error) rows."""
    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as prog:
        task = prog.add_task("Screening...", total=len(symbols))
        for sym in symbols:
            prog.update(task, description=f"Fetching {sym}...")
            row: dict = {
                "symbol": sym,
                "prefilter": None,
                "stage3": None,
                "error": None,
            }
            try:
                # --with-peers: compute industry aggregates first (US tickers
                # only — Finnhub peers for KRX are sparse, so KR tickers
                # silently skip the aggregator and proceed without peer
                # comparison fields).
                aggs = None
                if with_peers:
                    from wise_investor.screening.live_adapter_kr import (
                        is_korean_symbol,
                    )
                    if not is_korean_symbol(sym):
                        prog.update(task, description=f"Aggregating peers for {sym}...")
                        from wise_investor.screening.peer_aggregator import (
                            compute_industry_aggregates,
                        )
                        peer_result = compute_industry_aggregates(sym)
                        aggs = peer_result.industry_aggregates

                prog.update(task, description=f"Fetching {sym}...")
                funds = fetch_live_fundamentals(
                    sym,
                    industry_aggregates=aggs,
                )
                primary = (
                    funds.segments_history[-1] if funds.segments_history else None
                )
                row["prefilter"] = evaluate_ticker(funds, primary)
                if (
                    with_stage3
                    and row["prefilter"].hierarchy_decision == "ADVANCE_TO_STAGE_3"
                ):
                    # Lazy import: only load the LLM stack when we'll actually call it.
                    from wise_investor.screening.llm_screening import screen_ticker
                    row["stage3"] = screen_ticker(funds, row["prefilter"])
            except Exception as e:
                row["error"] = str(e)
            results.append(row)
            prog.advance(task)
    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(results: list[dict], *, with_stage3: bool) -> None:
    table = Table(title="Live screening", show_lines=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Stage 2")
    table.add_column("Moat")
    table.add_column("Frontier")
    table.add_column("Bottleneck")
    if with_stage3:
        table.add_column("Stage 3")

    for row in results:
        sym = row["symbol"]
        if row["error"] is not None:
            err_msg = row["error"][:50]
            cells = [sym, f"[red]ERROR: {err_msg}[/red]", "—", "—", "—"]
            if with_stage3:
                cells.append("—")
            table.add_row(*cells)
            continue

        p = row["prefilter"]
        decision = p.hierarchy_decision
        d_color = "green" if decision == "ADVANCE_TO_STAGE_3" else "red"
        d_short = (
            decision.replace("ADVANCE_TO_", "→")
            if decision != "REJECT"
            else "REJECT"
        )

        cells = [
            sym,
            f"[{d_color}]{d_short}[/{d_color}]",
            _verdict(p.moat.verdict),
            _verdict(p.new_frontier.verdict),
            _verdict(p.bottleneck.verdict),
        ]
        if with_stage3:
            s3 = row["stage3"]
            if s3 is None:
                cells.append("[dim]—[/dim]")
            else:
                s3d = s3.hierarchy_decision
                color = "green" if s3d == "ADVANCE_TO_STAGE_4" else "red"
                short = (
                    s3d.replace("ADVANCE_TO_", "→")
                    if s3d != "REJECT"
                    else "REJECT"
                )
                cells.append(f"[{color}]{short}[/{color}]")
        table.add_row(*cells)

    console.print(table)


def _verdict(v: str) -> str:
    color = {"PASS": "green", "FAIL": "red", "NEED_LLM": "yellow"}.get(v, "white")
    return f"[{color}]{v}[/{color}]"


def _to_json(results: list[dict]) -> list[dict]:
    """Convert dataclass rows to JSON-friendly dicts."""
    out = []
    for row in results:
        d: dict = {"symbol": row["symbol"], "error": row["error"]}
        if row["prefilter"] is not None:
            d["prefilter"] = asdict(row["prefilter"])
        if row["stage3"] is not None:
            d["stage3"] = asdict(row["stage3"])
        out.append(d)
    return out


if __name__ == "__main__":
    sys.exit(main())
