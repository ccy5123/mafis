"""Bulk-index 10-K filings for the calibration manifest universe.

Usage:
    python scripts/index_universe.py
    python scripts/index_universe.py --no-cache       # force re-download
    python scripts/index_universe.py --limit 5        # debug
    python scripts/index_universe.py --skip-existing  # don't re-index symbols
                                                       # already in the index

Background (P1a 2026-04): the Stage 2 prefilter consumes RAG signals
(`top5_customer_share`, `diversification_attempt_signals`) extracted
from indexed 10-Ks via `screening.rag_signals.extract_rag_signals`.
Those signals are populated only when a caller explicitly passes the
`rag_signals=` argument to the live/historical adapter — the
calibration runner (`run_back_validation.py`) currently does NOT, so
absent pre-indexing AND absent caller wiring, every ticker's
bottleneck axis routes to NEED_LLM by default.

This script is the manual pre-indexing step for the manifest universe.
KR tickers are skipped (DART filings aren't indexed in the same
ChromaDB collection) and SPY is skipped (ETF self-exclusion). Foreign
issuers (US-ADR market) like TSM/ASML/NVO file 20-F not 10-K — EDGAR
download will likely fail for these; failures are logged and the
script proceeds.

Cost: one-shot ~3-5 min for 22 US/US-ADR tickers, ~30-50MB ChromaDB
disk, free local embedding (MiniLM-L6-v2). Re-running the script is
idempotent — the same (symbol, filing_date) chunks overwrite cleanly.

Constitution alignment: this is operational infrastructure, not
constitutional policy. The §15 bottleneck axis still routes to
NEED_LLM when top5_customer_share is None, but with this index
populated the prefilter has the data to evaluate path 1-B (≥40% top-5
share) before delegating to Stage 3.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.rag.edgar import EdgarError, download_10k  # noqa: E402
from wise_investor.rag.index import stats, upsert_10k_sections  # noqa: E402
from wise_investor.rag.sections import extract_sections  # noqa: E402

console = Console()

MANIFEST_PATH = REPO_ROOT / "data" / "calibration" / "manifest.yaml"

# Markets / symbols to skip — pre-indexing only covers SEC EDGAR 10-Ks.
SKIP_MARKETS = {"KR"}
SKIP_SYMBOLS = {"SPY"}

# Polite gap between EDGAR fetches; sec.gov is lenient but explicit
# spacing avoids accidental rate-limit trips during a 22-symbol batch.
EDGAR_SLEEP_SEC = 0.4


def _load_manifest() -> list[dict]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("tickers", [])


def _index_one(symbol: str, use_cache: bool) -> tuple[str, int]:
    """Download → extract → upsert one ticker. Returns (status, chunks)."""
    try:
        ref, html = download_10k(symbol, use_cache=use_cache)
    except EdgarError as e:
        return (f"edgar_error: {e}", 0)
    except Exception as e:  # network / unexpected
        return (f"download_error: {e}", 0)

    try:
        sections = extract_sections(html)
        section_dict = sections.as_dict()
    except Exception as e:
        return (f"section_extract_error: {e}", 0)

    if not section_dict:
        return ("empty_sections", 0)

    try:
        total = upsert_10k_sections(
            symbol=symbol,
            filing_date=ref.filing_date,
            sections=section_dict,
        )
    except Exception as e:
        return (f"upsert_error: {e}", 0)

    return (f"ok ({ref.form} {ref.filing_date})", total)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh EDGAR downloads (slower, but picks up new filings).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Index only the first N matching tickers (debug).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip symbols that already have indexed chunks. Useful for "
            "incremental top-ups after the initial bulk run."
        ),
    )
    args = parser.parse_args()

    tickers = _load_manifest()
    targets = [
        t for t in tickers
        if t.get("market") not in SKIP_MARKETS
        and t.get("symbol") not in SKIP_SYMBOLS
    ]
    if args.limit:
        targets = targets[: args.limit]

    console.rule(
        f"[bold]Index universe — {len(targets)} tickers "
        f"(skipped {len(tickers) - len(targets)})[/bold]"
    )

    initial = stats()
    console.print(f"[dim]Initial collection: {initial}[/dim]")

    existing_symbols: set[str] = set()
    if args.skip_existing:
        # Chroma's `stats()` doesn't enumerate symbols, so we use a
        # cheap empty-section probe — anything with chunks shows up.
        from wise_investor.rag.index import _get_collection
        col = _get_collection()
        all_meta = col.get(include=["metadatas"])
        existing_symbols = {
            str(m.get("symbol", "")).upper()
            for m in (all_meta.get("metadatas") or [])
            if m
        }
        console.print(
            f"[dim]Skip-existing on: {len(existing_symbols)} symbols already indexed[/dim]"
        )

    successes = 0
    failures: list[tuple[str, str]] = []
    total_chunks = 0

    for i, t in enumerate(targets, 1):
        symbol = t["symbol"].upper()
        market = t.get("market", "?")

        if symbol in existing_symbols:
            console.print(f"[{i}/{len(targets)}] {symbol} ({market}) — already indexed, skip")
            continue

        console.print(f"[{i}/{len(targets)}] {symbol} ({market}) ... ", end="")
        status, chunks = _index_one(symbol, use_cache=not args.no_cache)
        if chunks > 0:
            console.print(f"[green]{status} → {chunks} chunks[/green]")
            successes += 1
            total_chunks += chunks
        else:
            console.print(f"[yellow]{status}[/yellow]")
            failures.append((symbol, status))

        time.sleep(EDGAR_SLEEP_SEC)

    final = stats()
    console.rule("[bold]Summary[/bold]")
    console.print(f"Successes: {successes}/{len(targets)}")
    console.print(f"Total chunks indexed this run: {total_chunks}")
    console.print(f"Final collection: {final}")
    if failures:
        console.print(f"\n[yellow]Failures ({len(failures)}):[/yellow]")
        for sym, why in failures:
            console.print(f"  {sym}: {why}")
        console.print(
            "\n[dim]Note: foreign issuers (US-ADR like TSM/ASML/NVO) file 20-F, "
            "not 10-K — EDGAR download is expected to fail for these until an "
            "ADR fallback (P1b) lands.[/dim]"
        )
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
