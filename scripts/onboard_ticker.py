"""End-to-end ticker onboarding — one command, zero manual value chain.

Usage:
    python scripts/onboard_ticker.py NVDA --tier 1
    python scripts/onboard_ticker.py AMD --tier 2 --notes "GPU peer of NVDA"
    python scripts/onboard_ticker.py ASML --tier 2 --skip-llm

What it does:
  1. Pulls Finnhub profile + peers for the symbol.
  2. Ensures the latest 10-K is downloaded from SEC EDGAR and indexed
     into ChromaDB (one-time network fetch per symbol).
  3. Runs a few RAG queries to pull Business / Risk Factors / Moat
     excerpts.
  4. Fetches a geopolitical snapshot (GDELT + Google News).
  5. Calls Qwen 2.5 7B with a structured template to produce a first-
     draft value chain brief.
  6. Saves the draft as `docs/value_chains/<SYMBOL>.draft.md`.
  7. Adds `<SYMBOL>` to `config/tickers.yaml` under the chosen tier.

Post-conditions:
  - The `.draft.md` file is NOT picked up by the crew. A human reviews
    it (2-3 minutes, Vulnerable links especially) and renames to
    `<SYMBOL>.md` to activate.
  - `config/tickers.yaml` entry activates the daily sweep under the
    assigned tier.

Runtime on first-time symbol: ~3-5 min (SEC download + ChromaDB
embed + one Qwen call). Cached subsequent runs: under 30 s.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.config import PROJECT_ROOT  # noqa: E402
from wise_investor.onboarding.brief_generator import (  # noqa: E402
    gather_raw_material,
    generate_value_chain_draft,
)
from wise_investor.onboarding.tickers_yaml import (  # noqa: E402
    RegistryError,
    add_ticker_to_registry,
    ticker_in_registry,
    load_registry_yaml,
)


console = Console()

VALUE_CHAINS_DIR = PROJECT_ROOT / "docs" / "value_chains"
TICKERS_YAML = PROJECT_ROOT / "config" / "tickers.yaml"


def run(args: argparse.Namespace) -> int:
    symbol = args.symbol.upper()
    draft_path = VALUE_CHAINS_DIR / f"{symbol}.draft.md"
    final_path = VALUE_CHAINS_DIR / f"{symbol}.md"

    # --- Guard: do not clobber an existing approved brief unless asked.
    if final_path.exists() and not args.force:
        console.print(
            f"[yellow]An approved brief already exists at {final_path}.[/yellow] "
            f"Use --force to overwrite, or delete the file if you want to "
            f"regenerate."
        )
        return 1
    if draft_path.exists() and not args.force:
        console.print(
            f"[yellow]A draft already exists at {draft_path}. Review and "
            f"rename, or rerun with --force.[/yellow]"
        )
        return 1

    # --- Registry guard: abort on tier conflict unless overwrite.
    registry = load_registry_yaml(TICKERS_YAML)
    existing_tier = ticker_in_registry(registry, symbol)
    if existing_tier is not None and existing_tier != args.tier and not args.overwrite_tier:
        console.print(
            f"[yellow]{symbol} is already registered in tier_{existing_tier}. "
            f"Pass --overwrite-tier to move it to tier_{args.tier}.[/yellow]"
        )
        return 1

    # --- Gather raw material.
    console.rule(f"[bold]Onboarding {symbol}[/bold]")
    console.print("Gathering raw material (Finnhub + 10-K + geopolitics)...")
    raw = gather_raw_material(symbol)
    console.print(
        f"  company_name: [cyan]{raw.company_name or '(none)'}[/cyan]\n"
        f"  industry:     [cyan]{raw.industry or '(none)'}[/cyan]\n"
        f"  peers:        [cyan]{', '.join(raw.peers) if raw.peers else '(none)'}[/cyan]\n"
        f"  filing_date:  [cyan]{raw.edgar_filing_date or '(no 10-K)'}[/cyan]"
    )

    # --- Generate or skip the LLM pass.
    if args.skip_llm:
        console.print("[dim]--skip-llm: writing stub draft without Ollama call.[/dim]")
        draft_body = (
            f"# {symbol} — Value Chain Brief (SKIPPED LLM)\n\n"
            "Run without --skip-llm to generate the full draft, or "
            "hand-author this file.\n"
        )
    else:
        console.print("Calling Qwen 2.5 7B to draft the brief...")
        draft_body = generate_value_chain_draft(symbol, raw=raw)

    VALUE_CHAINS_DIR.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(draft_body, encoding="utf-8")
    console.print(f"[green]Draft saved:[/green] {draft_path}")

    # --- Register in tickers.yaml.
    try:
        add_ticker_to_registry(
            TICKERS_YAML,
            symbol=symbol,
            tier=args.tier,
            notes=args.notes,
            overwrite=args.overwrite_tier,
        )
    except RegistryError as e:
        console.print(f"[red]Registry update failed: {e}[/red]")
        return 1
    console.print(f"[green]Registered[/green] in tier_{args.tier}")

    # --- Next-step guidance.
    console.print()
    console.print("[bold]Next steps[/bold]")
    console.print(f"1. Review [cyan]{draft_path}[/cyan] — Vulnerable links section "
                  f"especially. Look for [cyan][?UNCERTAIN][/cyan] markers.")
    console.print(f"2. Rename: [cyan]mv {draft_path.name} {final_path.name}[/cyan]")
    console.print(f"3. Run crew: [cyan]python scripts/run_crew.py {symbol}[/cyan]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker (e.g. NVDA, GEV)")
    parser.add_argument("--tier", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument(
        "--notes", default="", help="Optional one-line note for the registry entry"
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip the Ollama call (fast dry-run for registry + raw-material fetch)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing draft or approved brief",
    )
    parser.add_argument(
        "--overwrite-tier",
        action="store_true",
        help="Move the ticker to the new tier if it's already registered elsewhere",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
