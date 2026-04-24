"""Scan recent news against the 3-Tier registry → promotion recommendations.

Usage:
    python scripts/prefilter_scan.py                    # all tickers
    python scripts/prefilter_scan.py --tier 3           # only Tier 3
    python scripts/prefilter_scan.py --graph-context    # include stage 2
    python scripts/prefilter_scan.py --output reports/prefilter.md

What it does:
  1. Load config/tickers.yaml (all three tiers).
  2. Load the value chain graph if --graph-context is set.
  3. Pull a news pool: for each Tier 1/2 symbol, use its keyword
     profile; union and dedup across tickers.
  4. For each registered ticker, run Stage 1 (keyword match) and
     optionally Stage 2 (graph context within N hops).
  5. Aggregate scores and emit promotion recommendations.
  6. Print a Rich-formatted report; optionally save as markdown.

Designed for cron — stateless scan, no ledger. The recommender
outputs are advisory; a human still edits tickers.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from wise_investor.alerts.chain_alerts import NewsItemLike  # noqa: E402
from wise_investor.filters.pre_filter import (  # noqa: E402
    aggregate_scores,
    recommend_promotions,
    scan_graph_context,
    scan_keywords,
)
from wise_investor.onboarding.tickers_yaml import load_registry_yaml  # noqa: E402
from wise_investor.value_chain.graph import ValueChainGraph  # noqa: E402


console = Console()

TICKERS_YAML = REPO_ROOT / "config" / "tickers.yaml"
GRAPH_PATH = REPO_ROOT / "data" / "value_chain.graph.json"


def _pull_news_pool(active_symbols: list[str]) -> list[NewsItemLike]:
    """Pull news items for all active (Tier 1/2) symbols and dedup."""
    if not active_symbols:
        return []
    try:
        from wise_investor.geopolitics.snapshot import get_geopolitics_snapshot
    except Exception as e:
        console.print(f"[red]Geopolitics import failed: {e}[/red]")
        return []

    seen: set[str] = set()
    items: list[NewsItemLike] = []
    for sym in active_symbols:
        console.print(f"  pulling news for {sym}...")
        try:
            snap = get_geopolitics_snapshot(sym)
        except Exception as e:
            console.print(f"[yellow]Snapshot {sym} failed: {e}[/yellow]")
            continue
        for gn in snap.google_news:
            key = gn.title.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(
                NewsItemLike(
                    title=gn.title,
                    source=gn.source or "Google News",
                    published=gn.published or "",
                    kind="google_news",
                )
            )
        for theme in snap.gdelt_themes:
            for art in theme.articles:
                key = art.title.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    NewsItemLike(
                        title=art.title,
                        source=art.domain or "GDELT",
                        published=art.iso_date[:10] if art.iso_date else "",
                        kind="gdelt",
                    )
                )
    return items


def _flatten_registry(data: dict) -> dict[str, int]:
    """Return {SYMBOL: tier} across all three tiers."""
    out: dict[str, int] = {}
    for tier in (1, 2, 3):
        for entry in data.get(f"tier_{tier}", []) or []:
            if not isinstance(entry, dict):
                continue
            sym = str(entry.get("symbol", "")).upper()
            if sym:
                out[sym] = tier
    return out


def run(args: argparse.Namespace) -> int:
    registry = load_registry_yaml(TICKERS_YAML)
    symbols_by_tier = _flatten_registry(registry)
    if not symbols_by_tier:
        console.print("[yellow]No tickers registered.[/yellow]")
        return 0

    console.rule("[bold]Pre-filter scan[/bold]")
    by_tier = {1: [], 2: [], 3: []}
    for sym, tier in symbols_by_tier.items():
        by_tier[tier].append(sym)
    for tier in (1, 2, 3):
        console.print(
            f"  tier_{tier}: [cyan]{', '.join(by_tier[tier]) or '(none)'}[/cyan]"
        )

    # Filter to requested tier if specified.
    target_symbols = list(symbols_by_tier.keys())
    if args.tier:
        target_symbols = [s for s, t in symbols_by_tier.items() if t == args.tier]
        if not target_symbols:
            console.print(f"[yellow]No tickers in tier_{args.tier}[/yellow]")
            return 0

    # News pool: use Tier 1 + 2 keyword profiles.
    active_symbols = [s for s, t in symbols_by_tier.items() if t in (1, 2)]
    news = _pull_news_pool(active_symbols)
    console.print(f"Collected [cyan]{len(news)}[/cyan] unique news items")

    # Optionally load the value chain graph for Stage 2.
    graph: ValueChainGraph | None = None
    if args.graph_context:
        if not GRAPH_PATH.exists():
            console.print(
                f"[yellow]Graph not found at {GRAPH_PATH} — "
                "Stage 2 skipped.[/yellow]"
            )
        else:
            graph = ValueChainGraph.load(GRAPH_PATH)
            console.print(
                f"Graph loaded: [cyan]{graph.num_nodes} nodes, "
                f"{graph.num_edges} edges[/cyan]"
            )

    # Run stages per symbol.
    all_hits = []
    sample_hits: dict[str, list] = {}
    for sym in target_symbols:
        hits1 = scan_keywords(news, sym)
        hits2 = scan_graph_context(news, graph, sym) if graph else []
        all_hits.extend(hits1)
        all_hits.extend(hits2)
        combined = hits1 + hits2
        if combined:
            sample_hits[sym] = combined

    # Stage 3: optional semantic relevance filter. The LLM reads each
    # hit's (ticker, headline) pair and keeps only material ones.
    if args.semantic and all_hits:
        from wise_investor.filters.semantic import (
            filter_hits_semantically,
            materials_only,
        )

        console.print(
            f"Running Stage 3 semantic filter on "
            f"[cyan]{len(all_hits)}[/cyan] hit(s) (max {args.semantic_max})..."
        )
        decisions = filter_hits_semantically(
            all_hits, max_hits=args.semantic_max
        )
        kept = materials_only(decisions)
        dropped = len(all_hits) - len(kept)
        console.print(
            f"Semantic filter: [cyan]{len(kept)} kept[/cyan], "
            f"[yellow]{dropped} dropped[/yellow]"
        )
        all_hits = kept
        # Rebuild sample_hits off the kept list.
        sample_hits = {}
        for h in all_hits:
            sample_hits.setdefault(h.symbol, []).append(h)

    scores = aggregate_scores(all_hits)
    if not scores:
        console.print("[dim]No pre-filter hits in this news batch.[/dim]")
        return 0

    # Render score table.
    console.rule("[bold]Per-ticker hit counts[/bold]")
    tbl = Table()
    tbl.add_column("Symbol", style="cyan")
    tbl.add_column("Tier", justify="right")
    tbl.add_column("Hits", justify="right")
    tbl.add_column("Stages", justify="right")
    tbl.add_column("Sample headline")
    for sym, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        tier = symbols_by_tier.get(sym, "—")
        hits_for_sym = [h for h in all_hits if h.symbol == sym]
        stages = sorted({h.stage for h in hits_for_sym})
        sample = hits_for_sym[0].news_title if hits_for_sym else ""
        tbl.add_row(
            sym,
            str(tier) if tier != "—" else tier,
            str(score),
            ", ".join(stages),
            sample[:80] + ("..." if len(sample) > 80 else ""),
        )
    console.print(tbl)

    # Recommendations.
    recs = recommend_promotions(
        scores=scores,
        current_tiers={s: symbols_by_tier.get(s) for s in scores},
        sample_hits=sample_hits,
    )
    if recs:
        console.rule("[bold]Promotion recommendations[/bold]")
        for r in recs:
            console.print(
                f"  [cyan]{r.symbol}[/cyan]: "
                f"tier_{r.current_tier or '(unregistered)'} → "
                f"[green]tier_{r.suggested_tier}[/green] "
                f"(score={r.score})"
            )
            for t in r.sample_titles[:2]:
                console.print(f"    · {t[:100]}")
    else:
        console.print("[dim]No promotion thresholds crossed.[/dim]")

    if args.output:
        md_lines = ["# Pre-filter scan\n"]
        for r in recs:
            md_lines.append(
                f"- **{r.symbol}**: tier_{r.current_tier or '?'} → "
                f"tier_{r.suggested_tier} (score {r.score})"
            )
            for t in r.sample_titles[:2]:
                md_lines.append(f"  - {t}")
        Path(args.output).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        console.print(f"[green]Saved markdown to {args.output}[/green]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Scan only one tier (default: all three)",
    )
    parser.add_argument(
        "--graph-context",
        action="store_true",
        help="Also run Stage 2 graph-context matching (needs data/value_chain.graph.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown file path for the scan summary",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help=(
            "Run Stage 3 semantic filter via Qwen 2.5 7B on the keyword / "
            "graph-context hits to drop immaterial mentions. Adds ~3s "
            "per hit; use --semantic-max to cap."
        ),
    )
    parser.add_argument(
        "--semantic-max",
        type=int,
        default=20,
        help="Cap on Stage 3 LLM calls to bound cost (default 20)",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
