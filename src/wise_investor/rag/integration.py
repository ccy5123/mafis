"""RAG integration helpers — bridge between EDGAR ingestion and the agent
pre-gather pipeline.

`ensure_10k_indexed(symbol)` idempotently downloads + parses + indexes a
filing, reusing disk cache on repeat calls. `gather_section_passages()`
runs a named set of queries against the indexed collection and returns
passages formatted for inclusion in the LLM context.

Both functions are designed to fail soft: EDGAR outages, missing CIKs
(common for non-US / Korean tickers), and ChromaDB errors all degrade
to "no passages available" rather than aborting the whole crew run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from wise_investor.rag.edgar import EdgarError, FilingRef, download_10k
from wise_investor.rag.index import PassageHit, search, stats, upsert_10k_sections
from wise_investor.rag.sections import extract_sections


logger = logging.getLogger(__name__)


# Default query set for Analyst + Skeptic consumption. Labels become the
# `edgar.<label>` keys in the facts dict so the LLM can cite each block
# distinctly. Queries intentionally use financial-report vocabulary so
# MiniLM lands on high-signal passages rather than boilerplate.
DEFAULT_QUERIES: dict[str, str] = {
    "business_segments": "business segments products revenue operations",
    "moat_signals": (
        "competitive advantages market share intellectual property "
        "customer concentration barriers to entry"
    ),
    "risk_factors": (
        "supply chain risks regulatory export controls dependence "
        "concentration critical"
    ),
    "mdna_highlights": (
        "results of operations gross margin growth drivers "
        "capital expenditure outlook"
    ),
}


@dataclass
class SectionPassages:
    label: str
    filing_date: str | None
    passages: list[PassageHit]


def ensure_10k_indexed(
    symbol: str, use_cache: bool = True
) -> FilingRef | None:
    """Ensure the latest 10-K for `symbol` is downloaded, parsed, and indexed.

    Idempotent:
      - Disk cache in data/edgar_cache/ short-circuits download on repeat calls.
      - ChromaDB `upsert` uses deterministic IDs, so re-running is a no-op.

    Returns the FilingRef on success, None on any failure (no CIK, network,
    parse). Callers must tolerate None and skip RAG augmentation for that
    symbol.
    """
    try:
        ref, html = download_10k(symbol, use_cache=use_cache)
    except EdgarError as e:
        logger.warning("EDGAR download failed for %s: %s", symbol, e)
        return None
    except Exception as e:
        logger.warning("EDGAR download raised for %s: %s", symbol, e)
        return None

    try:
        parsed = extract_sections(html)
        as_dict = parsed.as_dict()
        if not as_dict:
            logger.warning("No sections extracted from 10-K for %s", symbol)
            return ref
        upsert_10k_sections(
            symbol=symbol,
            filing_date=ref.filing_date,
            sections=as_dict,
        )
    except Exception as e:
        logger.warning("Section extract/upsert failed for %s: %s", symbol, e)
        return ref

    return ref


def gather_section_passages(
    symbol: str,
    queries: dict[str, str] | None = None,
    k: int = 3,
) -> dict[str, SectionPassages]:
    """Run each named query against the indexed collection for `symbol`.

    Returns a mapping of label → SectionPassages. Labels for which the
    index produced no hits still appear in the result with an empty
    passages list so the caller can report "no matches" uniformly.
    """
    queries = queries or DEFAULT_QUERIES
    out: dict[str, SectionPassages] = {}

    total = stats().get("total_chunks", 0)
    if total == 0:
        # No index at all — return empty passages for each label.
        for label in queries:
            out[label] = SectionPassages(label=label, filing_date=None, passages=[])
        return out

    for label, query in queries.items():
        try:
            hits = search(query=query, symbol=symbol, k=k)
        except Exception as e:
            logger.warning("RAG search failed for %s/%s: %s", symbol, label, e)
            hits = []

        filing_date = hits[0].filing_date if hits else None
        out[label] = SectionPassages(
            label=label, filing_date=filing_date, passages=hits
        )

    return out


def format_passages_as_tool_output(
    symbol: str, section: SectionPassages, max_chars_per_passage: int = 700
) -> str:
    """Render one SectionPassages into the body of a <tool_output> block.

    The body carries explicit per-passage citation hints so the LLM can
    copy-paste them verbatim into its report, e.g.:

        [Source: 10-K risk_factors, filed 2025-02-26]

    The format matches the Universal Citation Rule's `[Source: ...]`
    syntax so citation_rate / citation_audit recognize it uniformly.
    """
    if not section.passages:
        return f"No passages matched query for {symbol}/{section.label}."

    lines: list[str] = []
    filing_date = section.filing_date or "unknown"
    lines.append(
        f"10-K {section.label} excerpts for {symbol.upper()} "
        f"(filed {filing_date}):"
    )

    for i, p in enumerate(section.passages, start=1):
        excerpt = p.text.strip()
        if len(excerpt) > max_chars_per_passage:
            excerpt = excerpt[:max_chars_per_passage].rstrip() + " ..."
        lines.append("")
        lines.append(
            f"### Passage {i} "
            f"(section={p.section}, distance={p.distance:.3f})"
        )
        lines.append(excerpt)
        lines.append(
            f"[Source: 10-K {p.section}, filed {p.filing_date}]"
        )

    return "\n".join(lines)


def gather_and_format_for_pre_gather(
    symbol: str,
    queries: dict[str, str] | None = None,
    k: int = 3,
) -> dict[str, str]:
    """End-to-end helper for runner.pre_gather_facts.

    Ensures the 10-K is indexed, runs each named query, and returns a
    mapping of `edgar.<label>` → formatted string body suitable for
    direct insertion into the facts dict.

    On failure (no CIK, EDGAR outage, empty index) the returned body is
    a single-line ERROR / NO-DATA note so the LLM sees the absence and
    can reflect it in data-gap notes rather than the crew crashing.
    """
    queries = queries or DEFAULT_QUERIES
    out: dict[str, str] = {}

    ref = ensure_10k_indexed(symbol)
    if ref is None:
        # Could not resolve or download — emit uniform ERROR entries so
        # the facts dict has a stable schema across symbols.
        for label in queries:
            out[f"edgar.{label}"] = (
                f"ERROR: no 10-K available for {symbol.upper()} "
                f"(ticker may not be in SEC EDGAR, e.g. Korean listings)"
            )
        return out

    sections_map = gather_section_passages(symbol, queries=queries, k=k)
    for label, section in sections_map.items():
        key = f"edgar.{label}"
        out[key] = format_passages_as_tool_output(symbol, section)
    return out


__all__ = [
    "DEFAULT_QUERIES",
    "SectionPassages",
    "ensure_10k_indexed",
    "gather_section_passages",
    "format_passages_as_tool_output",
    "gather_and_format_for_pre_gather",
]
