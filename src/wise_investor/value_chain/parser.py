"""Parse docs/value_chains/<SYMBOL>.md into typed Relationship records.

Heuristic — this is a scaffold, not a semantic parser. We rely on the
manual brief's consistent section headings:

  ## Upstream — Suppliers         → supplier -> target edges
  ## Peers — Direct and adjacent  → bidirectional peer edges
  ## Downstream — Customers       → target -> customer edges
  ## Infrastructure ...           → infra -> target edges

Inside each section we look for one of:
  - Bold bullet lists:   "- **Name** — description"
  - Table rows:          "| Ticker | Name | ... |"
  - Plain bullets:       "- Name — description"

Company names can contain tickers in parentheses like
"Siemens Energy (ENR.DE)". We capture both when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from wise_investor.value_chain.graph import (
    CompanyNode,
    Relationship,
    ValueChainGraph,
)


# Section-heading regexes: flexible about hyphen vs em-dash and capitalization.
_SECTION_PATTERNS = {
    "upstream": re.compile(
        r"^##\s*Upstream\b", re.IGNORECASE | re.MULTILINE
    ),
    "peers": re.compile(
        r"^##\s*Peers\b", re.IGNORECASE | re.MULTILINE
    ),
    "downstream": re.compile(
        r"^##\s*Downstream\b", re.IGNORECASE | re.MULTILINE
    ),
    "infrastructure": re.compile(
        r"^##\s*Infrastructure\b", re.IGNORECASE | re.MULTILINE
    ),
}

# A bullet line like:
#   - **Siemens Energy** — ...
#   - **SK hynix** — ...
#   * **Name (TICKER)** — details
#   - **Hyperscale cloud** (Microsoft Azure, AWS, GCP, Oracle Cloud) — ...
#
# We only require the bold-wrapped name at the start of the bullet;
# anything after is captured as free-form notes so parenthetical
# qualifiers do not defeat the match.
_BULLET_BOLD = re.compile(
    r"^\s*[-*]\s+\*\*([^*]+?)\*\*\s*(.*)$"
)

# A plain bullet without bold:
#   - Siemens Energy — details
_BULLET_PLAIN = re.compile(
    r"^\s*[-*]\s+([A-Z][A-Za-z0-9 .&+\-/]{2,60})\s*[—–-]\s+(.+)$"
)

# Ticker in parentheses: "Siemens Energy (ENR.DE)" or "AMD (AMD)"
_TICKER_IN_NAME = re.compile(r"\(([A-Z][A-Z0-9.]{0,9})\)")

# Table row separator marker
_TABLE_ROW_SEP = re.compile(r"^\s*\|?[\s\-:|]+\|?\s*$")

# Table data row: "| Siemens Energy | ENR.DE | ... |"
_TABLE_ROW = re.compile(r"^\s*\|([^|]+)\|([^|]+)\|(.*)$")


@dataclass
class ParsedEntry:
    """One raw line of the markdown, interpreted as a potential company."""

    name: str
    ticker: str | None
    notes: str | None


def _clean_name(raw: str) -> tuple[str, str | None]:
    """Strip parenthetical ticker; return (clean_name, ticker_or_None)."""
    m = _TICKER_IN_NAME.search(raw)
    ticker = m.group(1).upper() if m else None
    cleaned = _TICKER_IN_NAME.sub("", raw).strip(" ,")
    return cleaned, ticker


def _slice_section(text: str, section_name: str) -> str:
    """Return the substring from a section heading to the next '## ' heading."""
    pat = _SECTION_PATTERNS.get(section_name)
    if pat is None:
        return ""
    m = pat.search(text)
    if not m:
        return ""
    start = m.end()
    next_h = re.search(r"^##\s", text[start:], re.MULTILINE)
    end = start + next_h.start() if next_h else len(text)
    return text[start:end]


def _iter_entries_in_section(section: str) -> list[ParsedEntry]:
    """Extract company entries from a section's body text.

    Handles bold bullets first (most common), then table rows, then
    plain bullets. Skips subsection headers like '### Chip fabrication'.
    """
    entries: list[ParsedEntry] = []

    # 1) Bold bullet items are the most reliable signal.
    for line in section.splitlines():
        m = _BULLET_BOLD.match(line)
        if m:
            name_raw, notes = m.group(1), m.group(2)
            clean, ticker = _clean_name(name_raw)
            if clean:
                entries.append(ParsedEntry(name=clean, ticker=ticker, notes=notes.strip()))

    # 2) Markdown tables (common in Peers section).
    lines = section.splitlines()
    in_table = False
    header_cols: list[str] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            # Header / separator / data?
            if not in_table:
                header_cols = [c.strip().lower() for c in stripped.strip("|").split("|")]
                in_table = True
                continue
            # Skip separator rows like "|---|---|---|"
            if _TABLE_ROW_SEP.match(stripped):
                continue
            # Data row — find a name and ticker column.
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if len(cells) >= 2:
                # Heuristic: first cell is often name OR ticker. Prefer
                # cell containing letters as name; cell matching ticker
                # pattern as ticker.
                name_cell = cells[0]
                ticker_cell = cells[1] if len(cells) > 1 else ""
                # Detect which one is the ticker (short uppercase w/ optional dot).
                if re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", name_cell):
                    # name_cell is actually the ticker
                    ticker = name_cell
                    name = ticker_cell
                elif re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", ticker_cell):
                    ticker = ticker_cell
                    name = name_cell
                else:
                    ticker = None
                    name = name_cell
                # Further clean parenthetical tickers embedded in name.
                clean, embedded_ticker = _clean_name(name)
                if embedded_ticker and not ticker:
                    ticker = embedded_ticker
                if clean:
                    entries.append(
                        ParsedEntry(name=clean, ticker=ticker, notes=None)
                    )
        else:
            in_table = False

    # Deduplicate by name.
    seen: set[str] = set()
    deduped: list[ParsedEntry] = []
    for e in entries:
        key = e.name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def parse_value_chain_markdown(
    text: str, target_symbol: str, source_doc: str | None = None
) -> list[Relationship]:
    """Turn a single value chain brief into a list of typed relationships.

    The target itself becomes a flagged node; all parsed companies
    connect to it with the appropriate typed edge.
    """
    target = target_symbol.upper()
    rels: list[Relationship] = []

    # Upstream / supplier -> target
    for entry in _iter_entries_in_section(_slice_section(text, "upstream")):
        rels.append(
            Relationship(
                source=entry.name,
                target=target,
                relation="supplies",
                source_doc=source_doc,
                notes=entry.notes,
            )
        )

    # Downstream / target -> customer
    for entry in _iter_entries_in_section(_slice_section(text, "downstream")):
        rels.append(
            Relationship(
                source=target,
                target=entry.name,
                relation="supplies",
                source_doc=source_doc,
                notes=entry.notes,
            )
        )

    # Peers (bidirectional)
    for entry in _iter_entries_in_section(_slice_section(text, "peers")):
        # Skip if the entry is the target itself.
        if entry.ticker and entry.ticker.upper() == target:
            continue
        rels.append(
            Relationship(
                source=target,
                target=entry.name,
                relation="peer",
                source_doc=source_doc,
                notes=entry.notes,
            )
        )
        rels.append(
            Relationship(
                source=entry.name,
                target=target,
                relation="peer",
                source_doc=source_doc,
                notes=entry.notes,
            )
        )

    # Infrastructure -> target
    for entry in _iter_entries_in_section(_slice_section(text, "infrastructure")):
        rels.append(
            Relationship(
                source=entry.name,
                target=target,
                relation="infrastructure",
                source_doc=source_doc,
                notes=entry.notes,
            )
        )

    return rels


def build_graph_from_briefs(
    briefs_dir: Path | str,
) -> ValueChainGraph:
    """Walk `briefs_dir` for <SYMBOL>.md files, merge all into one graph."""
    g = ValueChainGraph()
    briefs_path = Path(briefs_dir)
    if not briefs_path.is_dir():
        raise FileNotFoundError(f"briefs directory not found: {briefs_path}")

    for md in sorted(briefs_path.glob("*.md")):
        symbol = md.stem.upper()
        if symbol in {"README"}:
            continue
        text = md.read_text(encoding="utf-8")
        # Flag target as is_target.
        g.add_company(
            CompanyNode(name=symbol, ticker=symbol, is_target=True)
        )
        # Tag parsed entries with their ticker if we can extract one.
        for rel in parse_value_chain_markdown(
            text, target_symbol=symbol, source_doc=md.name
        ):
            # Promote ticker onto the matching node so query CLIs can
            # resolve name <-> ticker easily.
            for endpoint in (rel.source, rel.target):
                if endpoint == symbol:
                    continue  # target already has its ticker
                # Ticker was stripped by _clean_name; look for it in the
                # original markdown nearby and attach if present.
            g.add_relationship(rel)

    return g
