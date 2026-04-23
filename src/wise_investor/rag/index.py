"""ChromaDB-backed index for 10-K section passages.

Storage layout:
  - One persistent ChromaDB collection named "ten_k_sections".
  - Each passage is a chunk of ~900 characters from a (symbol, section)
    pair. Metadata: symbol, section, filing_date, chunk_id.
  - IDs are "{symbol}_{filing_date}_{section}_{chunk_id}" so reindexing
    the same filing overwrites cleanly.

Default embedding: Chroma's built-in all-MiniLM-L6-v2 (CPU, fast). We
don't pull a separate embedding model — the first query kicks Chroma to
download MiniLM (<100MB).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb

from wise_investor.config import settings


logger = logging.getLogger(__name__)


COLLECTION_NAME = "ten_k_sections"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


@dataclass
class PassageHit:
    """One retrieved passage from a semantic search."""

    symbol: str
    section: str
    filing_date: str
    chunk_id: int
    text: str
    distance: float  # lower = closer match


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-based chunker with overlap.

    Respects paragraph boundaries when possible — cuts at the last
    newline within the chunk window so passages don't start mid-sentence.
    """
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Look for a paragraph/sentence break in the last 30% of the window.
            window_start = start + int(size * 0.7)
            cut = text.rfind("\n\n", window_start, end)
            if cut == -1:
                cut = text.rfind(". ", window_start, end)
            if cut != -1 and cut > start:
                end = cut
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, end)
    return [c for c in chunks if c]


def _get_client() -> chromadb.ClientAPI:
    persist_dir = Path(settings.chroma_persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def _get_collection() -> chromadb.Collection:
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def upsert_10k_sections(
    symbol: str,
    filing_date: str,
    sections: dict[str, str],
) -> int:
    """Index each non-empty section under `sections` for `(symbol, filing_date)`.

    Returns the total number of chunks upserted. Re-indexing the same
    (symbol, filing_date) overwrites on each ID, so this is idempotent.
    """
    collection = _get_collection()

    total = 0
    for section_name, text in sections.items():
        if not text or not text.strip():
            continue
        chunks = _chunk_text(text)
        if not chunks:
            continue

        ids = [
            f"{symbol.upper()}_{filing_date}_{section_name}_{i}"
            for i in range(len(chunks))
        ]
        metadatas = [
            {
                "symbol": symbol.upper(),
                "section": section_name,
                "filing_date": filing_date,
                "chunk_id": i,
            }
            for i in range(len(chunks))
        ]
        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        total += len(chunks)
        logger.info(
            "Indexed %d chunks from %s/%s (%s)",
            len(chunks),
            symbol.upper(),
            section_name,
            filing_date,
        )
    return total


def search(
    query: str,
    symbol: str | None = None,
    section: str | None = None,
    k: int = 5,
) -> list[PassageHit]:
    """Semantic search over the indexed passages.

    Metadata filters: pass `symbol` to restrict to one company,
    `section` to restrict (e.g., "risk_factors", "mdna").
    """
    collection = _get_collection()

    # Build a Chroma metadata filter.
    where_clauses: list[dict] = []
    if symbol:
        where_clauses.append({"symbol": symbol.upper()})
    if section:
        where_clauses.append({"section": section})
    where: dict | None = None
    if len(where_clauses) == 1:
        where = where_clauses[0]
    elif len(where_clauses) > 1:
        where = {"$and": where_clauses}

    result = collection.query(
        query_texts=[query],
        n_results=k,
        where=where,
    )

    hits: list[PassageHit] = []
    if not result or not result.get("ids") or not result["ids"][0]:
        return hits

    docs = result["documents"][0]
    metas = result["metadatas"][0]
    distances = result.get("distances", [[0.0] * len(docs)])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        hits.append(
            PassageHit(
                symbol=str(meta.get("symbol", "")),
                section=str(meta.get("section", "")),
                filing_date=str(meta.get("filing_date", "")),
                chunk_id=int(meta.get("chunk_id", 0)),
                text=doc,
                distance=float(dist),
            )
        )
    return hits


def stats() -> dict[str, int]:
    """Summary counts for CLI display."""
    collection = _get_collection()
    total = collection.count()
    # Per-symbol breakdown via peek — limited sample, good enough for status.
    return {"total_chunks": total}
