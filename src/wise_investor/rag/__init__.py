"""ChromaDB-based RAG store.

Phase 3D scaffold: SEC EDGAR 10-K ingestion pipeline.

  edgar.py     fetches the latest 10-K HTML for a ticker (polite UA, disk cache)
  sections.py  slices Business / Risk Factors / MD&A / Quant Market Risk
               from 10-K text via regex heading markers
  index.py     ChromaDB PersistentClient with a MiniLM collection,
               chunked upsert_10k_sections() and search() over metadata

Analyst integration (passing retrieved passages into the pre-gather
prompt) is intentionally deferred — first we confirm the ingest and
retrieval quality on real filings, then wire it into the crew.
Manual value chain docs continue to flow via prompt blocks, not RAG
(design-v2.2 re-review High #5).
"""
