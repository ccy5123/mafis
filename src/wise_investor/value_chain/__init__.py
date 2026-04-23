"""Phase 3C value chain graph package.

In-memory DAG of upstream/downstream/peer/infrastructure relationships
between companies, built from the hand-curated Markdown briefs in
docs/value_chains/. Today this is a read-only layer; future phases will
let pre-gather update it when new peers or suppliers are detected in
10-K text.
"""
