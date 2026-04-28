"""Stage 2 quantitative pre-filter for the MAFIS constitution v2.0.

Implements `docs/constitution.md` Part IV (quantitative proxies) — the
cheap, deterministic, LLM-free first cut on the universe before Stage
3 hands surviving candidates to a small LLM for qualitative judgment.

The split is deliberate: data fetch is messy and source-dependent
(Finnhub, FMP, DART, EDGAR all return different shapes), but proxy
computation is pure math on a uniform data structure. We keep the two
worlds separated through `types.TickerFundamentals` — adapters know
how to populate it; proxies/prefilter know nothing about where it
came from. That separation is what makes the calibration phase (§23
Step 4) tractable: hand-curated JSON files can drive the screening
pipeline end-to-end without a network round trip.

Constitution version stamping per §13 + §22 lives here so every
output that flows downstream carries the rubric version it was scored
against.
"""

CONSTITUTION_VERSION = "2.0"

__all__ = ["CONSTITUTION_VERSION"]
