"""Phase 3 pre-filter tiers — scale Tier 3 watchlist without doing
full-crew work on dormant names.

Design-v2.2 §10.2 Phase 3 task 5: "사전 필터 2~3단계 (밸류체인 키워드
매칭 + 로컬 AI 의미 필터링)". This package ships stages 1 and 2:

  Stage 1 — keyword match
    For each ticker in config/tickers.yaml, scan recent news
    headlines for the symbol, company name, or registered
    keyword tokens. Emit a FilterHit per match.

  Stage 2 — graph context match
    For tickers that have a value chain brief (Tier 1/2), also
    match mentions of graph nodes within N hops. A news item that
    names "TSMC" is relevant to NVDA even if NVDA isn't in the
    headline.

  Stage 3 — local AI semantic filter (deferred)
    Would feed finalist hits to Qwen 2.5 7B with "is this
    material for $TICKER?" and filter on the answer. Follow-up.

Output: aggregated scores per ticker, plus promotion recommendations
("Tier 3 ZZZZ has 5 news hits in the last 24h — consider promoting
to Tier 2 for active pre-gather").
"""

from wise_investor.filters.pre_filter import (
    FilterHit,
    PromotionRecommendation,
    aggregate_scores,
    recommend_promotions,
    scan_graph_context,
    scan_keywords,
)

__all__ = [
    "FilterHit",
    "PromotionRecommendation",
    "aggregate_scores",
    "recommend_promotions",
    "scan_graph_context",
    "scan_keywords",
]
