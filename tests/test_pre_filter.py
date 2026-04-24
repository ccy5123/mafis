"""Tests for the pre-filter pipeline (Stage 1 keyword + Stage 2 graph)."""

from __future__ import annotations

from wise_investor.alerts.chain_alerts import NewsItemLike
from wise_investor.filters.pre_filter import (
    DEFAULT_THRESHOLDS,
    FilterHit,
    aggregate_scores,
    recommend_promotions,
    scan_graph_context,
    scan_keywords,
)
from wise_investor.value_chain.graph import (
    CompanyNode,
    Relationship,
    ValueChainGraph,
)


def _news(title: str, source: str = "Reuters", published: str = "2026-04-24") -> NewsItemLike:
    return NewsItemLike(title=title, source=source, published=published, kind="news")


# ---------------------------------------------------------------------------
# Stage 1 — keyword match
# ---------------------------------------------------------------------------


def test_scan_keywords_matches_symbol() -> None:
    items = [_news("NVDA stock rallies on AI demand")]
    hits = scan_keywords(items, "NVDA")
    assert len(hits) == 1
    assert hits[0].stage == "keyword"
    assert hits[0].matched_term == "NVDA"


def test_scan_keywords_matches_company_name() -> None:
    items = [_news("NVIDIA announces new GPU architecture")]
    hits = scan_keywords(items, "NVDA")
    # Company-name alias "NVIDIA" lives in SYMBOL_KEYWORDS for NVDA.
    assert len(hits) == 1
    assert "NVIDIA" in hits[0].matched_term


def test_scan_keywords_skips_unrelated() -> None:
    items = [_news("Weather alert in Tokyo")]
    hits = scan_keywords(items, "NVDA")
    assert hits == []


def test_scan_keywords_word_boundary_avoids_substring_false_positive() -> None:
    items = [_news("Scientists study amphetamide receptor")]
    hits = scan_keywords(items, "AMD")
    # "AMD" should NOT match inside "amphetamide" thanks to word boundary.
    assert hits == []


def test_scan_keywords_one_hit_per_news_even_if_multiple_keywords_match() -> None:
    items = [_news("NVIDIA (NVDA) reports strong quarter")]
    hits = scan_keywords(items, "NVDA")
    # Both "NVIDIA" and "NVDA" appear but dedup to one FilterHit per news.
    assert len(hits) == 1


def test_scan_keywords_dedupes_by_title() -> None:
    items = [
        _news("NVDA rallies"),
        _news("NVDA rallies"),  # identical title
    ]
    hits = scan_keywords(items, "NVDA")
    assert len(hits) == 1


def test_scan_keywords_accepts_extra_keywords() -> None:
    items = [_news("Quiet day for unusual term xyzzy-corp")]
    hits = scan_keywords(items, "ZZZZ", extra_keywords=["xyzzy-corp"])
    assert len(hits) == 1
    assert hits[0].matched_term == "xyzzy-corp"


# ---------------------------------------------------------------------------
# Stage 2 — graph context
# ---------------------------------------------------------------------------


def _fixture_graph() -> ValueChainGraph:
    g = ValueChainGraph()
    g.add_company(CompanyNode(name="NVDA", ticker="NVDA", is_target=True))
    g.add_company(CompanyNode(name="TSMC", ticker="TSM"))
    g.add_company(CompanyNode(name="AMD", ticker="AMD"))
    g.add_company(CompanyNode(name="ASML", ticker="ASML"))
    g.add_relationship(Relationship("TSMC", "NVDA", "supplies"))
    g.add_relationship(Relationship("ASML", "TSMC", "supplies"))
    g.add_relationship(Relationship("NVDA", "AMD", "peer"))
    g.add_relationship(Relationship("AMD", "NVDA", "peer"))
    return g


def test_scan_graph_context_matches_supplier_news() -> None:
    g = _fixture_graph()
    items = [_news("TSMC Q2 outlook: strong demand from AI customers")]
    hits = scan_graph_context(items, g, "NVDA")
    assert len(hits) == 1
    assert hits[0].stage == "graph_context"
    assert hits[0].matched_term == "TSMC"
    assert "TSMC" in hits[0].reason
    # Path should include both TSMC and NVDA.
    assert "TSMC" in hits[0].graph_path
    assert "NVDA" in hits[0].graph_path


def test_scan_graph_context_two_hop_indirect() -> None:
    g = _fixture_graph()
    # ASML is 2 hops from NVDA (ASML → TSMC → NVDA).
    items = [_news("ASML raises 2026 EUV scanner shipment target")]
    hits = scan_graph_context(items, g, "NVDA", max_hops=2)
    assert any(h.matched_term == "ASML" for h in hits)


def test_scan_graph_context_respects_hops_ceiling() -> None:
    g = _fixture_graph()
    items = [_news("ASML EUV shipment up")]
    # With max_hops=1, ASML → NVDA should NOT register.
    hits = scan_graph_context(items, g, "NVDA", max_hops=1)
    assert not any(h.matched_term == "ASML" for h in hits)


def test_scan_graph_context_returns_empty_when_symbol_absent_from_graph() -> None:
    g = _fixture_graph()
    items = [_news("TSMC news")]
    hits = scan_graph_context(items, g, "ZZZZ")
    assert hits == []


def test_scan_graph_context_excludes_direct_symbol_mentions() -> None:
    """Stage 2 is about CONTEXT via neighbors — direct symbol mentions
    are Stage 1 territory. Avoid double-counting.
    """
    g = _fixture_graph()
    items = [_news("NVDA posts record Q1 results")]
    hits = scan_graph_context(items, g, "NVDA")
    # Target's own mention handled by Stage 1 / chain_alerts, not here.
    assert hits == []


# ---------------------------------------------------------------------------
# Aggregation + promotion logic
# ---------------------------------------------------------------------------


def test_aggregate_scores_counts_hits_per_symbol() -> None:
    hits = [
        FilterHit(symbol="NVDA", stage="keyword", matched_term="NVDA",
                  news_title="A", news_source="", news_published=""),
        FilterHit(symbol="NVDA", stage="graph_context", matched_term="TSMC",
                  news_title="B", news_source="", news_published=""),
        FilterHit(symbol="AMD", stage="keyword", matched_term="AMD",
                  news_title="C", news_source="", news_published=""),
    ]
    scores = aggregate_scores(hits)
    assert scores == {"NVDA": 2, "AMD": 1}


def test_aggregate_scores_dedupes_same_symbol_same_title() -> None:
    hits = [
        FilterHit(symbol="NVDA", stage="keyword", matched_term="NVDA",
                  news_title="Same title", news_source="", news_published=""),
        FilterHit(symbol="NVDA", stage="graph_context", matched_term="TSMC",
                  news_title="Same title", news_source="", news_published=""),
    ]
    scores = aggregate_scores(hits)
    # Same symbol + same title → one composite hit.
    assert scores["NVDA"] == 1


def test_recommend_promotions_promotes_tier3_to_tier2() -> None:
    scores = {"AMD": 5}
    recs = recommend_promotions(scores=scores, current_tiers={"AMD": 3})
    assert len(recs) == 1
    r = recs[0]
    assert r.symbol == "AMD"
    assert r.current_tier == 3
    assert r.suggested_tier == 2


def test_recommend_promotions_promotes_tier2_to_tier1() -> None:
    scores = {"TSM": 10}
    recs = recommend_promotions(scores=scores, current_tiers={"TSM": 2})
    assert len(recs) == 1
    assert recs[0].suggested_tier == 1


def test_recommend_promotions_no_op_when_already_tier1() -> None:
    scores = {"NVDA": 20}
    recs = recommend_promotions(scores=scores, current_tiers={"NVDA": 1})
    assert recs == []


def test_recommend_promotions_no_op_below_threshold() -> None:
    scores = {"AMD": 2}  # threshold for Tier 3→2 is 3
    recs = recommend_promotions(scores=scores, current_tiers={"AMD": 3})
    assert recs == []


def test_recommend_promotions_unknown_ticker_gets_tier3_suggestion() -> None:
    """If a scored ticker isn't in the registry at all, recommend
    adding it at Tier 3 so future scans have a baseline.
    """
    scores = {"NEWCO": 2}
    recs = recommend_promotions(scores=scores, current_tiers={"NEWCO": None})
    assert len(recs) == 1
    assert recs[0].current_tier is None
    assert recs[0].suggested_tier == 3


def test_recommend_promotions_sample_titles_populated() -> None:
    hit = FilterHit(
        symbol="AMD",
        stage="keyword",
        matched_term="AMD",
        news_title="AMD beats earnings",
        news_source="Reuters",
        news_published="2026-04-24",
    )
    recs = recommend_promotions(
        scores={"AMD": 5},
        current_tiers={"AMD": 3},
        sample_hits={"AMD": [hit]},
    )
    assert recs[0].sample_titles == ["AMD beats earnings"]


def test_recommend_promotions_ranks_by_score_desc() -> None:
    scores = {"AMD": 4, "TSM": 10, "INTC": 3}
    recs = recommend_promotions(
        scores=scores,
        current_tiers={"AMD": 3, "TSM": 2, "INTC": 3},
    )
    symbols_in_order = [r.symbol for r in recs]
    # TSM score 10 ranks first; AMD (4) before INTC (3).
    assert symbols_in_order == ["TSM", "AMD", "INTC"]


def test_default_thresholds_are_sensible() -> None:
    assert DEFAULT_THRESHOLDS[2] < DEFAULT_THRESHOLDS[1]  # tier 1 needs more signal
