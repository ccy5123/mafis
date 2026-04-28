"""Stage 2 integration tests (constitution §16)."""

from __future__ import annotations

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.prefilter import (
    BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN,
    FRONTIER_MIN_YEARS_SINCE_INTRO,
    MOAT_ROIC_ADVANTAGE_MIN,
    _apply_hierarchy_gate,
    evaluate_ticker,
)
from wise_investor.screening.segments import (
    resolve_primary_segment,
    single_segment_default,
)
from wise_investor.screening.types import (
    AnnualFinancials,
    AxisVerdict,
    QuarterlyMargin,
    Segment,
    SegmentBreakdown,
    TickerFundamentals,
)


def _annual(year: int, nopat: float | None, capital: float | None) -> AnnualFinancials:
    return AnnualFinancials(
        fiscal_year=year,
        revenue=None,
        gross_profit=None,
        operating_income=None,
        nopat=nopat,
        invested_capital=capital,
        rd_expense=None,
    )


def _funds(
    *,
    symbol: str = "TEST",
    annual: tuple[AnnualFinancials, ...] = (),
    industry_roic: float | None = 0.10,
    industry_gm_std: float | None = 0.02,
    quarterly: tuple[QuarterlyMargin, ...] = (),
    segments: tuple[SegmentBreakdown, ...] = (),
    top5: float | None = None,
    div_signals: int = 0,
) -> TickerFundamentals:
    return TickerFundamentals(
        symbol=symbol,
        industry_classification="Test Sub-Industry",
        annual=annual,
        quarterly_margins=quarterly,
        segments_history=segments,
        top5_customer_share=top5,
        diversification_attempt_signals=div_signals,
        industry_roic_3y_median=industry_roic,
        industry_gross_margin_3y_std=industry_gm_std,
    )


# ---------------------------------------------------------------------------
# §13 multi-segment 30% gate
# ---------------------------------------------------------------------------


def test_no_primary_segment_excludes_outright() -> None:
    """Conglomerate with 4 equal-sized segments → excluded before any
    axis evaluation runs.
    """
    seg = resolve_primary_segment(
        [
            Segment(name="A", revenue=25, share_of_total=0.25),
            Segment(name="B", revenue=25, share_of_total=0.25),
            Segment(name="C", revenue=25, share_of_total=0.25),
            Segment(name="D", revenue=25, share_of_total=0.25),
        ],
        fiscal_year=2024,
        source="stub",
    )
    funds = _funds()
    result = evaluate_ticker(funds, seg)
    assert result.hierarchy_decision == "REJECT"
    assert result.excluded_reason is not None
    assert "§13" in result.excluded_reason
    # Per-axis verdicts are FAIL with the §13 explanation.
    assert result.moat.verdict == "FAIL"
    assert result.new_frontier.verdict == "FAIL"
    assert result.bottleneck.verdict == "FAIL"


def test_constitution_version_is_stamped() -> None:
    """Every result must record the rubric version it ran under (§13)."""
    funds = _funds()
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.constitution_version == CONSTITUTION_VERSION


# ---------------------------------------------------------------------------
# Moat axis paths
# ---------------------------------------------------------------------------


def test_moat_fail_when_no_roic_history() -> None:
    """Constitution Auto-PASS 1: ROIC data spans <3 years."""
    funds = _funds(
        annual=(),  # no annual rows at all
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.moat.verdict == "FAIL"
    assert "auto-PASS 1" in result.moat.reason


def test_moat_fail_when_advantage_below_5pp() -> None:
    """ROIC advantage 3pp < 5pp threshold → outright FAIL."""
    funds = _funds(
        annual=(
            _annual(2022, nopat=12, capital=100),  # ROIC 0.12
            _annual(2023, nopat=13, capital=100),
            _annual(2024, nopat=14, capital=100),
        ),
        industry_roic=0.10,
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.moat.verdict == "FAIL"
    assert "ROIC advantage" in result.moat.reason


def test_moat_fail_when_advantage_eroding_fast() -> None:
    """Even with strong absolute advantage, fast erosion → FAIL."""
    funds = _funds(
        annual=(
            _annual(2022, nopat=30, capital=100),  # 0.30 advantage 0.20
            _annual(2023, nopat=20, capital=100),  # 0.20 advantage 0.10
            _annual(2024, nopat=10, capital=100),  # 0.10 advantage 0.00
        ),
        industry_roic=0.10,
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.moat.verdict == "FAIL"
    assert "eroding" in result.moat.reason


def test_moat_need_llm_when_quant_clean_but_structural_reason_unverified() -> None:
    """Strong, durable ROIC advantage → quant green, but structural
    reason is qualitative — must go to Stage 3."""
    funds = _funds(
        annual=(
            _annual(2022, nopat=20, capital=100),
            _annual(2023, nopat=21, capital=100),
            _annual(2024, nopat=22, capital=100),
        ),
        industry_roic=0.10,
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.moat.verdict == "NEED_LLM"
    assert "Stage 3" in result.moat.reason


def test_moat_need_llm_when_industry_median_unavailable() -> None:
    funds = _funds(
        annual=(
            _annual(2022, nopat=20, capital=100),
            _annual(2023, nopat=22, capital=100),
            _annual(2024, nopat=24, capital=100),
        ),
        industry_roic=None,
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.moat.verdict == "NEED_LLM"


# ---------------------------------------------------------------------------
# New Frontier axis paths
# ---------------------------------------------------------------------------


def test_frontier_fail_when_paradigm_too_recent() -> None:
    """<3 years since paradigm intro → constitution Auto-PASS 4."""
    history = (
        SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Bet",
            primary_segment_revenue_share=1.0,
            all_segments=(Segment(name="Bet", revenue=None, share_of_total=1.0),),
            fiscal_year=2024,  # only 1 year of history
            source="stub",
        ),
    )
    funds = _funds(segments=history)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.new_frontier.verdict == "FAIL"
    assert "auto-PASS 4" in result.new_frontier.reason


def test_frontier_need_llm_when_old_enough() -> None:
    """≥3 years passed → Stage 3 must check imitation evidence."""
    history = (
        SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Original",
            primary_segment_revenue_share=1.0,
            all_segments=(Segment(name="Original", revenue=None, share_of_total=1.0),),
            fiscal_year=2018,
            source="stub",
        ),
        SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Original",
            primary_segment_revenue_share=0.80,
            all_segments=(
                Segment(name="Original", revenue=None, share_of_total=0.80),
                Segment(name="New", revenue=None, share_of_total=0.20),
            ),
            fiscal_year=2024,
            source="stub",
        ),
    )
    funds = _funds(segments=history)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.new_frontier.verdict == "NEED_LLM"


# ---------------------------------------------------------------------------
# Bottleneck axis paths
# ---------------------------------------------------------------------------


def test_bottleneck_below_threshold_still_need_llm() -> None:
    """Calibration finding (#5, 2026-04): Stage 2 must NOT auto-FAIL
    on top-5 < 40%. Constitution §12 condition 1 has two alternative
    paths (5x downstream OR 40% top-5), and the threshold check on
    path B is necessary but not sufficient — Risk Factors materiality
    is also required (LLM-only). top-5 < 40% just means path B is
    closed; Stage 3 must still verify path A before concluding."""
    funds = _funds(top5=0.30)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.bottleneck.verdict == "NEED_LLM"
    # The reason should explicitly tell Stage 3 that path B is not
    # met but path A is still open.
    assert "1-A" in result.bottleneck.reason or "5×" in result.bottleneck.reason


def test_bottleneck_need_llm_when_top5_clears_threshold() -> None:
    """Quant side OK; replacement difficulty is qualitative."""
    funds = _funds(top5=0.50)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.bottleneck.verdict == "NEED_LLM"


def test_bottleneck_need_llm_when_diversification_signals_present() -> None:
    """Constitution Auto-PASS 4 trigger — defer to Stage 3 for judgment."""
    funds = _funds(top5=0.50, div_signals=1)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.bottleneck.verdict == "NEED_LLM"
    assert "diversification" in result.bottleneck.reason


def test_bottleneck_need_llm_when_concentration_undisclosed() -> None:
    funds = _funds(top5=None)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.bottleneck.verdict == "NEED_LLM"
    assert "undisclosed" in result.bottleneck.reason


# ---------------------------------------------------------------------------
# Hierarchy gate (§9)
# ---------------------------------------------------------------------------


def test_hierarchy_advances_when_two_axes_potential_with_growth() -> None:
    """Quant moat + quant bottleneck both potentially passing →
    growth axis present → ADVANCE_TO_STAGE_3.
    """
    funds = _funds(
        annual=(
            _annual(2022, nopat=20, capital=100),
            _annual(2023, nopat=21, capital=100),
            _annual(2024, nopat=22, capital=100),
        ),
        industry_roic=0.10,
        top5=0.50,  # bottleneck NEED_LLM
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.hierarchy_decision == "ADVANCE_TO_STAGE_3"
    assert "moat" in result.need_llm_axes
    assert "bottleneck" in result.need_llm_axes


def _make_axis_verdict(axis: str, verdict: str) -> AxisVerdict:
    """Hand-build an AxisVerdict for hierarchy-gate unit tests.

    Calibration finding (#5, 2026-04): bottleneck-axis Stage 2 verdict
    can no longer be FAIL by design — every code path emits NEED_LLM
    per constitution §12. Tests that previously forced bottleneck FAIL
    via top-5 manipulation can't replicate that scenario through
    `evaluate_ticker` anymore. Use this helper to feed the hierarchy
    gate directly with the verdict shape we want to test.
    """
    return AxisVerdict(axis=axis, verdict=verdict, reason="test", details={})


def test_hierarchy_rejects_when_no_growth_axis() -> None:
    """Constitution §9: a growth axis (new_frontier OR bottleneck) must
    clear the gate. moat alone cannot advance even if it's a potential
    pass."""
    moat = _make_axis_verdict("moat", "NEED_LLM")
    frontier = _make_axis_verdict("new_frontier", "FAIL")
    bottleneck = _make_axis_verdict("bottleneck", "FAIL")
    decision, _passed, _need_llm, reason = _apply_hierarchy_gate(
        moat, frontier, bottleneck
    )
    assert decision == "REJECT"
    assert reason is not None
    assert "growth axis" in reason


def test_hierarchy_rejects_when_only_one_axis_potential() -> None:
    """Constitution §9: 2+ axes must clear the gate. A single potential
    pass — even if that axis is a growth axis — is insufficient."""
    moat = _make_axis_verdict("moat", "FAIL")
    frontier = _make_axis_verdict("new_frontier", "NEED_LLM")
    bottleneck = _make_axis_verdict("bottleneck", "FAIL")
    decision, _passed, _need_llm, reason = _apply_hierarchy_gate(
        moat, frontier, bottleneck
    )
    assert decision == "REJECT"
    assert reason is not None
    assert "only 1" in reason or "<2" in reason or "need 2+" in reason


def test_hierarchy_advances_with_two_axes_including_growth() -> None:
    """Constitution §9 boundary: minimum-viable advance — moat + one
    growth axis (frontier OR bottleneck) with NEED_LLM each."""
    moat = _make_axis_verdict("moat", "NEED_LLM")
    frontier = _make_axis_verdict("new_frontier", "NEED_LLM")
    bottleneck = _make_axis_verdict("bottleneck", "FAIL")
    decision, _passed, _need_llm, _reason = _apply_hierarchy_gate(
        moat, frontier, bottleneck
    )
    assert decision == "ADVANCE_TO_STAGE_3"


def test_hierarchy_rejects_when_only_moat_passes() -> None:
    """Constitution §9: even moat=PASS alone fails the gate without
    a growth-axis companion. Guards against future drift toward
    moat-as-sufficient."""
    moat = _make_axis_verdict("moat", "PASS")
    frontier = _make_axis_verdict("new_frontier", "FAIL")
    bottleneck = _make_axis_verdict("bottleneck", "FAIL")
    decision, _passed, _need_llm, reason = _apply_hierarchy_gate(
        moat, frontier, bottleneck
    )
    assert decision == "REJECT"
    assert reason is not None
    assert "growth axis" in reason


# ---------------------------------------------------------------------------
# Constants exposed at module level — calibration may override
# ---------------------------------------------------------------------------


def test_thresholds_match_constitution() -> None:
    """If these drift, the tests are wrong, not the code — fix the
    constitution v3 and update threshold + test together.
    """
    assert MOAT_ROIC_ADVANTAGE_MIN == 0.05
    assert FRONTIER_MIN_YEARS_SINCE_INTRO == 3
    assert BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN == 0.40
