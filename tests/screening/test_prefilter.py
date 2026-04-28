"""Stage 2 integration tests (constitution §16)."""

from __future__ import annotations

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.prefilter import (
    BOTTLENECK_TOP5_CUSTOMER_SHARE_MIN,
    FRONTIER_MIN_YEARS_SINCE_INTRO,
    MOAT_ROIC_ADVANTAGE_MIN,
    evaluate_ticker,
)
from wise_investor.screening.segments import (
    resolve_primary_segment,
    single_segment_default,
)
from wise_investor.screening.types import (
    AnnualFinancials,
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


def test_bottleneck_fail_when_top5_below_threshold() -> None:
    funds = _funds(top5=0.30)
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.bottleneck.verdict == "FAIL"
    assert "top-5 customer share" in result.bottleneck.reason


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


def test_hierarchy_rejects_when_no_growth_axis() -> None:
    """Moat NEED_LLM, but new_frontier and bottleneck both FAIL → no
    growth axis cleared the gate → REJECT.
    """
    funds = _funds(
        annual=(
            _annual(2022, nopat=20, capital=100),
            _annual(2023, nopat=21, capital=100),
            _annual(2024, nopat=22, capital=100),
        ),
        industry_roic=0.10,
        # No segment history → frontier NEED_LLM (still potential)
        # …but we need to FAIL frontier explicitly to test "no growth"
        segments=(
            SegmentBreakdown(
                primary_segment_exists=True,
                primary_segment_name="Single",
                primary_segment_revenue_share=1.0,
                all_segments=(
                    Segment(name="Single", revenue=None, share_of_total=1.0),
                ),
                fiscal_year=2024,  # only 1 year → FAIL
                source="stub",
            ),
        ),
        top5=0.10,  # below threshold → bottleneck FAIL
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.hierarchy_decision == "REJECT"
    assert result.excluded_reason is not None
    assert (
        "no growth axis" in result.excluded_reason
        or "<2 axis" in result.excluded_reason
        or "only" in result.excluded_reason
    )


def test_hierarchy_rejects_when_only_one_axis_potential() -> None:
    """Single axis potential pass → REJECT (need 2+)."""
    funds = _funds(
        annual=(
            _annual(2022, nopat=20, capital=100),
            _annual(2023, nopat=21, capital=100),
            _annual(2024, nopat=22, capital=100),
        ),
        industry_roic=0.10,
        # No segments → frontier NEED_LLM (potential pass)
        # No customer disclosure → bottleneck NEED_LLM (potential pass)
        # → 3 potential, growth present → would advance.
        # Force frontier and bottleneck to FAIL:
        segments=(
            SegmentBreakdown(
                primary_segment_exists=True,
                primary_segment_name="Solo",
                primary_segment_revenue_share=1.0,
                all_segments=(Segment(name="Solo", revenue=None, share_of_total=1.0),),
                fiscal_year=2024,
                source="stub",
            ),
        ),
        top5=0.05,  # very low → FAIL
    )
    seg = single_segment_default("TEST", fiscal_year=2024)
    result = evaluate_ticker(funds, seg)
    assert result.hierarchy_decision == "REJECT"


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
