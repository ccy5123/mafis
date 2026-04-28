"""Pure proxy-computation tests (constitution §15)."""

from __future__ import annotations

from wise_investor.screening.proxies import (
    _hhi_from_share_distribution,
    _ols_slope,
    compute_bottleneck_proxies,
    compute_frontier_proxies,
    compute_moat_proxies,
)
from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    Segment,
    SegmentBreakdown,
    TickerFundamentals,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


def _qm(qid: str, gm: float | None) -> QuarterlyMargin:
    return QuarterlyMargin(quarter_id=qid, gross_margin=gm)


def _make_funds(
    *,
    annual: tuple[AnnualFinancials, ...] = (),
    quarterly: tuple[QuarterlyMargin, ...] = (),
    segments: tuple[SegmentBreakdown, ...] = (),
    industry_roic: float | None = 0.10,
    industry_gm_std: float | None = 0.02,
    top5: float | None = None,
    div_signals: int = 0,
) -> TickerFundamentals:
    return TickerFundamentals(
        symbol="TEST",
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
# Helper unit tests
# ---------------------------------------------------------------------------


def test_ols_slope_known_line() -> None:
    """y = 2x: slope must be exactly 2."""
    points = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
    assert _ols_slope(points) == 2.0


def test_ols_slope_constant_x_returns_none() -> None:
    points = [(1.0, 5.0), (1.0, 7.0)]
    assert _ols_slope(points) is None


def test_ols_slope_too_few_points_returns_none() -> None:
    assert _ols_slope([(1.0, 2.0)]) is None
    assert _ols_slope([]) is None


def test_hhi_from_uniform_top5_share() -> None:
    """Top-5 share of 50% spread evenly = HHI 500 (5 × 10²)."""
    out = _hhi_from_share_distribution([0.10] * 5)
    assert out == 500


def test_hhi_empty_returns_none() -> None:
    assert _hhi_from_share_distribution([]) is None


# ---------------------------------------------------------------------------
# Moat proxies
# ---------------------------------------------------------------------------


def test_moat_roic_avg_uses_recent_three_years() -> None:
    """When given five years of data, the avg uses the latest three."""
    funds = _make_funds(
        annual=(
            _annual(2020, nopat=10, capital=100),  # ROIC 0.10
            _annual(2021, nopat=12, capital=100),  # ROIC 0.12
            _annual(2022, nopat=14, capital=100),  # ROIC 0.14
            _annual(2023, nopat=16, capital=100),  # ROIC 0.16
            _annual(2024, nopat=18, capital=100),  # ROIC 0.18
        ),
    )
    out = compute_moat_proxies(funds)
    # Recent three are 2022, 2023, 2024 → avg of 0.14, 0.16, 0.18 = 0.16
    assert out.roic_3y_avg is not None
    assert abs(out.roic_3y_avg - 0.16) < 1e-9


def test_moat_roic_advantage_above_threshold() -> None:
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=20, capital=100),  # 0.20
            _annual(2023, nopat=22, capital=100),  # 0.22
            _annual(2024, nopat=24, capital=100),  # 0.24
        ),
        industry_roic=0.10,
    )
    out = compute_moat_proxies(funds)
    # avg 0.22, advantage 0.12 — well above the 5pp threshold.
    assert out.roic_advantage is not None
    assert abs(out.roic_advantage - 0.12) < 1e-9


def test_moat_advantage_trend_positive_when_roic_growing() -> None:
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=15, capital=100),  # 0.15
            _annual(2023, nopat=18, capital=100),  # 0.18
            _annual(2024, nopat=21, capital=100),  # 0.21
        ),
        industry_roic=0.10,
    )
    out = compute_moat_proxies(funds)
    # Advantage grew 0.05 → 0.08 → 0.11 → slope +0.03/yr.
    assert out.roic_advantage_trend is not None
    assert out.roic_advantage_trend > 0


def test_moat_advantage_trend_negative_when_eroding() -> None:
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=25, capital=100),  # 0.25
            _annual(2023, nopat=18, capital=100),  # 0.18
            _annual(2024, nopat=11, capital=100),  # 0.11
        ),
        industry_roic=0.10,
    )
    out = compute_moat_proxies(funds)
    # Eroding fast — slope sharply negative.
    assert out.roic_advantage_trend is not None
    assert out.roic_advantage_trend < -0.05


def test_moat_handles_missing_invested_capital() -> None:
    """A row with None capital is skipped, not crashed on. Per
    constitution §10 Auto-PASS 1, fewer than 3 valid years yields
    roic_3y_avg=None so the prefilter can fire the explicit FAIL.
    """
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=10, capital=None),  # skip
            _annual(2023, nopat=12, capital=100),
            _annual(2024, nopat=14, capital=100),
        ),
        industry_roic=0.10,
    )
    out = compute_moat_proxies(funds)
    # Only 2 valid rows → constitution §10 says insufficient history.
    assert out.roic_3y_avg is None


def test_moat_three_valid_years_yields_avg() -> None:
    """When 3 valid ROIC rows exist, the average is computed
    normally — covers the threshold above the §10 Auto-PASS 1.
    """
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=10, capital=100),  # ROIC 0.10
            _annual(2023, nopat=12, capital=100),  # ROIC 0.12
            _annual(2024, nopat=14, capital=100),  # ROIC 0.14
        ),
        industry_roic=0.10,
    )
    out = compute_moat_proxies(funds)
    assert out.roic_3y_avg is not None
    assert abs(out.roic_3y_avg - 0.12) < 1e-9


def test_moat_no_industry_median_yields_none_advantage() -> None:
    """With 3+ years of ROIC history but no industry median, we
    can compute the absolute ROIC but not the advantage.
    """
    funds = _make_funds(
        annual=(
            _annual(2022, nopat=18, capital=100),
            _annual(2023, nopat=20, capital=100),
            _annual(2024, nopat=22, capital=100),
        ),
        industry_roic=None,
    )
    out = compute_moat_proxies(funds)
    assert out.roic_3y_avg is not None
    assert abs(out.roic_3y_avg - 0.20) < 1e-9
    assert out.roic_advantage is None
    assert out.roic_advantage_trend is None


def test_moat_gross_margin_std_computed_when_enough_quarters() -> None:
    funds = _make_funds(
        quarterly=(
            _qm("2024Q1", 0.50),
            _qm("2024Q2", 0.52),
            _qm("2024Q3", 0.51),
            _qm("2024Q4", 0.49),
        ),
        industry_gm_std=0.02,
    )
    out = compute_moat_proxies(funds)
    assert out.gross_margin_3y_std is not None
    assert out.gross_margin_3y_std > 0
    # Ratio relative to industry std should be sensible.
    assert out.gross_margin_industry_ratio is not None


def test_moat_gross_margin_std_none_when_too_few_quarters() -> None:
    funds = _make_funds(
        quarterly=(_qm("2024Q4", 0.50),),
    )
    out = compute_moat_proxies(funds)
    assert out.gross_margin_3y_std is None


# ---------------------------------------------------------------------------
# Frontier proxies
# ---------------------------------------------------------------------------


def test_frontier_years_since_intro_simple_case() -> None:
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
                Segment(name="New Bet", revenue=None, share_of_total=0.20),
            ),
            fiscal_year=2024,
            source="stub",
        ),
    )
    funds = _make_funds(segments=history)
    out = compute_frontier_proxies(funds)
    # 2018 to 2024 → 6 years.
    assert out.years_since_first_segment_introduction == 6


def test_frontier_new_segments_added_in_last_5y() -> None:
    history = (
        SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Original",
            primary_segment_revenue_share=1.0,
            all_segments=(Segment(name="Original", revenue=None, share_of_total=1.0),),
            fiscal_year=2015,
            source="stub",
        ),
        SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name="Original",
            primary_segment_revenue_share=0.50,
            all_segments=(
                Segment(name="Original", revenue=None, share_of_total=0.50),
                Segment(name="New Bet", revenue=None, share_of_total=0.30),
                Segment(name="Newer", revenue=None, share_of_total=0.20),
            ),
            fiscal_year=2024,
            source="stub",
        ),
    )
    funds = _make_funds(segments=history)
    out = compute_frontier_proxies(funds)
    # 2 new segments after cutoff (2024-5 = 2019): "New Bet", "Newer"
    assert out.new_segments_added_5y == 2


def test_frontier_no_history_returns_nones() -> None:
    funds = _make_funds()
    out = compute_frontier_proxies(funds)
    assert out.years_since_first_segment_introduction is None
    assert out.new_segments_added_5y is None


def test_frontier_single_segment_default_treated_as_no_data() -> None:
    """Calibration finding (#1, 2026-04): every adapter falls back to
    single_segment_default(fiscal_year=latest) when no real segment
    table is available. The previous code treated that length-1
    history as 1-year-of-real-data → years_since=0 → §11 auto-FAIL,
    which auto-FAILed all 17/17 manifest tickers in the first
    calibration ledger. The fix is to recognize the fallback and
    return None for both fields so _evaluate_frontier routes the
    axis to NEED_LLM (matching constitution §17 — "primarily
    LLM-driven")."""
    fallback = SegmentBreakdown(
        primary_segment_exists=True,
        primary_segment_name="ACME",
        primary_segment_revenue_share=1.0,
        all_segments=(Segment(name="ACME", revenue=None, share_of_total=1.0),),
        fiscal_year=2024,
        source="single_segment_default",
    )
    funds = _make_funds(segments=(fallback,))
    out = compute_frontier_proxies(funds)
    assert out.years_since_first_segment_introduction is None
    assert out.new_segments_added_5y is None


def test_frontier_real_single_segment_year_still_zero() -> None:
    """A NON-fallback single-year segment entry (e.g., source='dart'
    with one fiscal year of data) is still real data — years_since
    correctly reads as 0. This is a separate signal from the
    fallback case: the company genuinely has only 1 year of segment
    history, which §11's time gate should fail. The fix above must
    NOT swallow this case."""
    real_one_year = SegmentBreakdown(
        primary_segment_exists=True,
        primary_segment_name="Foundry",
        primary_segment_revenue_share=1.0,
        all_segments=(Segment(name="Foundry", revenue=None, share_of_total=1.0),),
        fiscal_year=2024,
        source="dart",  # NOT single_segment_default
    )
    funds = _make_funds(segments=(real_one_year,))
    out = compute_frontier_proxies(funds)
    assert out.years_since_first_segment_introduction == 0
    assert out.new_segments_added_5y == 1  # 1 new segment within the 5y cutoff


# ---------------------------------------------------------------------------
# Bottleneck proxies
# ---------------------------------------------------------------------------


def test_bottleneck_pass_through_top5() -> None:
    funds = _make_funds(top5=0.55)
    out = compute_bottleneck_proxies(funds)
    assert out.top5_customer_share == 0.55
    # HHI computed as if top-5 is uniform → 5 × 11² = 605.
    assert out.hhi == 605


def test_bottleneck_no_top5_yields_no_hhi() -> None:
    funds = _make_funds(top5=None)
    out = compute_bottleneck_proxies(funds)
    assert out.top5_customer_share is None
    assert out.hhi is None


def test_bottleneck_diversification_signals_passthrough() -> None:
    funds = _make_funds(top5=0.50, div_signals=2)
    out = compute_bottleneck_proxies(funds)
    assert out.diversification_attempt_signals == 2
