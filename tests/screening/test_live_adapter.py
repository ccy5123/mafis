"""Live fundamentals adapter tests.

The Finnhub client is fully stubbed: every test passes a duck-typed
fake whose `financials()`/`profile()` mirror the real client's shape
but return canned data. No network calls.

The stubs only need:
  - `entry.year`, `entry.quarter`, `entry.report.ic`, `entry.report.bs`,
    `entry.report.cf` (each a list of `.concept`/`.value` items)
  - `profile.finnhub_industry`
  - `response.data` (list of entries)

This is the same shape the real `FinnhubClient` returns via pydantic;
`extract_field()` and `total_debt()` from `wise_investor.data.finnhub`
are pure functions that walk those attributes, so they work over the
stubs unchanged.
"""

from __future__ import annotations

import pytest

from wise_investor.screening.live_adapter import (
    DEFAULT_EFFECTIVE_TAX_RATE,
    IndustryAggregates,
    fetch_live_fundamentals,
    fetch_live_universe,
)

# ---------------------------------------------------------------------------
# Stub builders mimicking Finnhub's pydantic models (duck-typed)
# ---------------------------------------------------------------------------


class _Item:
    def __init__(self, concept: str, value: float | None) -> None:
        self.concept = concept
        self.value = value


class _Report:
    def __init__(self, ic=(), bs=(), cf=()) -> None:
        self.ic = list(ic)
        self.bs = list(bs)
        self.cf = list(cf)


class _Entry:
    def __init__(self, year, *, quarter=None, form="10-K", report=None) -> None:
        self.year = year
        self.quarter = quarter
        self.form = form
        self.report = report or _Report()


class _Resp:
    def __init__(self, data) -> None:
        self.data = data


class _Profile:
    def __init__(self, industry: str | None) -> None:
        self.finnhub_industry = industry


class _StubClient:
    """Default-success client; tests subclass to inject failures."""

    def __init__(
        self,
        *,
        annual=None,
        quarterly=None,
        profile_industry: str | None = "Test Sub-Industry",
    ) -> None:
        self.annual_resp = _Resp(annual or [])
        self.quarterly_resp = _Resp(quarterly or [])
        self.profile_obj = _Profile(profile_industry)
        self.calls: list = []

    def financials(self, symbol: str, freq: str = "annual"):
        self.calls.append(("financials", symbol, freq))
        return self.quarterly_resp if freq == "quarterly" else self.annual_resp

    def profile(self, symbol: str):
        self.calls.append(("profile", symbol))
        return self.profile_obj


def _annual(year, *, revenue, gross, operating, debt, equity, cash) -> _Entry:
    """A canonical annual filing with all balance-sheet/income items the
    extractor looks for, on the most-common XBRL concepts."""
    ic = [
        _Item("us-gaap_Revenues", revenue),
        _Item("us-gaap_GrossProfit", gross),
        _Item("us-gaap_OperatingIncomeLoss", operating),
    ]
    bs = [
        _Item("us-gaap_LongTermDebt", debt),
        _Item("us-gaap_StockholdersEquity", equity),
        _Item("us-gaap_CashAndCashEquivalentsAtCarryingValue", cash),
    ]
    return _Entry(year=year, form="10-K", report=_Report(ic=ic, bs=bs))


def _quarter(year, q, *, revenue, gross) -> _Entry:
    ic = [
        _Item("us-gaap_Revenues", revenue),
        _Item("us-gaap_GrossProfit", gross),
    ]
    return _Entry(year=year, quarter=q, form="10-Q", report=_Report(ic=ic))


# ---------------------------------------------------------------------------
# Symbol normalization + basic shape
# ---------------------------------------------------------------------------


def test_symbol_uppercased() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=1000, gross=600, operating=300, debt=50, equity=400, cash=100),
    ])
    funds = fetch_live_fundamentals("nvda", client=client)
    assert funds.symbol == "NVDA"


def test_annual_sorted_oldest_first() -> None:
    """`AnnualFinancials` contract: newest fiscal year LAST."""
    client = _StubClient(annual=[
        _annual(2024, revenue=1000, gross=600, operating=300, debt=50, equity=400, cash=100),
        _annual(2022, revenue=800, gross=480, operating=240, debt=50, equity=300, cash=80),
        _annual(2023, revenue=900, gross=540, operating=270, debt=50, equity=350, cash=90),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert [a.fiscal_year for a in funds.annual] == [2022, 2023, 2024]


# ---------------------------------------------------------------------------
# NOPAT and invested capital — same definitions as historical_adapter
# ---------------------------------------------------------------------------


def test_nopat_uses_default_tax_rate() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=1000, gross=600, operating=200, debt=0, equity=300, cash=50),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual[0].nopat == pytest.approx(
        200 * (1.0 - DEFAULT_EFFECTIVE_TAX_RATE)
    )


def test_nopat_uses_custom_tax_rate() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=1000, gross=600, operating=200, debt=0, equity=300, cash=50),
    ])
    funds = fetch_live_fundamentals(
        "TEST", client=client, effective_tax_rate=0.30
    )
    assert funds.annual[0].nopat == pytest.approx(200 * 0.70)


def test_invested_capital_is_debt_plus_equity_minus_cash() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=1000, gross=600, operating=200, debt=100, equity=400, cash=50),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual[0].invested_capital == 100 + 400 - 50


def test_missing_equity_yields_none_invested_capital() -> None:
    bs = [_Item("us-gaap_LongTermDebt", 100)]
    entry = _Entry(year=2024, report=_Report(
        ic=[_Item("us-gaap_OperatingIncomeLoss", 200)],
        bs=bs,
    ))
    client = _StubClient(annual=[entry])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual[0].invested_capital is None


def test_missing_debt_treated_as_zero() -> None:
    """Debt-free company shouldn't lose its IC just because Finnhub's
    filing didn't carry a Long-term-debt entry."""
    bs = [
        _Item("us-gaap_StockholdersEquity", 500),
        _Item("us-gaap_CashAndCashEquivalentsAtCarryingValue", 100),
    ]
    entry = _Entry(year=2024, report=_Report(
        ic=[_Item("us-gaap_OperatingIncomeLoss", 200)],
        bs=bs,
    ))
    client = _StubClient(annual=[entry])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual[0].invested_capital == 500 - 100


def test_no_operating_income_yields_none_nopat() -> None:
    bs = [_Item("us-gaap_StockholdersEquity", 200), _Item("us-gaap_CashAndCashEquivalentsAtCarryingValue", 50)]
    entry = _Entry(year=2024, report=_Report(ic=[], bs=bs))
    client = _StubClient(annual=[entry])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual[0].nopat is None


# ---------------------------------------------------------------------------
# Industry classification
# ---------------------------------------------------------------------------


def test_industry_pulled_from_profile() -> None:
    client = _StubClient(
        annual=[_annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0)],
        profile_industry="Semiconductors",
    )
    funds = fetch_live_fundamentals("NVDA", client=client)
    assert funds.industry_classification == "Semiconductors"


def test_unknown_industry_when_profile_missing() -> None:
    class _NoProfile(_StubClient):
        def profile(self, symbol):
            raise RuntimeError("404")

    client = _NoProfile(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.industry_classification == "Unknown"


def test_unknown_industry_when_profile_field_is_none() -> None:
    client = _StubClient(
        annual=[_annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0)],
        profile_industry=None,
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.industry_classification == "Unknown"


# ---------------------------------------------------------------------------
# Segments default + 10-K-RAG-driven fields stay None / 0
# ---------------------------------------------------------------------------


def test_segments_history_falls_back_to_single_segment() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.segments_history) == 1
    seg = funds.segments_history[0]
    assert seg.primary_segment_exists is True
    assert seg.primary_segment_revenue_share == 1.0
    assert seg.fiscal_year == 2024


def test_top5_and_diversification_default_to_none_zero() -> None:
    """Per Commitment 3: live mode doesn't fabricate values for fields
    that need 10-K RAG. The prefilter routes the missing data through
    NEED_LLM rather than PASS, preserving precision over recall.
    """
    client = _StubClient(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.top5_customer_share is None
    assert funds.diversification_attempt_signals == 0


# ---------------------------------------------------------------------------
# Industry aggregates pass-through
# ---------------------------------------------------------------------------


def test_industry_aggregates_passed_through_when_supplied() -> None:
    aggs = IndustryAggregates(
        industry_roic_3y_median=0.15,
        industry_gross_margin_3y_std=0.03,
    )
    client = _StubClient(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    funds = fetch_live_fundamentals(
        "TEST", client=client, industry_aggregates=aggs
    )
    assert funds.industry_roic_3y_median == 0.15
    assert funds.industry_gross_margin_3y_std == 0.03


def test_industry_aggregates_default_to_none() -> None:
    client = _StubClient(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.industry_roic_3y_median is None
    assert funds.industry_gross_margin_3y_std is None


# ---------------------------------------------------------------------------
# Quarterly margins
# ---------------------------------------------------------------------------


def test_quarterly_margins_computed() -> None:
    client = _StubClient(
        annual=[
            _annual(2024, revenue=1000, gross=500, operating=200, debt=0, equity=300, cash=50),
        ],
        quarterly=[
            _quarter(2024, 1, revenue=250, gross=125),  # 50% GM
            _quarter(2024, 2, revenue=260, gross=130),
            _quarter(2024, 3, revenue=240, gross=120),
        ],
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.quarterly_margins) == 3
    assert all(abs(qm.gross_margin - 0.5) < 1e-9 for qm in funds.quarterly_margins)
    assert funds.quarterly_margins[0].quarter_id == "2024Q1"


def test_quarterly_with_zero_revenue_skipped() -> None:
    client = _StubClient(
        annual=[
            _annual(2024, revenue=1000, gross=500, operating=200, debt=0, equity=300, cash=50),
        ],
        quarterly=[
            _quarter(2024, 1, revenue=0, gross=0),  # skipped
            _quarter(2024, 2, revenue=250, gross=125),
        ],
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.quarterly_margins) == 1
    assert funds.quarterly_margins[0].quarter_id == "2024Q2"


def test_quarterly_with_missing_gross_profit_skipped() -> None:
    """Don't fabricate margin from missing gross profit."""
    entry = _Entry(year=2024, quarter=2, report=_Report(
        ic=[_Item("us-gaap_Revenues", 100)],  # no GrossProfit
    ))
    client = _StubClient(
        annual=[
            _annual(2024, revenue=1000, gross=500, operating=200, debt=0, equity=300, cash=50),
        ],
        quarterly=[entry],
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.quarterly_margins == ()


def test_quarterly_fetch_failure_yields_empty_quarterly() -> None:
    """Quarterly endpoint failures must NOT break the whole fetch."""
    class _AnnualOnly(_StubClient):
        def financials(self, symbol, freq="annual"):
            if freq == "quarterly":
                raise RuntimeError("data outage")
            return self.annual_resp

    client = _AnnualOnly(annual=[
        _annual(2024, revenue=1000, gross=500, operating=200, debt=0, equity=300, cash=50),
    ])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.annual) == 1
    assert funds.quarterly_margins == ()


def test_quarterly_capped_at_12() -> None:
    """The contract is 'last 12 quarters preferred'; any extras are dropped."""
    quarterly = [
        _quarter(2020 + (i // 4), (i % 4) + 1, revenue=100 + i, gross=50 + i)
        for i in range(15)
    ]
    client = _StubClient(
        annual=[
            _annual(2024, revenue=1000, gross=500, operating=200, debt=0, equity=300, cash=50),
        ],
        quarterly=quarterly,
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.quarterly_margins) == 12


# ---------------------------------------------------------------------------
# Empty / malformed inputs
# ---------------------------------------------------------------------------


def test_empty_annual_response_yields_empty_annual() -> None:
    client = _StubClient(annual=[])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert funds.annual == ()
    # Single-segment default still applies even with no annual; year=0 fallback.
    assert funds.segments_history[0].fiscal_year == 0


def test_entry_with_no_year_skipped() -> None:
    bad = _Entry(year=None, report=_Report(
        ic=[_Item("us-gaap_OperatingIncomeLoss", 200)],
    ))
    good = _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0)
    client = _StubClient(annual=[bad, good])
    funds = fetch_live_fundamentals("TEST", client=client)
    assert len(funds.annual) == 1
    assert funds.annual[0].fiscal_year == 2024


# ---------------------------------------------------------------------------
# Universe: fetch_live_universe error swallowing
# ---------------------------------------------------------------------------


def test_universe_swallows_per_ticker_exceptions() -> None:
    base_annual = [
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ]

    class _Flaky(_StubClient):
        def __init__(self):
            super().__init__(annual=base_annual)

        def financials(self, symbol, freq="annual"):
            if symbol == "BAD":
                raise RuntimeError("data outage")
            return super().financials(symbol, freq)

    client = _Flaky()
    out = fetch_live_universe(["GOOD", "BAD", "ALSOGOOD"], client=client)
    assert {f.symbol for f in out} == {"GOOD", "ALSOGOOD"}


def test_universe_uses_industry_aggregates_by_symbol() -> None:
    aggs_map = {
        "GOOD": IndustryAggregates(
            industry_roic_3y_median=0.15,
            industry_gross_margin_3y_std=0.03,
        ),
    }
    client = _StubClient(annual=[
        _annual(2024, revenue=100, gross=60, operating=30, debt=0, equity=50, cash=0),
    ])
    out = fetch_live_universe(
        ["GOOD", "OTHER"],
        client=client,
        industry_aggregates_by_symbol=aggs_map,
    )
    by_sym = {f.symbol: f for f in out}
    assert by_sym["GOOD"].industry_roic_3y_median == 0.15
    assert by_sym["OTHER"].industry_roic_3y_median is None


# ---------------------------------------------------------------------------
# Integration: live adapter output feeds the prefilter without crashing
# ---------------------------------------------------------------------------


def test_live_output_is_compatible_with_prefilter() -> None:
    """Smoke test: the shape produced by fetch_live_fundamentals must be
    consumable by evaluate_ticker without missing-attribute errors. This
    catches contract drift between the adapter and the screening pipeline.
    """
    from wise_investor.screening.prefilter import evaluate_ticker

    client = _StubClient(
        annual=[
            _annual(2022, revenue=1000, gross=600, operating=200, debt=50, equity=400, cash=100),
            _annual(2023, revenue=1100, gross=660, operating=220, debt=50, equity=420, cash=100),
            _annual(2024, revenue=1200, gross=720, operating=240, debt=50, equity=440, cash=100),
        ],
        quarterly=[
            _quarter(2024, 1, revenue=300, gross=180),
            _quarter(2024, 2, revenue=300, gross=180),
            _quarter(2024, 3, revenue=300, gross=180),
            _quarter(2024, 4, revenue=300, gross=180),
        ],
    )
    funds = fetch_live_fundamentals("TEST", client=client)
    primary = funds.segments_history[-1]
    result = evaluate_ticker(funds, primary)
    # No assertions on verdict here — the data is synthetic. The fact
    # that evaluate_ticker returned a PrefilterResult at all is the
    # invariant we're testing.
    assert result.symbol == "TEST"
    assert result.constitution_version  # non-empty
    assert result.hierarchy_decision in ("ADVANCE_TO_STAGE_3", "REJECT")
