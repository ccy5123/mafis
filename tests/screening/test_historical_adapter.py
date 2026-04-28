"""Historical fundamentals adapter tests.

yfinance is stubbed; tests exercise filtering, computation, and
caching behavior without network calls.
"""

from __future__ import annotations

import datetime as dt

from wise_investor.screening.historical_adapter import (
    FILING_LAG_DAYS,
    fetch_historical_fundamentals,
)


_DEFAULT_INFO = {"industry": "Test Sub-Industry"}


def _stub_fetcher_factory(income_dict, balance_dict, info=None):
    def _fetcher(symbol: str):
        # Use the passed `info` exactly when non-None (including {}),
        # otherwise default. The previous `info or default` pattern
        # treated an explicit empty dict as falsy and substituted the
        # default, which made it impossible for tests to assert
        # behavior on missing-info input.
        return {
            "income_stmt": income_dict,
            "balance_sheet": balance_dict,
            "info": info if info is not None else _DEFAULT_INFO,
        }
    return _fetcher


def _income_row(revenue, gross, operating, tax_rate=0.21, rd=None):
    return {
        "Total Revenue": revenue,
        "Gross Profit": gross,
        "Operating Income": operating,
        "Tax Rate For Calcs": tax_rate,
        "Research And Development": rd,
    }


def _balance_row(debt, equity, cash):
    return {
        "Total Debt": debt,
        "Stockholders Equity": equity,
        "Cash And Cash Equivalents": cash,
    }


# ---------------------------------------------------------------------------
# Filing-lag filter — only fundamentals public on as_of_date are returned
# ---------------------------------------------------------------------------


def test_filtering_includes_year_published_before_as_of_date(tmp_path) -> None:
    income = {
        "2017-12-31": _income_row(100, 60, 30),  # public Apr 2018
        "2018-12-31": _income_row(120, 75, 40),  # public Apr 2019
    }
    balance = {
        "2017-12-31": _balance_row(20, 80, 10),
        "2018-12-31": _balance_row(25, 95, 12),
    }
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    fiscal_years = [a.fiscal_year for a in funds.annual]
    # Both 2017 and 2018 are public by mid-2019 (90-day lag).
    assert 2017 in fiscal_years
    assert 2018 in fiscal_years


def test_filtering_excludes_unpublished_year(tmp_path) -> None:
    income = {
        "2018-12-31": _income_row(100, 60, 30),  # public Apr 2019
        "2019-12-31": _income_row(120, 75, 40),  # public Apr 2020 → exclude
    }
    balance = {
        "2018-12-31": _balance_row(20, 80, 10),
        "2019-12-31": _balance_row(25, 95, 12),
    }
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    fiscal_years = [a.fiscal_year for a in funds.annual]
    assert 2018 in fiscal_years
    assert 2019 not in fiscal_years


def test_filing_lag_is_90_days() -> None:
    """The constant should match the docstring claim — if calibration
    later finds 90 days too short / long, change it here and the
    docstring together.
    """
    assert FILING_LAG_DAYS == 90


def test_filtering_boundary_at_lag_day(tmp_path) -> None:
    """A FY ending 2018-12-31 + 90 days = 2019-03-31. The rule is
    "include only if fy_end + lag <= as_of_date", so on 2019-03-31
    the 2018 row is included (just barely); on 2019-03-30 it's not.
    """
    income = {"2018-12-31": _income_row(100, 60, 30)}
    balance = {"2018-12-31": _balance_row(20, 80, 10)}

    on_lag_day = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 3, 31),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    assert any(a.fiscal_year == 2018 for a in on_lag_day.annual)

    one_day_before = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 3, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    assert not any(a.fiscal_year == 2018 for a in one_day_before.annual)


# ---------------------------------------------------------------------------
# NOPAT / invested-capital computation
# ---------------------------------------------------------------------------


def test_nopat_computed_from_operating_income_and_tax_rate(tmp_path) -> None:
    income = {
        "2018-12-31": _income_row(
            revenue=1000, gross=600, operating=200, tax_rate=0.25
        ),
    }
    balance = {"2018-12-31": _balance_row(50, 200, 30)}
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    nopat = funds.annual[0].nopat
    assert nopat is not None
    assert abs(nopat - 200 * 0.75) < 1e-9


def test_invested_capital_is_debt_plus_equity_minus_cash(tmp_path) -> None:
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(debt=100, equity=300, cash=50)}
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    ic = funds.annual[0].invested_capital
    assert ic == 100 + 300 - 50  # 350


def test_missing_equity_yields_none_invested_capital(tmp_path) -> None:
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {
        "2018-12-31": {
            "Total Debt": 100,
            "Cash And Cash Equivalents": 50,
            # equity missing entirely
        }
    }
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    assert funds.annual[0].invested_capital is None


def test_missing_debt_treated_as_zero(tmp_path) -> None:
    """A debt-free company shouldn't lose its IC computation just
    because Yahoo's row didn't include a Total Debt entry.
    """
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {
        "2018-12-31": {
            "Stockholders Equity": 500,
            "Cash And Cash Equivalents": 100,
        }
    }
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    assert funds.annual[0].invested_capital == 500 - 100  # 400


# ---------------------------------------------------------------------------
# Industry classification + segment defaults
# ---------------------------------------------------------------------------


def test_industry_pulled_from_info_dict(tmp_path) -> None:
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(50, 200, 30)}
    funds = fetch_historical_fundamentals(
        "NVDA",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(
            income, balance, info={"industry": "Semiconductors"}
        ),
    )
    assert funds.industry_classification == "Semiconductors"


def test_unknown_industry_when_info_missing(tmp_path) -> None:
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(50, 200, 30)}
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance, info={}),
    )
    assert funds.industry_classification == "Unknown"


def test_default_segment_is_single_100_pct(tmp_path) -> None:
    """yfinance doesn't surface segment data — fall back to single
    segment so the §13 30% rule doesn't accidentally exclude every
    ticker we back-validate.
    """
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(50, 200, 30)}
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory(income, balance),
    )
    assert len(funds.segments_history) == 1
    assert funds.segments_history[0].primary_segment_exists is True
    assert funds.segments_history[0].primary_segment_revenue_share == 1.0


# ---------------------------------------------------------------------------
# Cache layer
# ---------------------------------------------------------------------------


def test_cache_roundtrip(tmp_path, monkeypatch) -> None:
    """Write to cache, then read back without invoking the fetcher."""
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(50, 200, 30)}

    # Redirect cache to tmp_path so we don't pollute repo data/.
    from wise_investor.screening import historical_adapter as ha

    monkeypatch.setattr(ha, "HISTORICAL_CACHE_DIR", tmp_path / "cache")

    fetch_count = {"n": 0}

    def _counting_fetcher(symbol: str):
        fetch_count["n"] += 1
        return {
            "income_stmt": income,
            "balance_sheet": balance,
            "info": {"industry": "X"},
        }

    funds_first = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=True,
        fetcher=_counting_fetcher,
    )
    funds_second = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=True,
        fetcher=_counting_fetcher,
    )
    assert fetch_count["n"] == 1  # only fetched once
    assert funds_first.industry_classification == funds_second.industry_classification
    assert len(funds_first.annual) == len(funds_second.annual)


def test_cache_disabled_always_fetches(tmp_path, monkeypatch) -> None:
    income = {"2018-12-31": _income_row(1000, 600, 200)}
    balance = {"2018-12-31": _balance_row(50, 200, 30)}

    from wise_investor.screening import historical_adapter as ha

    monkeypatch.setattr(ha, "HISTORICAL_CACHE_DIR", tmp_path / "cache")

    fetch_count = {"n": 0}

    def _counting_fetcher(symbol: str):
        fetch_count["n"] += 1
        return {"income_stmt": income, "balance_sheet": balance, "info": {}}

    fetch_historical_fundamentals(
        "TEST", dt.date(2019, 6, 30), cache=False, fetcher=_counting_fetcher
    )
    fetch_historical_fundamentals(
        "TEST", dt.date(2019, 6, 30), cache=False, fetcher=_counting_fetcher
    )
    assert fetch_count["n"] == 2


def test_empty_data_yields_empty_annual(tmp_path) -> None:
    funds = fetch_historical_fundamentals(
        "TEST",
        as_of_date=dt.date(2019, 6, 30),
        cache=False,
        fetcher=_stub_fetcher_factory({}, {}),
    )
    assert funds.annual == ()
    assert funds.segments_history == ()  # empty annual → empty history default
