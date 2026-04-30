"""DART-based Korean adapter + dispatcher tests.

The DartClient is fully stubbed: tests pass a duck-typed fake whose
`corp_code_from_stock_code()` and `financials()` mirror the real
client's shape but return canned data. No network calls.

Stub shape requirements:
  - Stub client exposes:
      - corp_code_from_stock_code(stock_code) → str | None
      - financials(corp_code, year, ...) → object with `.ok` and `.rows`
        (the rows have `.account_id`, `.account_nm`, `.sj_div`, and
        `.thstrm_amount` as comma-formatted strings or numbers — the
        adapter delegates to `extract_account_value()` from
        wise_investor.data.dart, which is a pure helper.)
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from wise_investor.data.dart import FinancialRow, FinancialsResponse
from wise_investor.screening.live_adapter import (
    fetch_live_fundamentals,
    fetch_live_universe,
)
from wise_investor.screening.live_adapter_kr import (
    DEFAULT_EFFECTIVE_TAX_RATE_KR,
    _normalize_kr_symbol,
    fetch_live_fundamentals_kr,
    is_korean_symbol,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


def _row(
    *,
    account_id: str | None = None,
    account_nm: str | None = None,
    sj_div: str,
    amount: float | None,
) -> FinancialRow:
    """Build a real FinancialRow so `extract_account_value` works on it
    unchanged. DART amounts are comma-string in the live API; pydantic
    accepts plain numbers too via this constructor.
    """
    return FinancialRow(
        account_id=account_id,
        account_nm=account_nm,
        sj_div=sj_div,
        thstrm_amount=str(amount) if amount is not None else None,
    )


def _samsung_like_response(
    *,
    revenue: float | None = 200_000_000_000_000,  # KRW 200 trillion
    gross_profit: float | None = 80_000_000_000_000,
    operating_income: float | None = 30_000_000_000_000,
    equity: float | None = 300_000_000_000_000,
    cash: float | None = 40_000_000_000_000,
    short_debt: float | None = 5_000_000_000_000,
    long_debt: float | None = 10_000_000_000_000,
) -> FinancialsResponse:
    """A canonical DART annual response for a profitable, large-cap KR
    industrial. Numbers are in KRW (won), unconverted. The adapter
    doesn't care about units — it computes ratios, not absolutes."""
    rows = []
    if revenue is not None:
        rows.append(_row(account_id="ifrs-full_Revenue", sj_div="IS", amount=revenue))
    if gross_profit is not None:
        rows.append(_row(account_id="ifrs-full_GrossProfit", sj_div="IS", amount=gross_profit))
    if operating_income is not None:
        rows.append(_row(account_id="dart_OperatingIncomeLoss", sj_div="IS", amount=operating_income))
    if equity is not None:
        rows.append(_row(account_id="ifrs-full_Equity", sj_div="BS", amount=equity))
    if cash is not None:
        rows.append(_row(account_id="ifrs-full_CashAndCashEquivalents", sj_div="BS", amount=cash))
    if short_debt is not None:
        rows.append(_row(account_id="ifrs-full_ShorttermBorrowings", sj_div="BS", amount=short_debt))
    if long_debt is not None:
        rows.append(_row(account_id="ifrs-full_LongtermBorrowings", sj_div="BS", amount=long_debt))
    return FinancialsResponse(status="000", message="정상", list=rows)


class _StubDartClient:
    """Default-success DART client; tests subclass to inject failures."""

    def __init__(
        self,
        *,
        corp_code: str | None = "00126380",
        responses: dict[int, FinancialsResponse] | None = None,
    ) -> None:
        self.corp_code_lookup = corp_code
        self.responses = responses or {}
        self.calls: list[Any] = []

    def corp_code_from_stock_code(self, stock_code: str):
        self.calls.append(("corp_code", stock_code))
        return self.corp_code_lookup

    def financials(self, corp_code: str, year, reprt_code: str = "11011", fs_div: str = "CFS"):
        self.calls.append(("financials", corp_code, year))
        if year in self.responses:
            return self.responses[year]
        # Default: a Samsung-like response for any year
        return _samsung_like_response()


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("005930", "005930"),
        ("005930.KS", "005930"),
        ("005930.KQ", "005930"),
        ("005930.KRX", "005930"),
        ("KRX:005930", "005930"),
        ("KS:005930", "005930"),
        ("5930", "005930"),       # zero-pad short code
        ("000660.KS", "000660"),
        ("207940.KS", "207940"),
    ],
)
def test_normalize_kr_symbol(raw: str, expected: str) -> None:
    assert _normalize_kr_symbol(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("005930", True),
        ("005930.KS", True),
        ("000660.KS", True),
        ("KRX:005930", True),
        ("5930", True),
        ("NVDA", False),
        ("BRK-B", False),
        ("AAPL.US", False),
        ("", False),
    ],
)
def test_is_korean_symbol(raw: str, expected: bool) -> None:
    assert is_korean_symbol(raw) is expected


# ---------------------------------------------------------------------------
# fetch_live_fundamentals_kr — symbol → corp_code → financials
# ---------------------------------------------------------------------------


def test_kr_corp_code_lookup_uses_normalized_stock_code() -> None:
    client = _StubDartClient()
    fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1, today=dt.date(2024, 6, 1),
    )
    # Stub records the lookup call with the 6-digit normalized form.
    assert ("corp_code", "005930") in client.calls


def test_kr_corp_code_not_found_raises() -> None:
    client = _StubDartClient(corp_code=None)
    with pytest.raises(ValueError, match="DART corp_code not found"):
        fetch_live_fundamentals_kr(
            "999999.KS", client=client, history_years=1,
            today=dt.date(2024, 6, 1),
        )


def test_kr_pulls_history_years_of_filings() -> None:
    client = _StubDartClient()
    fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=3,
        today=dt.date(2024, 6, 1),
    )
    # today=2024-06 → most_recent_fy = 2023, history = 2021..2023
    # P1c (2026-04): each fiscal year now triggers up to 4 reprt_code
    # fetches (annual + 3 quarterly cumulatives). Verify the *distinct*
    # years touched, not the raw call count.
    fetched_years = sorted({
        c[2] for c in client.calls if c[0] == "financials"
    })
    assert fetched_years == [2021, 2022, 2023]


def test_kr_annual_sorted_oldest_first() -> None:
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=3,
        today=dt.date(2024, 6, 1),
    )
    fiscal_years = [a.fiscal_year for a in funds.annual]
    assert fiscal_years == [2021, 2022, 2023]


def test_kr_skips_year_when_dart_returns_error() -> None:
    """Status != '000' → row is silently dropped, not raised."""
    bad_resp = FinancialsResponse(status="013", message="조회된 데이터가 없습니다.")
    client = _StubDartClient(responses={2023: bad_resp})
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=3,
        today=dt.date(2024, 6, 1),
    )
    fiscal_years = [a.fiscal_year for a in funds.annual]
    assert 2023 not in fiscal_years
    # 2021 and 2022 should still be there (default success response).
    assert 2021 in fiscal_years
    assert 2022 in fiscal_years


def test_kr_per_year_exception_does_not_break_run() -> None:
    class _FlakyForOneYear(_StubDartClient):
        def financials(self, corp_code, year, reprt_code="11011", fs_div="CFS"):
            if year == 2022:
                raise RuntimeError("DART 500 transient")
            return super().financials(corp_code, year)

    client = _FlakyForOneYear()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=3,
        today=dt.date(2024, 6, 1),
    )
    fiscal_years = [a.fiscal_year for a in funds.annual]
    assert fiscal_years == [2021, 2023]


# ---------------------------------------------------------------------------
# Account extraction: NOPAT, IC, fallbacks
# ---------------------------------------------------------------------------


def test_kr_nopat_uses_kr_default_tax_rate() -> None:
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    # operating_income 30T × (1 - 0.22) = 23.4T
    expected = 30_000_000_000_000 * (1.0 - DEFAULT_EFFECTIVE_TAX_RATE_KR)
    assert funds.annual[-1].nopat == pytest.approx(expected)


def test_kr_invested_capital_is_debt_plus_equity_minus_cash() -> None:
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    # short 5T + long 10T + equity 300T - cash 40T = 275T
    expected = 5e12 + 10e12 + 300e12 - 40e12
    assert funds.annual[-1].invested_capital == pytest.approx(expected)


def test_kr_missing_equity_yields_none_invested_capital() -> None:
    resp = _samsung_like_response(equity=None)
    client = _StubDartClient(responses={2023: resp})
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert funds.annual[-1].invested_capital is None


def test_kr_missing_both_debts_treated_as_no_debt() -> None:
    """When both short and long borrowings are absent, invested capital
    falls back to equity − cash (debt = 0)."""
    resp = _samsung_like_response(short_debt=None, long_debt=None)
    client = _StubDartClient(responses={2023: resp})
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    expected = 0 + 300e12 - 40e12  # 260T
    assert funds.annual[-1].invested_capital == pytest.approx(expected)


def test_kr_extracts_via_korean_account_name_fallback() -> None:
    """When XBRL account_id is missing but Korean account_nm is present,
    the adapter falls back to name-based extraction."""
    rows = [
        _row(account_nm="매출액", sj_div="IS", amount=100_000_000_000),
        _row(account_nm="영업이익", sj_div="IS", amount=20_000_000_000),
        _row(account_nm="자본총계", sj_div="BS", amount=500_000_000_000),
        _row(account_nm="현금및현금성자산", sj_div="BS", amount=50_000_000_000),
    ]
    resp = FinancialsResponse(status="000", list=rows)
    client = _StubDartClient(responses={2023: resp})
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert funds.annual[-1].revenue == 100_000_000_000
    assert funds.annual[-1].operating_income == 20_000_000_000


def test_kr_empty_response_yields_no_annual_row() -> None:
    """When DART returns rows but none match our account candidates,
    the annual entry is skipped (not a fabricated zero row)."""
    resp = FinancialsResponse(status="000", list=[
        _row(account_nm="기타비유동자산", sj_div="BS", amount=999),  # not in our table
    ])
    client = _StubDartClient(responses={2023: resp})
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert funds.annual == ()


# ---------------------------------------------------------------------------
# TickerFundamentals contract: shape + Commitment 3 honesty
# ---------------------------------------------------------------------------


def test_kr_industry_classification_is_static_dart_string() -> None:
    """DART doesn't expose GICS sub-industries; the adapter is honest
    about the limitation rather than fabricating one."""
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert funds.industry_classification == "Korean Equity (DART)"


def test_kr_quarterly_margins_recovered_from_dart_cumulatives() -> None:
    """P1c (2026-04): quarterly margins are no longer empty.

    With the default stub returning the same payload for every reprt_code,
    Q1 standalone equals the cumulative value while Q2/Q3/Q4 standalones
    subtract to zero (degenerate but mathematically correct for this
    stub). The first non-degenerate quarter is recovered as a sanity
    check that the helper actually runs.
    """
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert len(funds.quarterly_margins) >= 1
    # Q1 of the most-recent FY surfaces with a real GM; subsequent
    # quarters drop because Δ-revenue is 0 in this stub.
    assert funds.quarterly_margins[0].quarter_id == "2023Q1"
    assert 0.0 < funds.quarterly_margins[0].gross_margin <= 1.0


def test_kr_top5_and_diversification_default_to_none_zero() -> None:
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert funds.top5_customer_share is None
    assert funds.diversification_attempt_signals == 0


def test_kr_segments_history_falls_back_to_single_segment() -> None:
    client = _StubDartClient()
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    assert len(funds.segments_history) == 1
    assert funds.segments_history[0].primary_segment_exists is True
    assert funds.segments_history[0].primary_segment_revenue_share == 1.0


# ---------------------------------------------------------------------------
# Dispatcher (top-level fetch_live_fundamentals)
# ---------------------------------------------------------------------------


def test_dispatcher_routes_kr_symbol_to_dart() -> None:
    """A .KS symbol with only a dart_client must succeed; the dispatcher
    should not even try to instantiate FinnhubClient."""
    dart_client = _StubDartClient()
    funds = fetch_live_fundamentals(
        "005930.KS",
        dart_client=dart_client,
    )
    assert funds.industry_classification == "Korean Equity (DART)"
    assert any(c[0] == "financials" for c in dart_client.calls)


def test_dispatcher_routes_us_symbol_to_finnhub() -> None:
    """A US symbol with only a finnhub_client must succeed."""
    # Reuse the Finnhub stub style from test_live_adapter — duck-typed.
    class _F:
        def __init__(self):
            self.financials_calls = 0

        def financials(self, symbol, freq="annual"):
            self.financials_calls += 1
            class _R:
                data: list = []
            return _R()

        def profile(self, symbol):
            class _P:
                finnhub_industry = "Test Industry"
            return _P()

    finnhub = _F()
    funds = fetch_live_fundamentals(
        "NVDA",
        finnhub_client=finnhub,
    )
    assert funds.symbol == "NVDA"
    # Called annual + quarterly = 2
    assert finnhub.financials_calls == 2


# ---------------------------------------------------------------------------
# P1c (2026-04): quarterly margin reconstruction from DART cumulatives
# ---------------------------------------------------------------------------


def _quarterly_response(*, revenue: float | None, gross_profit: float | None) -> FinancialsResponse:
    """Build a DART response with just (revenue, gross_profit) — the
    minimum fields the quarterly helper extracts.
    """
    rows = []
    if revenue is not None:
        rows.append(_row(account_id="ifrs-full_Revenue", sj_div="IS", amount=revenue))
    if gross_profit is not None:
        rows.append(_row(account_id="ifrs-full_GrossProfit", sj_div="IS", amount=gross_profit))
    # Stub still needs to satisfy annual extraction when `reprt_code=11011`
    # is queried; supply a minimal annual surface too.
    rows.append(_row(account_id="dart_OperatingIncomeLoss", sj_div="IS", amount=10.0))
    rows.append(_row(account_id="ifrs-full_Equity", sj_div="BS", amount=100.0))
    rows.append(_row(account_id="ifrs-full_CashAndCashEquivalents", sj_div="BS", amount=10.0))
    return FinancialsResponse(status="000", message="정상", list=rows)


class _CumulativeStubClient(_StubDartClient):
    """DART stub that returns DIFFERENT payloads per `reprt_code`.

    `cumulatives_by_year_code` keys are (fiscal_year, reprt_code). Lets
    a test inject a realistic cumulative-to-date pattern and verify the
    standalone quarter values come out correctly via subtraction.
    """

    def __init__(
        self,
        *,
        cumulatives_by_year_code: dict[tuple[int, str], FinancialsResponse],
    ) -> None:
        super().__init__()
        self._cumulatives = cumulatives_by_year_code

    def financials(self, corp_code: str, year, reprt_code: str = "11011", fs_div: str = "CFS"):
        self.calls.append(("financials", corp_code, year, reprt_code))
        return self._cumulatives.get((year, reprt_code), _samsung_like_response())


def test_quarterly_margins_subtract_dart_cumulatives_correctly() -> None:
    """Verify Q-standalone = cumulative_to_date − previous_cumulative.

    Pattern (one fiscal year):
      Q1   rev=100  gp=40    → standalone Q1: 100/40,  GM=0.40
      H1   rev=240  gp=80    → standalone Q2: 140/40,  GM=0.286
      9M   rev=380  gp=130   → standalone Q3: 140/50,  GM=0.357
      FY   rev=540  gp=170   → standalone Q4: 160/40,  GM=0.250
    """
    fy = 2023
    cumulatives = {
        (fy, "11013"): _quarterly_response(revenue=100, gross_profit=40),
        (fy, "11012"): _quarterly_response(revenue=240, gross_profit=80),
        (fy, "11014"): _quarterly_response(revenue=380, gross_profit=130),
        (fy, "11011"): _quarterly_response(revenue=540, gross_profit=170),
    }
    client = _CumulativeStubClient(cumulatives_by_year_code=cumulatives)
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )

    qm_by_id = {q.quarter_id: q.gross_margin for q in funds.quarterly_margins}
    assert "2023Q1" in qm_by_id
    assert "2023Q2" in qm_by_id
    assert "2023Q3" in qm_by_id
    assert "2023Q4" in qm_by_id
    assert qm_by_id["2023Q1"] == pytest.approx(0.40, abs=1e-6)
    assert qm_by_id["2023Q2"] == pytest.approx(40 / 140, abs=1e-6)
    assert qm_by_id["2023Q3"] == pytest.approx(50 / 140, abs=1e-6)
    assert qm_by_id["2023Q4"] == pytest.approx(40 / 160, abs=1e-6)


def test_quarterly_margins_skip_missing_quarter() -> None:
    """If a single reprt_code is absent (e.g., DART filing not posted yet),
    only the affected quarter drops; the others survive."""
    fy = 2023
    # Skip the 9M (REPORT_Q3 = 11014) cumulative — Q3 (subtraction) and
    # Q4 (depends on 9M) should both be missing; Q1 and Q2 still surface.
    cumulatives = {
        (fy, "11013"): _quarterly_response(revenue=100, gross_profit=40),
        (fy, "11012"): _quarterly_response(revenue=240, gross_profit=80),
        # Intentionally omit (fy, "11014")
        (fy, "11011"): _quarterly_response(revenue=540, gross_profit=170),
    }

    class _MissingQ3Client(_CumulativeStubClient):
        def financials(self, corp_code: str, year, reprt_code: str = "11011", fs_div: str = "CFS"):
            self.calls.append(("financials", corp_code, year, reprt_code))
            if (year, reprt_code) == (fy, "11014"):
                # Simulate DART returning status != "000"
                return FinancialsResponse(status="013", message="조회된 데이타가 없습니다", list=[])
            return self._cumulatives.get((year, reprt_code), _samsung_like_response())

    client = _MissingQ3Client(cumulatives_by_year_code=cumulatives)
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    qm_ids = {q.quarter_id for q in funds.quarterly_margins}
    assert "2023Q1" in qm_ids
    assert "2023Q2" in qm_ids
    assert "2023Q3" not in qm_ids
    assert "2023Q4" not in qm_ids


def test_quarterly_margins_capped_at_12() -> None:
    """A 5-year window that yields >12 quarters trims to the most recent 12."""
    cumulatives: dict[tuple[int, str], FinancialsResponse] = {}
    for fy in range(2019, 2024):  # 5 years × 4 quarters = 20 potential quarters
        cumulatives[(fy, "11013")] = _quarterly_response(revenue=100, gross_profit=40)
        cumulatives[(fy, "11012")] = _quarterly_response(revenue=210, gross_profit=80)
        cumulatives[(fy, "11014")] = _quarterly_response(revenue=320, gross_profit=120)
        cumulatives[(fy, "11011")] = _quarterly_response(revenue=440, gross_profit=160)
    client = _CumulativeStubClient(cumulatives_by_year_code=cumulatives)
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=5,
        today=dt.date(2024, 6, 1),
    )
    # quarterly_window in adapter is `most_recent_fy − 2 .. most_recent_fy`
    # = 2021..2023 = 12 quarters max.
    assert len(funds.quarterly_margins) <= 12
    # Most recent should be 2023Q4
    assert funds.quarterly_margins[-1].quarter_id == "2023Q4"


def test_quarterly_margins_drop_on_zero_revenue() -> None:
    """A quarter where standalone revenue resolves to zero (e.g., the
    cumulative didn't move) is silently dropped — don't divide by zero."""
    fy = 2023
    cumulatives = {
        (fy, "11013"): _quarterly_response(revenue=100, gross_profit=40),
        # H1 same as Q1 → Q2 standalone revenue = 0 → must drop
        (fy, "11012"): _quarterly_response(revenue=100, gross_profit=40),
        (fy, "11014"): _quarterly_response(revenue=250, gross_profit=90),
        (fy, "11011"): _quarterly_response(revenue=400, gross_profit=140),
    }
    client = _CumulativeStubClient(cumulatives_by_year_code=cumulatives)
    funds = fetch_live_fundamentals_kr(
        "005930.KS", client=client, history_years=1,
        today=dt.date(2024, 6, 1),
    )
    qm_ids = {q.quarter_id for q in funds.quarterly_margins}
    assert "2023Q2" not in qm_ids  # standalone rev = 0
    # Q1, Q3, Q4 all valid
    assert "2023Q1" in qm_ids
    assert "2023Q3" in qm_ids
    assert "2023Q4" in qm_ids


def test_universe_dispatches_per_symbol() -> None:
    """Mixed-market universe: Finnhub used for US, DART used for KR."""
    class _F:
        def __init__(self):
            self.symbols_seen: list[str] = []

        def financials(self, symbol, freq="annual"):
            self.symbols_seen.append(symbol)
            class _R:
                data: list = []
            return _R()

        def profile(self, symbol):
            class _P:
                finnhub_industry = "Test"
            return _P()

    finnhub = _F()
    dart = _StubDartClient()
    out = fetch_live_universe(
        ["NVDA", "005930.KS", "MSFT"],
        finnhub_client=finnhub,
        dart_client=dart,
    )
    assert {f.symbol for f in out} == {"NVDA", "005930.KS", "MSFT"}
    # Finnhub got the two US tickers; DART got Samsung.
    assert "NVDA" in finnhub.symbols_seen
    assert "MSFT" in finnhub.symbols_seen
    assert any(c[0] == "financials" for c in dart.calls)
