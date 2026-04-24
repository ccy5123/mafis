"""Tests for the DART → crew facts adapter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wise_investor.data.dart_facts import (
    is_korean_ticker,
    normalize_korean_symbol,
    pre_gather_dart_facts,
)


# ---------------------------------------------------------------------------
# Ticker detection / normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sym", ["005930", "000660", "035420", "373220"])
def test_is_korean_ticker_detects_bare_6digit(sym: str) -> None:
    assert is_korean_ticker(sym) is True


@pytest.mark.parametrize("sym", ["005930.KS", "005930.KQ", "005930.ks"])
def test_is_korean_ticker_strips_suffix(sym: str) -> None:
    assert is_korean_ticker(sym) is True


@pytest.mark.parametrize("sym", ["NVDA", "GEV", "AAPL", "12345", "1234567", ""])
def test_is_korean_ticker_rejects_us_tickers(sym: str) -> None:
    assert is_korean_ticker(sym) is False


def test_normalize_korean_symbol_strips_suffix_and_zero_pads() -> None:
    assert normalize_korean_symbol("005930.KS") == "005930"
    assert normalize_korean_symbol("5930") == "005930"
    assert normalize_korean_symbol("000660") == "000660"
    assert normalize_korean_symbol("35420.KQ") == "035420"


# ---------------------------------------------------------------------------
# pre_gather_dart_facts with a stub DartClient
# ---------------------------------------------------------------------------


@dataclass
class _FakeRow:
    account_nm: str
    thstrm_amount: str

    @property
    def this_period_value(self) -> float | None:
        try:
            return float(self.thstrm_amount.replace(",", ""))
        except Exception:
            return None


@dataclass
class _FakeResponse:
    status: str
    message: str
    rows: list[_FakeRow]

    @property
    def ok(self) -> bool:
        return self.status == "000"


@dataclass
class _FakeMapping:
    corp_code: str
    corp_name: str
    stock_code: str | None


class _StubDart:
    """Minimal DartClient stand-in — no network, no SQLite."""

    def __init__(
        self,
        corp_code: str | None = "00126380",
        corp_name: str = "삼성전자",
        account_values: dict[str, str] | None = None,
        financials_ok: bool = True,
        financials_status: str = "000",
        financials_message: str = "OK",
    ) -> None:
        self.corp_code_value = corp_code
        self.corp_name = corp_name
        self.account_values = account_values or {
            "매출액": "300,870,903,000,000",
            "영업이익": "32,725,961,000,000",
            "당기순이익": "34,451,351,000,000",
        }
        self.financials_ok = financials_ok
        self.financials_status = financials_status
        self.financials_message = financials_message

    def corp_code_from_stock_code(self, stock_code: str) -> str | None:
        return self.corp_code_value

    def load_corp_mapping(self) -> list[_FakeMapping]:
        if self.corp_code_value is None:
            return []
        return [
            _FakeMapping(
                corp_code=self.corp_code_value,
                corp_name=self.corp_name,
                stock_code="005930",
            )
        ]

    def financials(self, corp_code: str, year: int | str, reprt_code: str):
        if not self.financials_ok:
            return _FakeResponse(
                status=self.financials_status,
                message=self.financials_message,
                rows=[],
            )
        rows = [
            _FakeRow(account_nm=name, thstrm_amount=amount)
            for name, amount in self.account_values.items()
        ]
        # extract_account_value uses response.rows, so our fake shapes
        # align. The real helper also checks sj_div; since our stub
        # rows lack that attribute, set sj_div=None via the parent
        # namespace.
        for r in rows:
            # Dynamic attribute monkey-fill for fields extract_account_value
            # reads (account_id / sj_div are both optional None).
            r.account_id = None  # type: ignore[attr-defined]
            r.sj_div = None  # type: ignore[attr-defined]
        return _FakeResponse(status="000", message="OK", rows=rows)

    def close(self) -> None:
        pass


def test_pre_gather_dart_facts_happy_path() -> None:
    stub = _StubDart()
    facts = pre_gather_dart_facts("005930", year=2024, client=stub)
    assert "dart.metadata" in facts
    assert "삼성전자" in facts["dart.metadata"]
    # All 9 accounts produce a fact line.
    assert facts["dart.revenue"].startswith("revenue:")
    assert "300,870,903,000,000 KRW" in facts["dart.revenue"]


def test_pre_gather_dart_facts_includes_usd_conversion() -> None:
    stub = _StubDart()
    # 1461.66 KRW per USD — today's FRED DEXKOUS rate.
    facts = pre_gather_dart_facts(
        "005930", year=2024, usd_krw_rate=1461.66, client=stub
    )
    # Samsung revenue ≈ 300.87T KRW → ~$205.8B USD.
    assert "USD" in facts["dart.revenue"]
    assert "205" in facts["dart.revenue"] or "$205" in facts["dart.revenue"]


def test_pre_gather_dart_facts_no_rate_omits_usd() -> None:
    stub = _StubDart()
    facts = pre_gather_dart_facts("005930", year=2024, client=stub)
    assert "USD" not in facts["dart.revenue"]


def test_pre_gather_dart_facts_missing_account_renders_na() -> None:
    # Remove the revenue entry from the fake response.
    stub = _StubDart(account_values={"당기순이익": "34,451,351,000,000"})
    facts = pre_gather_dart_facts("005930", year=2024, client=stub)
    assert "N/A" in facts["dart.revenue"]


def test_pre_gather_dart_facts_unknown_symbol_returns_error() -> None:
    stub = _StubDart(corp_code=None)
    facts = pre_gather_dart_facts("999999", year=2024, client=stub)
    assert facts["dart.metadata"].startswith("ERROR")
    assert "999999" in facts["dart.metadata"]


def test_pre_gather_dart_facts_dart_error_surfaces() -> None:
    stub = _StubDart(
        financials_ok=False,
        financials_status="013",
        financials_message="Record not found",
    )
    facts = pre_gather_dart_facts("005930", year=2099, client=stub)
    assert "DART error 013" in facts["dart.metadata"]


def test_pre_gather_dart_facts_accepts_ks_suffix() -> None:
    stub = _StubDart()
    facts = pre_gather_dart_facts("005930.KS", year=2024, client=stub)
    assert "삼성전자" in facts["dart.metadata"]
    assert "005930" in facts["dart.metadata"]


# ---------------------------------------------------------------------------
# Runner dispatcher integration — Korean symbols route to DART path
# ---------------------------------------------------------------------------


def test_runner_pre_gather_facts_dispatches_korean(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """pre_gather_facts for a Korean ticker hits the DART builder, NOT
    the Finnhub path. The Finnhub stubs would crash if called.
    """
    from wise_investor.agents import runner

    # Redirect the facts cache to tmp.
    monkeypatch.setattr(runner, "FACTS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(
        runner, "_facts_cache_path", lambda sym: tmp_path / f"{sym}.json"
    )

    # Patch the DART builder to return a deterministic dict.
    import wise_investor.data.dart_facts as dart_facts_mod

    def _fake_dart(symbol, year=2024, reprt_code=None, usd_krw_rate=None, client=None):
        return {
            "dart.metadata": f"Symbol: {symbol} / ok",
            "dart.revenue": "revenue: 300T KRW",
        }

    monkeypatch.setattr(dart_facts_mod, "pre_gather_dart_facts", _fake_dart)

    # Patch FRED so the dispatcher doesn't hit the network for the rate.
    from wise_investor.data import fred as fred_mod

    monkeypatch.setattr(
        fred_mod,
        "get_macro_snapshot",
        lambda client=None: _FakeSnapshot(None),
    )

    # Also patch the Finnhub-path exec functions to raise loudly if
    # accidentally called.
    def _should_not_fire(*a, **k):
        raise AssertionError("Finnhub path fired for Korean ticker")

    monkeypatch.setattr(runner, "_exec_cross_validate_quote", _should_not_fire)
    monkeypatch.setattr(runner, "_exec_calculate_per", _should_not_fire)

    facts = runner.pre_gather_facts("005930", use_cache=False)
    assert "dart.metadata" in facts
    assert "dart.revenue" in facts
    # Finnhub-path keys are absent.
    assert "calculate_per" not in facts


@dataclass
class _FakeSnapshot:
    usd_krw_rate: None
