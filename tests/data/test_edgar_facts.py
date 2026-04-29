"""edgar_facts.py — companyfacts → Finnhub-shape conversion (P1b 2026-04).

Unit tests use synthetic companyfacts JSON payloads to exercise:
  - Currency detection (USD vs IFRS native)
  - ifrs-full vs us-gaap dual-namespace
  - ASC 606 splice (SalesRevenueNet pre-2018 + RevenueFromContract* post)
  - Restatement handling (later filed wins)
  - Cache round-trip
  - Form preservation (10-K vs 20-F on entries)

Network-touching tests (companyfacts API, ticker_to_cik) are NOT here —
those rely on SEC availability and would slow CI. The probe scripts
(`scripts/probe_edgar_*.py`) cover live verification on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from wise_investor.data.edgar_facts import (
    EdgarFactsError,
    _decide_primary_currency,
    companyfacts_to_response,
    fetch_company_facts,
    fetch_financials_via_edgar,
)
from wise_investor.data.finnhub import extract_field


# ---------------------------------------------------------------------------
# Helpers to build synthetic companyfacts JSONs
# ---------------------------------------------------------------------------


def _fy_item(
    *, fy: int, val: float, filed: str, form: str = "20-F", end: str | None = None
) -> dict:
    return {
        "end": end or f"{fy}-12-31",
        "val": val,
        "fy": fy,
        "fp": "FY",
        "form": form,
        "filed": filed,
        "accn": f"acc-{fy}",
    }


def _build_facts(
    cik: str,
    namespace_concepts: dict[str, dict[str, dict[str, list[dict]]]],
) -> dict:
    """Build a minimal companyfacts JSON.

    `namespace_concepts` is `{namespace: {concept: {unit: [items...]}}}`.
    """
    return {
        "cik": int(cik),
        "facts": {
            ns: {
                concept: {"units": units}
                for concept, units in concepts.items()
            }
            for ns, concepts in namespace_concepts.items()
        },
    }


def _ifrs_filer_facts(cik: str = "1234567890") -> dict:
    """Synthetic TSM-shape: ifrs-full + dual-currency Assets (TWD+USD)."""
    return _build_facts(
        cik,
        {
            "ifrs-full": {
                "Revenue": {
                    "USD": [
                        _fy_item(fy=2017, val=32_977_300_000, filed="2018-04-19"),
                        _fy_item(fy=2018, val=33_697_300_000, filed="2019-04-17"),
                    ],
                    "TWD": [
                        _fy_item(fy=2017, val=977_447_000_000, filed="2018-04-19"),
                    ],
                },
                "GrossProfit": {
                    "USD": [
                        _fy_item(fy=2017, val=16_694_500_000, filed="2018-04-19"),
                        _fy_item(fy=2018, val=16_265_100_000, filed="2019-04-17"),
                    ],
                },
                "ProfitLossFromOperatingActivities": {
                    "USD": [
                        _fy_item(fy=2017, val=13_008_100_000, filed="2018-04-19"),
                        _fy_item(fy=2018, val=12_532_600_000, filed="2019-04-17"),
                    ],
                },
                "Assets": {
                    "USD": [
                        _fy_item(fy=2017, val=67_197_400_000, filed="2018-04-19"),
                        _fy_item(fy=2018, val=68_279_400_000, filed="2019-04-17"),
                    ],
                    "TWD": [
                        _fy_item(fy=2017, val=1_886_296_700_000, filed="2018-04-19"),
                    ],
                },
                "CashAndCashEquivalents": {
                    "USD": [
                        _fy_item(fy=2017, val=18_260_900_000, filed="2018-04-19"),
                        _fy_item(fy=2018, val=18_078_800_000, filed="2019-04-17"),
                    ],
                },
            },
        },
    )


def _us_gaap_eur_filer_facts(cik: str = "9999999999") -> dict:
    """Synthetic ASML-shape: us-gaap concepts in EUR (no USD), with ASC 606 split."""
    return _build_facts(
        cik,
        {
            "us-gaap": {
                # Pre-ASC 606 revenue tag (2009-2017)
                "SalesRevenueNet": {
                    "EUR": [
                        _fy_item(fy=2016, val=5_856_277_000, filed="2017-02-08"),
                        _fy_item(fy=2017, val=6_287_400_000, filed="2018-02-07"),
                    ],
                },
                # Post-ASC 606 revenue tag (2018+)
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "EUR": [
                        _fy_item(fy=2018, val=6_875_100_000, filed="2019-02-06"),
                    ],
                },
                "GrossProfit": {
                    "EUR": [
                        _fy_item(fy=2017, val=2_895_700_000, filed="2018-02-07"),
                        _fy_item(fy=2018, val=3_145_300_000, filed="2019-02-06"),
                    ],
                },
                "OperatingIncomeLoss": {
                    "EUR": [
                        _fy_item(fy=2017, val=1_565_100_000, filed="2018-02-07"),
                        _fy_item(fy=2018, val=1_758_500_000, filed="2019-02-06"),
                    ],
                },
                "Assets": {
                    "EUR": [
                        _fy_item(fy=2017, val=17_205_900_000, filed="2018-02-07"),
                        _fy_item(fy=2018, val=18_188_900_000, filed="2019-02-06"),
                    ],
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "EUR": [
                        _fy_item(fy=2017, val=2_419_500_000, filed="2018-02-07"),
                        _fy_item(fy=2018, val=2_458_700_000, filed="2019-02-06"),
                    ],
                },
            },
        },
    )


# ---------------------------------------------------------------------------
# _decide_primary_currency
# ---------------------------------------------------------------------------


def test_primary_currency_prefers_USD_when_available() -> None:
    facts = _ifrs_filer_facts()  # has both TWD and USD on Assets
    assert _decide_primary_currency(facts) == "USD"


def test_primary_currency_falls_back_to_native_when_no_USD() -> None:
    facts = _us_gaap_eur_filer_facts()
    assert _decide_primary_currency(facts) == "EUR"


def test_primary_currency_returns_USD_when_no_assets_at_all() -> None:
    # Pathological: no Assets concept anywhere.
    facts = {"cik": 0, "facts": {"us-gaap": {"Revenues": {"USD": []}}}}
    assert _decide_primary_currency(facts) == "USD"


# ---------------------------------------------------------------------------
# companyfacts_to_response — IFRS path
# ---------------------------------------------------------------------------


def test_ifrs_response_yields_entries_in_USD() -> None:
    facts = _ifrs_filer_facts()
    resp = companyfacts_to_response(facts, symbol="TSM")
    assert resp.symbol == "TSM"
    assert len(resp.data) == 2
    # newest-first (mirrors Finnhub ordering)
    assert resp.data[0].year == 2018
    assert resp.data[1].year == 2017
    # IC = Total Assets - Cash
    e2017 = resp.data[1]
    ta = extract_field(e2017, "total_assets")
    cash = extract_field(e2017, "cash_and_cash_equivalents")
    assert ta == pytest.approx(67_197_400_000.0)
    assert cash == pytest.approx(18_260_900_000.0)
    assert e2017.form == "20-F"
    # All FinancialLineItems should be tagged USD (the primary currency)
    for line in e2017.report.bs + e2017.report.ic:
        assert line.unit == "USD"


def test_ifrs_response_concept_label_preserves_namespace() -> None:
    facts = _ifrs_filer_facts()
    resp = companyfacts_to_response(facts, symbol="TSM")
    e2017 = next(e for e in resp.data if e.year == 2017)
    revenue_concepts = [li.concept for li in e2017.report.ic if "Revenue" in li.concept]
    # The concept tag must reveal which namespace it came from for audit.
    assert any(c.startswith("ifrs-full_") for c in revenue_concepts)


# ---------------------------------------------------------------------------
# companyfacts_to_response — US-GAAP/EUR path with ASC 606 splice
# ---------------------------------------------------------------------------


def test_us_gaap_eur_response_uses_native_currency() -> None:
    facts = _us_gaap_eur_filer_facts()
    resp = companyfacts_to_response(facts, symbol="ASML")
    assert resp.symbol == "ASML"
    assert len(resp.data) == 3  # 2016, 2017, 2018
    e2018 = next(e for e in resp.data if e.year == 2018)
    rev = extract_field(e2018, "revenue")
    assert rev == pytest.approx(6_875_100_000.0)
    # 2018 revenue must come from the ASC 606 tag
    rev_concepts = [li.concept for li in e2018.report.ic]
    assert "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax" in rev_concepts
    # Currency on every line should be EUR
    for line in e2018.report.ic + e2018.report.bs:
        assert line.unit == "EUR"


def test_asc606_splice_pre_2018_uses_legacy_tag() -> None:
    """ASML 2017 revenue should come from SalesRevenueNet (pre-ASC 606)."""
    facts = _us_gaap_eur_filer_facts()
    resp = companyfacts_to_response(facts, symbol="ASML")
    e2017 = next(e for e in resp.data if e.year == 2017)
    rev = extract_field(e2017, "revenue")
    assert rev == pytest.approx(6_287_400_000.0)
    rev_concepts = [li.concept for li in e2017.report.ic]
    assert "us-gaap_SalesRevenueNet" in rev_concepts


# ---------------------------------------------------------------------------
# Restatement: later filed wins
# ---------------------------------------------------------------------------


def test_restatement_later_filed_wins() -> None:
    facts = _build_facts(
        "1234",
        {
            "us-gaap": {
                "Assets": {
                    "USD": [
                        _fy_item(fy=2020, val=100.0, filed="2021-02-15", form="10-K"),
                        # Restated value — later filed
                        _fy_item(
                            fy=2020,
                            val=110.0,
                            filed="2022-04-30",
                            form="10-K/A",
                        ),
                    ],
                },
                "OperatingIncomeLoss": {
                    "USD": [
                        _fy_item(fy=2020, val=20.0, filed="2021-02-15"),
                    ],
                },
                "CashAndCashEquivalentsAtCarryingValue": {
                    "USD": [
                        _fy_item(fy=2020, val=10.0, filed="2021-02-15"),
                    ],
                },
            },
        },
    )
    resp = companyfacts_to_response(facts, symbol="TEST")
    assert len(resp.data) == 1
    e = resp.data[0]
    assert extract_field(e, "total_assets") == pytest.approx(110.0)
    # The entry's form should reflect the latest amendment too.
    assert e.form == "10-K/A"


# ---------------------------------------------------------------------------
# Cache round-trip
# ---------------------------------------------------------------------------


class _StubResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


def test_fetch_company_facts_caches_to_disk(tmp_path: Path) -> None:
    payload = _ifrs_filer_facts(cik="0000123456")
    calls: list[str] = []

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _StubResponse:
        calls.append(url)
        return _StubResponse(payload)

    # First call — fetches and writes cache.
    out1 = fetch_company_facts(
        "0000123456", cache=True, cache_dir=tmp_path, http_get=fake_get,
    )
    # Second call — must hit the cache, not the network.
    out2 = fetch_company_facts(
        "0000123456", cache=True, cache_dir=tmp_path, http_get=fake_get,
    )
    assert out1 == out2
    assert len(calls) == 1
    cached_files = list(tmp_path.glob("CIK*.json"))
    assert len(cached_files) == 1


def test_fetch_company_facts_raises_on_http_error(tmp_path: Path) -> None:
    def boom(url: str, *args: Any, **kwargs: Any) -> _StubResponse:
        raise RuntimeError("network down")

    with pytest.raises(EdgarFactsError):
        fetch_company_facts(
            "0001234567", cache=False, cache_dir=tmp_path, http_get=boom,
        )


# ---------------------------------------------------------------------------
# fetch_financials_via_edgar — end-to-end (CIK passed explicitly to skip lookup)
# ---------------------------------------------------------------------------


def test_fetch_financials_via_edgar_returns_response(tmp_path: Path) -> None:
    payload = _ifrs_filer_facts(cik="0001046179")

    def fake_get(url: str, *args: Any, **kwargs: Any) -> _StubResponse:
        return _StubResponse(payload)

    resp = fetch_financials_via_edgar(
        "TSM",
        cik="0001046179",
        cache=False,
        cache_dir=tmp_path,
        http_get=fake_get,
    )
    assert resp.symbol == "TSM"
    assert len(resp.data) == 2
    assert resp.data[0].form == "20-F"
