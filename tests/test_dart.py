"""Tests for the OpenDART client (Phase 3 Korean stocks data layer).

HTTP boundary stubbed with httpx.MockTransport so the suite is fully
offline. Live-network smoke test is opt-in via `pytest -m network`
(not included here — add when the user has registered a DART key).
"""

from __future__ import annotations

import io
import json
import zipfile

import httpx
import pytest

from wise_investor.data.dart import (
    BASE_URL,
    CorpMapping,
    DartClient,
    DartError,
    FS_CONSOLIDATED,
    FinancialsResponse,
    REPORT_ANNUAL,
    extract_account_value,
    parse_corp_code_xml,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_FIXTURE_CORP_XML = b"""<?xml version='1.0' encoding='utf-8'?>
<result>
  <list>
    <corp_code>00126380</corp_code>
    <corp_name>Samsung Electronics</corp_name>
    <stock_code>005930</stock_code>
    <modify_date>20260401</modify_date>
  </list>
  <list>
    <corp_code>00164779</corp_code>
    <corp_name>SK hynix</corp_name>
    <stock_code>000660</stock_code>
    <modify_date>20260401</modify_date>
  </list>
  <list>
    <corp_code>00234820</corp_code>
    <corp_name>Unlisted Holdings</corp_name>
    <stock_code></stock_code>
    <modify_date>20260301</modify_date>
  </list>
</result>
"""


_FIXTURE_FINANCIALS_JSON = {
    "status": "000",
    "message": "OK",
    "list": [
        {
            "rcept_no": "20260315000001",
            "corp_code": "00126380",
            "bsns_year": "2024",
            "reprt_code": "11011",
            "sj_div": "IS",
            "sj_nm": "Income Statement",
            "account_id": "ifrs-full_Revenue",
            "account_nm": "매출액",
            "thstrm_nm": "FY2024",
            "thstrm_amount": "300,800,000,000,000",
            "frmtrm_nm": "FY2023",
            "frmtrm_amount": "258,935,000,000,000",
            "currency": "KRW",
        },
        {
            "sj_div": "IS",
            "account_nm": "영업이익",
            "thstrm_amount": "32,725,000,000,000",
            "frmtrm_amount": "6,567,000,000,000",
            "currency": "KRW",
        },
        {
            "sj_div": "IS",
            "account_nm": "당기순이익",
            "thstrm_amount": "25,409,000,000,000",
            "frmtrm_amount": "15,487,000,000,000",
            "currency": "KRW",
        },
        {
            "sj_div": "BS",
            "account_nm": "자산총계",
            "thstrm_amount": "455,905,000,000,000",
            "frmtrm_amount": "448,421,000,000,000",
            "currency": "KRW",
        },
        {
            "sj_div": "BS",
            "account_nm": "부채총계",
            "thstrm_amount": "122,431,000,000,000",
            "frmtrm_amount": "118,995,000,000,000",
            "currency": "KRW",
        },
        {
            "sj_div": "BS",
            "account_nm": "자본총계",
            "thstrm_amount": "333,474,000,000,000",
            "frmtrm_amount": "329,426,000,000,000",
            "currency": "KRW",
        },
    ],
}


def _zip_fixture_corp_xml() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("CORPCODE.xml", _FIXTURE_CORP_XML)
    return buf.getvalue()


def _mock_transport(
    corp_zip: bytes | None = None,
    financials: dict | None = None,
) -> httpx.MockTransport:
    """Build a MockTransport that routes the two DART endpoints we use."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/corpCode.xml" in url:
            return httpx.Response(
                200,
                content=corp_zip or _zip_fixture_corp_xml(),
                headers={"content-type": "application/zip"},
            )
        if "/fnlttSinglAcntAll.json" in url:
            body = financials or _FIXTURE_FINANCIALS_JSON
            return httpx.Response(
                200,
                content=json.dumps(body).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
        return httpx.Response(404, content=b"{}")

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Construction + auth
# ---------------------------------------------------------------------------


def test_dart_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("wise_investor.data.dart.settings.dart_api_key", "")
    with pytest.raises(DartError, match="DART API key"):
        DartClient()


def test_dart_client_accepts_explicit_key() -> None:
    # Explicit non-empty key bypasses the settings fallback.
    client = DartClient(api_key="testkey", transport=_mock_transport())
    assert client.api_key == "testkey"
    client.close()


# ---------------------------------------------------------------------------
# corpCode XML parsing
# ---------------------------------------------------------------------------


def test_parse_corp_code_xml_returns_mapping_rows() -> None:
    rows = parse_corp_code_xml(_FIXTURE_CORP_XML)
    assert len(rows) == 3
    samsung = next(r for r in rows if r.corp_code == "00126380")
    assert samsung.corp_name == "Samsung Electronics"
    assert samsung.stock_code == "005930"


def test_parse_corp_code_xml_blank_stock_code_becomes_none() -> None:
    rows = parse_corp_code_xml(_FIXTURE_CORP_XML)
    unlisted = next(r for r in rows if r.corp_code == "00234820")
    assert unlisted.stock_code is None


def test_parse_corp_code_xml_zero_pads_codes() -> None:
    # corp_code and stock_code must end up at their canonical widths
    # even if the source XML omits leading zeros.
    raw = (
        b"<result><list>"
        b"<corp_code>126380</corp_code>"
        b"<corp_name>X</corp_name>"
        b"<stock_code>005930</stock_code>"
        b"<modify_date>20260401</modify_date>"
        b"</list></result>"
    )
    rows = parse_corp_code_xml(raw)
    assert rows[0].corp_code == "00126380"  # 8 digits enforced


# ---------------------------------------------------------------------------
# Corp-code ZIP → XML extraction via client
# ---------------------------------------------------------------------------


def test_client_fetches_and_extracts_corp_zip(tmp_path) -> None:
    client = DartClient(api_key="testkey", transport=_mock_transport())
    mappings = client.load_corp_mapping(cache_path=tmp_path / "corpCode.xml")
    samsung_codes = [m.corp_code for m in mappings if m.corp_name == "Samsung Electronics"]
    assert samsung_codes == ["00126380"]
    client.close()


def test_client_caches_corp_xml_across_calls(tmp_path) -> None:
    """Second load_corp_mapping reads from disk, not the network.

    Drive a transport that would RAISE on the second call; if caching
    is broken the test fails loudly.
    """
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            200,
            content=_zip_fixture_corp_xml(),
            headers={"content-type": "application/zip"},
        )

    cache = tmp_path / "corpCode.xml"
    client = DartClient(api_key="testkey", transport=httpx.MockTransport(handler))
    client.load_corp_mapping(cache_path=cache)
    client.load_corp_mapping(cache_path=cache)
    client.close()
    assert len(calls) == 1  # second call served from disk cache


def test_corp_code_from_stock_code_finds_samsung(tmp_path) -> None:
    client = DartClient(api_key="testkey", transport=_mock_transport())
    # Override the default cache path onto tmp so tests stay hermetic.
    import wise_investor.data.dart as dart_mod

    dart_mod.CORPCODE_CACHE = tmp_path / "corpCode.xml"
    corp = client.corp_code_from_stock_code("005930")
    assert corp == "00126380"
    client.close()


def test_corp_code_from_stock_code_none_for_unknown(tmp_path) -> None:
    client = DartClient(api_key="testkey", transport=_mock_transport())
    import wise_investor.data.dart as dart_mod

    dart_mod.CORPCODE_CACHE = tmp_path / "corpCode.xml"
    assert client.corp_code_from_stock_code("999999") is None
    client.close()


def test_corp_code_from_stock_code_zero_pads(tmp_path) -> None:
    client = DartClient(api_key="testkey", transport=_mock_transport())
    import wise_investor.data.dart as dart_mod

    dart_mod.CORPCODE_CACHE = tmp_path / "corpCode.xml"
    # "5930" → "005930" → Samsung.
    corp = client.corp_code_from_stock_code("5930")
    assert corp == "00126380"
    client.close()


# ---------------------------------------------------------------------------
# Financial statements
# ---------------------------------------------------------------------------


def test_financials_returns_parsed_response() -> None:
    client = DartClient(api_key="testkey", transport=_mock_transport())
    resp = client.financials(corp_code="00126380", year=2024)
    client.close()
    assert resp.ok
    assert resp.status == "000"
    assert len(resp.rows) == 6
    revenue_row = resp.rows[0]
    assert revenue_row.this_period_value == 300_800_000_000_000.0
    assert revenue_row.currency == "KRW"


def test_financials_parses_comma_amounts_to_float() -> None:
    resp = FinancialsResponse.model_validate(_FIXTURE_FINANCIALS_JSON)
    samsung_revenue = resp.rows[0]
    assert samsung_revenue.this_period_value == 300_800_000_000_000.0
    assert samsung_revenue.prior_period_value == 258_935_000_000_000.0


def test_financials_tolerates_dash_amount() -> None:
    fixture = {
        "status": "000",
        "message": "OK",
        "list": [
            {
                "sj_div": "BS",
                "account_nm": "기타",
                "thstrm_amount": "-",
                "frmtrm_amount": "",
            }
        ],
    }
    resp = FinancialsResponse.model_validate(fixture)
    assert resp.rows[0].this_period_value is None
    assert resp.rows[0].prior_period_value is None


def test_extract_account_value_finds_revenue() -> None:
    resp = FinancialsResponse.model_validate(_FIXTURE_FINANCIALS_JSON)
    v = extract_account_value(resp, account_nm="매출액")
    assert v == 300_800_000_000_000.0


def test_extract_account_value_respects_sj_div_filter() -> None:
    """When the same account name appears in two statements, sj_div
    narrows the search.
    """
    fixture = {
        "status": "000",
        "message": "OK",
        "list": [
            {"sj_div": "IS", "account_nm": "자산", "thstrm_amount": "1"},
            {"sj_div": "BS", "account_nm": "자산", "thstrm_amount": "2"},
        ],
    }
    resp = FinancialsResponse.model_validate(fixture)
    assert extract_account_value(resp, account_nm="자산", sj_div="BS") == 2.0
    assert extract_account_value(resp, account_nm="자산", sj_div="IS") == 1.0


def test_extract_account_value_prefers_account_id_when_given() -> None:
    resp = FinancialsResponse.model_validate(_FIXTURE_FINANCIALS_JSON)
    v = extract_account_value(resp, account_id="ifrs-full_Revenue")
    assert v == 300_800_000_000_000.0


def test_extract_account_value_requires_at_least_one_key() -> None:
    resp = FinancialsResponse.model_validate(_FIXTURE_FINANCIALS_JSON)
    with pytest.raises(ValueError, match="account_id or account_nm"):
        extract_account_value(resp)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_financials_raises_on_http_400() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, content=b'{"status": "013"}')

    client = DartClient(api_key="testkey", transport=httpx.MockTransport(handler))
    with pytest.raises(DartError, match="HTTP 400"):
        client.financials(corp_code="00126380", year=2024)
    client.close()


def test_financials_dart_error_code_preserved_in_response() -> None:
    """DART returns 200 OK with a status-string when auth fails or
    corp/year combo is invalid. The client surfaces it as response.ok.
    """
    bad = {"status": "013", "message": "No data", "list": []}
    client = DartClient(
        api_key="testkey",
        transport=_mock_transport(financials=bad),
    )
    resp = client.financials(corp_code="00000000", year=2099)
    assert resp.ok is False
    assert resp.status == "013"
    assert resp.message == "No data"
    client.close()
