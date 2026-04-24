"""OpenDART client — Korean equivalent of SEC EDGAR.

Source: https://opendart.fss.or.kr (Korean Financial Supervisory Service)

Why this exists: Finnhub free tier returns 403 on Korean-exchange
tickers (suffix `.KS`), and SEC EDGAR obviously doesn't host KRX
filings. DART is the authoritative regulatory filing system for
Korea. Free tier allows unlimited personal/research use with a
registered API key.

Scope of this module (Phase 3 data layer):
  - Ticker-to-corp_code lookup (stock_code 6-digit → corp_code 8-digit)
    via the /corpCode.xml endpoint. Cached on disk after first fetch.
  - Full financial statements via /fnlttSinglAcntAll.json — returns
    BS/IS/CIS/CF/SCE rows for a (corp_code, year, report, cfs|ofs)
    tuple. Parsed into a FinancialsResponse dataclass.
  - Helper to extract a named account value from a filing response.

NOT in this module (deferred):
  - XBRL parsing (DART also serves /document.xml; out of scope for v1).
  - Multi-period historical pulls (callers can loop).
  - KRW → USD normalization (tools that consume DART data do this via
    the existing FRED DEXKOUS rate).
"""

from __future__ import annotations

import io
import logging
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from wise_investor.config import settings


logger = logging.getLogger(__name__)


BASE_URL = "https://opendart.fss.or.kr/api"

REPO_ROOT = Path(__file__).resolve().parents[3]
CORPCODE_CACHE = REPO_ROOT / "data" / "dart_cache" / "corpCode.xml"

# DART report-code enumeration, per the OpenDART developer guide.
REPORT_ANNUAL = "11011"   # 사업보고서 (annual)
REPORT_H1 = "11012"       # 반기보고서
REPORT_Q1 = "11013"       # 1분기보고서
REPORT_Q3 = "11014"       # 3분기보고서

# Consolidated vs separate financial statements.
FS_CONSOLIDATED = "CFS"
FS_SEPARATE = "OFS"


class DartError(RuntimeError):
    """Raised on HTTP failure or DART error response payload."""


@dataclass
class CorpMapping:
    """One row of the corpCode.xml registry."""

    corp_code: str  # 8-digit zero-padded
    corp_name: str
    stock_code: str | None  # 6-digit KRX code, None for unlisted
    modify_date: str  # YYYYMMDD


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FinancialRow(_Model):
    """One row of the fnlttSinglAcntAll response (one account per row)."""

    rcept_no: str | None = None
    reprt_code: str | None = None
    bsns_year: str | None = None
    corp_code: str | None = None
    sj_div: str | None = None      # BS / IS / CIS / CF / SCE
    sj_nm: str | None = None       # Korean statement name
    account_id: str | None = None
    account_nm: str | None = None
    account_detail: str | None = None
    thstrm_nm: str | None = None   # current period label
    thstrm_amount: str | None = None  # amounts are STRING in DART (comma-formatted)
    frmtrm_nm: str | None = None
    frmtrm_amount: str | None = None
    bfefrmtrm_nm: str | None = None
    bfefrmtrm_amount: str | None = None
    ord: str | None = None
    currency: str | None = None

    @property
    def this_period_value(self) -> float | None:
        return _to_float(self.thstrm_amount)

    @property
    def prior_period_value(self) -> float | None:
        return _to_float(self.frmtrm_amount)


class FinancialsResponse(_Model):
    # Accept the source's "list" key but expose it as `rows` in Python
    # — `list` would shadow the builtin type used in the annotation.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    status: str = "000"
    message: str = ""
    rows: list[FinancialRow] = Field(default_factory=list, alias="list")

    @property
    def ok(self) -> bool:
        return self.status == "000"


def _to_float(raw: str | None) -> float | None:
    """DART returns amounts as strings with comma thousands separators
    ('1,234,567'). Parse to float; empty / '-' / non-numeric → None.
    """
    if raw is None:
        return None
    s = raw.strip().replace(",", "")
    if not s or s in {"-", "—"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class DartClient:
    """Thin sync client over the OpenDART /api/ endpoints we depend on."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 15.0,
        max_retries: int = 2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.dart_api_key
        if not self.api_key or self.api_key == "your_dart_api_key_here":
            raise DartError(
                "DART API key missing. Register at "
                "opendart.fss.or.kr/mngInfo/mngInfoMain.do and set "
                "DART_API_KEY in .env."
            )
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        client_kwargs: dict[str, Any] = {"timeout": timeout}
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DartClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        expected: str = "json",
    ) -> httpx.Response:
        full_params = dict(params or {})
        full_params["crtfc_key"] = self.api_key
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.get(f"{self.base_url}{path}", params=full_params)
            except httpx.TransportError as e:
                last_exc = e
                logger.warning("DART transport error on %s: %s", path, e)
                time.sleep(min(2 ** attempt, 4))
                continue
            if r.status_code >= 500:
                logger.warning("DART %d on %s (attempt %d)", r.status_code, path, attempt)
                time.sleep(min(2 ** attempt, 4))
                continue
            if r.status_code >= 400:
                raise DartError(f"HTTP {r.status_code} on {path}: {r.text[:200]}")
            return r
        raise DartError(f"{path} failed after {self.max_retries} retries: {last_exc}")

    # ---- corpCode.xml ------------------------------------------------

    def fetch_corp_code_zip(self) -> bytes:
        """Download the raw ZIP payload from /corpCode.xml.

        Returns the bytes of the ZIP file (not the inner CORPCODE.xml).
        Callers are expected to pass the bytes to parse_corp_code_zip().
        """
        r = self._get("/corpCode.xml")
        return r.content

    def load_corp_mapping(
        self,
        force_refresh: bool = False,
        cache_path: Path | None = None,
    ) -> list[CorpMapping]:
        """Return the full corpCode registry, cached to disk.

        First call downloads the ZIP, extracts CORPCODE.xml, and
        persists it at `data/dart_cache/corpCode.xml`. Subsequent calls
        read from disk. Pass `force_refresh=True` to re-download.
        """
        cache_path = cache_path or CORPCODE_CACHE
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if force_refresh or not cache_path.exists():
            zip_bytes = self.fetch_corp_code_zip()
            xml_bytes = _extract_corpcode_xml(zip_bytes)
            cache_path.write_bytes(xml_bytes)
        else:
            xml_bytes = cache_path.read_bytes()

        return parse_corp_code_xml(xml_bytes)

    def corp_code_from_stock_code(
        self, stock_code: str, force_refresh: bool = False
    ) -> str | None:
        """Resolve a 6-digit KRX stock code (e.g. '005930' for Samsung
        Electronics) to the 8-digit DART corp_code. Returns None if
        the stock code isn't a DART-listed entity.
        """
        target = stock_code.strip().zfill(6)
        mappings = self.load_corp_mapping(force_refresh=force_refresh)
        for m in mappings:
            if m.stock_code and m.stock_code == target:
                return m.corp_code
        return None

    # ---- financial statements ---------------------------------------

    def financials(
        self,
        corp_code: str,
        year: int | str,
        reprt_code: str = REPORT_ANNUAL,
        fs_div: str = FS_CONSOLIDATED,
    ) -> FinancialsResponse:
        """Fetch the full annual/quarterly statements for a corp_code.

        Defaults: annual report (11011), consolidated (CFS). Callers
        analysing a separate-entity company pass fs_div='OFS'.
        """
        r = self._get(
            "/fnlttSinglAcntAll.json",
            params={
                "corp_code": str(corp_code).zfill(8),
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "fs_div": fs_div,
            },
        )
        try:
            data = r.json()
        except Exception as e:
            raise DartError(f"JSON parse failed: {e}") from e
        return FinancialsResponse.model_validate(data)


# ---------------------------------------------------------------------------
# Parsing helpers (module-level so tests can hit them without a client)
# ---------------------------------------------------------------------------


def _extract_corpcode_xml(zip_bytes: bytes) -> bytes:
    """Pull CORPCODE.xml out of the ZIP payload /corpCode.xml returns."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        names = z.namelist()
        for candidate in ("CORPCODE.xml", "corpCode.xml"):
            if candidate in names:
                return z.read(candidate)
        if names:
            # Some responses name the file differently in lowercase.
            return z.read(names[0])
    raise DartError("CORPCODE.xml not found inside ZIP payload")


def parse_corp_code_xml(xml_bytes: bytes) -> list[CorpMapping]:
    """Parse the CORPCODE.xml body into CorpMapping rows.

    Shape of the XML:
      <result>
        <list>
          <corp_code>...</corp_code>
          <corp_name>...</corp_name>
          <stock_code>...</stock_code>  <!-- blank for unlisted -->
          <modify_date>...</modify_date>
        </list>
        <list>...</list>
        ...
      </result>
    """
    root = ET.fromstring(xml_bytes)
    out: list[CorpMapping] = []
    for item in root.findall("list"):
        corp_code = (item.findtext("corp_code") or "").strip().zfill(8)
        corp_name = (item.findtext("corp_name") or "").strip()
        stock_code = (item.findtext("stock_code") or "").strip()
        modify_date = (item.findtext("modify_date") or "").strip()
        out.append(
            CorpMapping(
                corp_code=corp_code,
                corp_name=corp_name,
                stock_code=stock_code if stock_code else None,
                modify_date=modify_date,
            )
        )
    return out


def extract_account_value(
    response: FinancialsResponse,
    account_id: str | None = None,
    account_nm: str | None = None,
    sj_div: str | None = None,
) -> float | None:
    """Find a specific account in a FinancialsResponse and return its
    current-period value in the native currency (usually KRW).

    Match priority: account_id (XBRL) > account_nm (Korean display
    name). `sj_div` (BS/IS/CIS/CF/SCE) narrows the search when the
    same account name appears in multiple statements.
    """
    if account_id is None and account_nm is None:
        raise ValueError("Must provide either account_id or account_nm")
    for row in response.rows:
        if sj_div and row.sj_div != sj_div:
            continue
        if account_id and row.account_id and row.account_id == account_id:
            return row.this_period_value
        if account_nm and row.account_nm and row.account_nm == account_nm:
            return row.this_period_value
    return None


__all__ = [
    "BASE_URL",
    "CorpMapping",
    "DartClient",
    "DartError",
    "FS_CONSOLIDATED",
    "FS_SEPARATE",
    "FinancialRow",
    "FinancialsResponse",
    "REPORT_ANNUAL",
    "REPORT_H1",
    "REPORT_Q1",
    "REPORT_Q3",
    "extract_account_value",
    "parse_corp_code_xml",
]
