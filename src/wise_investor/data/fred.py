"""FRED (Federal Reserve Economic Data) client for the Economist agent.

https://fred.stlouisfed.org/docs/api/fred/

Free, unlimited API. One endpoint we care about:
  /fred/series/observations?series_id=...&limit=1&sort_order=desc

Returns the latest observation (value + date) for any FRED series ID.

For Phase 2 we expose a small curated set of macro series that matter for
a US-equity long-term investor based in Korea:

  FEDFUNDS   - Effective Federal Funds Rate, percent, monthly
  CPIAUCSL   - Consumer Price Index (All Urban Consumers), index 1982-84=100
  UNRATE     - Unemployment Rate, percent, monthly
  GDP        - Gross Domestic Product, billions USD, quarterly
  GDPC1      - Real GDP, billions chained USD, quarterly
  DEXKOUS    - South Korea / US Foreign Exchange Rate, daily
  DGS10      - 10-Year Treasury Constant Maturity Rate, percent, daily
  T10YIE     - 10-Year Breakeven Inflation Rate, percent, daily

The Economist agent synthesises narrative from a `MacroSnapshot` dataclass
built by `get_macro_snapshot()` — all dollar/percent values with fresh dates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import BaseModel

from wise_investor.config import settings


logger = logging.getLogger(__name__)


BASE_URL = "https://api.stlouisfed.org/fred"


class FredError(RuntimeError):
    """FRED API returned an error payload or non-retryable HTTP failure."""


class FredObservation(BaseModel):
    """One observation (value + date) for a FRED series."""

    series_id: str
    date: str  # ISO YYYY-MM-DD
    value: float | None
    units: str | None = None  # human label e.g. "Percent", "Billions of Dollars"


# Curated series IDs with human-readable unit labels for narrative use.
MACRO_SERIES: dict[str, dict[str, str]] = {
    "FEDFUNDS": {"label": "Fed Funds Rate", "unit": "Percent"},
    "CPIAUCSL": {"label": "CPI (All Urban Consumers)", "unit": "Index 1982-84=100"},
    "UNRATE": {"label": "Unemployment Rate", "unit": "Percent"},
    "GDPC1": {"label": "Real GDP", "unit": "Billions of chained USD"},
    "DEXKOUS": {"label": "KRW / USD", "unit": "Korean won per 1 USD"},
    "DGS10": {"label": "10-Year Treasury Yield", "unit": "Percent"},
    "T10YIE": {"label": "10-Year Breakeven Inflation", "unit": "Percent"},
}


@dataclass
class MacroSnapshot:
    """A curated snapshot of US macro indicators plus the KR/US FX rate.

    Includes both current levels and, for the price-index series, a
    12-month percent change so the Economist can discuss inflation
    without computing ratios itself.
    """

    fed_funds_rate: FredObservation | None
    cpi_latest: FredObservation | None
    cpi_yoy_percent: float | None
    unemployment_rate: FredObservation | None
    real_gdp_latest: FredObservation | None
    real_gdp_yoy_percent: float | None
    usd_krw_rate: FredObservation | None
    ten_year_treasury: FredObservation | None
    ten_year_breakeven_inflation: FredObservation | None

    def as_dict(self) -> dict[str, Any]:
        """Render for pre-gather injection and human inspection.

        Percent-unit values are formatted with a trailing "%" so that the
        invention_audit metric sees the same string shape in the fact pool
        that the Economist emits in its narrative (e.g., "3.64%").
        """

        def _fmt(obs: FredObservation | None) -> str:
            if obs is None or obs.value is None:
                return "N/A"
            if obs.units and "percent" in obs.units.lower():
                return f"{obs.value:g}% ({obs.units}, as of {obs.date})"
            return f"{obs.value:g} ({obs.units}, as of {obs.date})"

        parts = {
            "Fed Funds Rate": _fmt(self.fed_funds_rate),
            "CPI (latest level)": _fmt(self.cpi_latest),
            "CPI YoY %": "N/A" if self.cpi_yoy_percent is None else f"{self.cpi_yoy_percent:.2f}%",
            "Unemployment Rate": _fmt(self.unemployment_rate),
            "Real GDP (latest)": _fmt(self.real_gdp_latest),
            "Real GDP YoY %": (
                "N/A" if self.real_gdp_yoy_percent is None else f"{self.real_gdp_yoy_percent:.2f}%"
            ),
            "KRW / USD": _fmt(self.usd_krw_rate),
            "10-Year Treasury Yield": _fmt(self.ten_year_treasury),
            "10-Year Breakeven Inflation": _fmt(self.ten_year_breakeven_inflation),
        }
        return parts


class FredClient:
    """Thin sync client over FRED's series/observations endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.fred_api_key
        if not self.api_key or self.api_key == "your_fred_api_key_here":
            raise FredError("FRED API key missing. Set FRED_API_KEY in .env.")
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "FredClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, **params: Any) -> Any:
        full_params = {k: v for k, v in params.items() if v is not None}
        full_params["api_key"] = self.api_key
        full_params["file_type"] = "json"
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.get(f"{self.base_url}{path}", params=full_params)
            except httpx.TransportError as e:
                last_exc = e
                time.sleep(min(2 ** attempt, 4))
                continue

            if r.status_code == 429 or 500 <= r.status_code < 600:
                time.sleep(min(2 ** attempt, 4))
                continue

            if r.status_code >= 400:
                try:
                    body = r.json()
                except Exception:
                    body = r.text[:200]
                raise FredError(f"HTTP {r.status_code} on {path}: {body}")

            try:
                return r.json()
            except Exception as e:
                raise FredError(f"JSON parse failed on {path}: {e}") from e

        raise FredError(f"{path} failed after {self.max_retries} retries: {last_exc}")

    def latest_observation(self, series_id: str) -> FredObservation | None:
        """Return the most recent non-null observation for a series.

        FRED marks missing values as '.' in the JSON response; we skip those
        and walk back until a real value is found (typical depth: 1-3).
        """
        data = self._get(
            "/series/observations",
            series_id=series_id,
            limit=10,
            sort_order="desc",
        )
        observations = data.get("observations", [])
        for obs in observations:
            raw_value = obs.get("value")
            if raw_value in (None, ".", ""):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            meta = MACRO_SERIES.get(series_id, {})
            return FredObservation(
                series_id=series_id,
                date=obs.get("date", ""),
                value=value,
                units=meta.get("unit"),
            )
        return None

    def observation_on_or_before(
        self, series_id: str, target_date: str
    ) -> FredObservation | None:
        """Return an observation at or before `target_date` for YoY calcs."""
        data = self._get(
            "/series/observations",
            series_id=series_id,
            observation_end=target_date,
            limit=10,
            sort_order="desc",
        )
        observations = data.get("observations", [])
        for obs in observations:
            raw_value = obs.get("value")
            if raw_value in (None, ".", ""):
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            meta = MACRO_SERIES.get(series_id, {})
            return FredObservation(
                series_id=series_id,
                date=obs.get("date", ""),
                value=value,
                units=meta.get("unit"),
            )
        return None


def _yoy_percent(latest: FredObservation | None, year_ago: FredObservation | None) -> float | None:
    if latest is None or year_ago is None:
        return None
    if latest.value is None or year_ago.value is None:
        return None
    if year_ago.value == 0:
        return None
    return round((latest.value - year_ago.value) / year_ago.value * 100.0, 3)


def _one_year_earlier(iso_date: str) -> str:
    """YYYY-MM-DD → YYYY-1-MM-DD (same day, previous year)."""
    parts = iso_date.split("-")
    if len(parts) != 3:
        return iso_date
    try:
        year = int(parts[0]) - 1
    except ValueError:
        return iso_date
    return f"{year:04d}-{parts[1]}-{parts[2]}"


def get_macro_snapshot(client: FredClient | None = None) -> MacroSnapshot:
    """Fetch the full curated macro snapshot.

    One network call per series (≈7-9 calls). Free tier, no quota concerns.
    """
    owned = False
    if client is None:
        client = FredClient()
        owned = True

    try:
        fed = client.latest_observation("FEDFUNDS")
        cpi = client.latest_observation("CPIAUCSL")
        cpi_yoy = None
        if cpi is not None:
            year_ago = client.observation_on_or_before(
                "CPIAUCSL", _one_year_earlier(cpi.date)
            )
            cpi_yoy = _yoy_percent(cpi, year_ago)

        unrate = client.latest_observation("UNRATE")

        gdp = client.latest_observation("GDPC1")
        gdp_yoy = None
        if gdp is not None:
            year_ago = client.observation_on_or_before(
                "GDPC1", _one_year_earlier(gdp.date)
            )
            gdp_yoy = _yoy_percent(gdp, year_ago)

        krw = client.latest_observation("DEXKOUS")
        dgs10 = client.latest_observation("DGS10")
        t10yie = client.latest_observation("T10YIE")
    finally:
        if owned:
            client.close()

    return MacroSnapshot(
        fed_funds_rate=fed,
        cpi_latest=cpi,
        cpi_yoy_percent=cpi_yoy,
        unemployment_rate=unrate,
        real_gdp_latest=gdp,
        real_gdp_yoy_percent=gdp_yoy,
        usd_krw_rate=krw,
        ten_year_treasury=dgs10,
        ten_year_breakeven_inflation=t10yie,
    )


def format_macro_snapshot(snapshot: MacroSnapshot) -> str:
    """Render the snapshot as a multi-line string for agent consumption."""
    lines = ["Macro snapshot (latest available from FRED):"]
    for label, value in snapshot.as_dict().items():
        lines.append(f"  - {label}: {value}")
    return "\n".join(lines)
