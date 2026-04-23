"""GDELT Project DOC 2.0 client — free, no API key required.

Base: https://api.gdeltproject.org/api/v2/doc/doc

A polite User-Agent is MANDATORY: GDELT returns an empty body `{}` for
requests with the default curl/httpx UA. We set the same
"MAFIS/0.1 personal-research ..." UA we use for SEC EDGAR.

The API exposes many modes; we use two:

  mode=artlist&format=json        List of matching articles with domain /
                                  language / source country metadata.
  mode=timelinetone&format=json   Daily tone score (-100..+100) over time
                                  for a query; used for "is geopolitical
                                  risk rising" narrative context.

Query syntax supports theme filters (`theme:ECON_TRADE_SANCTIONS`), Boolean
OR (`(NVIDIA OR AMD)`), and quoted exact phrases. See:
https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


logger = logging.getLogger(__name__)


BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
USER_AGENT = "MAFIS/0.1 personal-research (github.com/ccy5123/mafis)"


# Pre-curated GDELT GKG 2.0 themes that matter for a long-term equity
# investor. Maps theme key → human label for display. These codes are
# stable identifiers in GDELT's knowledge graph.
GEOPOLITICAL_THEMES: dict[str, str] = {
    "ECON_TRADE_SANCTIONS": "Trade sanctions / embargoes",
    "ECON_EMBARGO": "Economic embargo",
    "TRADE_WAR": "Trade war rhetoric",
    "EPU_POLICY": "Economic policy uncertainty",
    "TAX_POLITICAL_UNREST": "Political unrest with economic effect",
    "MILITARY": "Military activity",
    "TERROR": "Terrorism / security incidents",
    "SLFID_ECON_SANCTIONS": "Self-identified economic sanctions discussion",
}


class GdeltError(RuntimeError):
    pass


@dataclass
class GdeltArticle:
    """One article from the artlist endpoint."""

    url: str
    title: str
    seen_date: str  # GDELT format YYYYMMDDTHHMMSSZ — we store it verbatim
    domain: str
    language: str
    source_country: str

    @property
    def iso_date(self) -> str:
        """Render seen_date as ISO 8601 (YYYY-MM-DDTHH:MM:SSZ) for display."""
        s = self.seen_date
        if len(s) < 15:
            return s
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}Z"


@dataclass
class GdeltTonePoint:
    """One (date, tone) point from the timelinetone endpoint."""

    date: str  # ISO YYYY-MM-DD
    tone: float  # average tone score; positive = optimistic, negative = negative


@dataclass
class GdeltThemeResult:
    """Articles + optional tone trajectory for a single theme query."""

    theme: str
    label: str
    articles: list[GdeltArticle] = field(default_factory=list)
    error: str | None = None


class GdeltClient:
    """Thin sync client for GDELT DOC 2.0.

    No auth, no documented rate limit, but a polite User-Agent is required
    (see module docstring). Callers should still be considerate: one call
    per theme per run, cached daily upstream via pre_gather_facts.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        retry_backoff_sec: float = 1.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.retry_backoff_sec = retry_backoff_sec
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        client_kwargs: dict[str, Any] = {
            "timeout": timeout,
            "headers": {"User-Agent": USER_AGENT},
        }
        if transport is not None:
            client_kwargs["transport"] = transport
        self._client = httpx.Client(**client_kwargs)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "GdeltClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ----- raw fetch with retry --------------------------------------

    def _fetch_json(self, params: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                r = self._client.get(self.base_url, params=params)
            except httpx.TransportError as e:
                last_exc = e
                logger.warning("GDELT transport error (attempt %d): %s", attempt, e)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * attempt)
                continue

            if r.status_code >= 500:
                logger.warning("GDELT %d (attempt %d)", r.status_code, attempt)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_sec * attempt)
                continue
            if r.status_code >= 400:
                raise GdeltError(f"HTTP {r.status_code}: {r.text[:200]}")

            text = r.text.strip()
            if not text or text == "{}":
                # GDELT returns empty JSON for bogus UA or zero results; treat
                # as zero results rather than a hard error. Callers get [].
                return {}
            try:
                return r.json()
            except Exception as e:
                raise GdeltError(f"JSON parse failed: {e}") from e

        raise GdeltError(f"GDELT failed after {self.max_retries} retries: {last_exc}")

    # ----- public API ------------------------------------------------

    def search_articles(
        self,
        query: str,
        timespan: str = "7days",
        max_records: int = 25,
        source_country: str | None = None,
    ) -> list[GdeltArticle]:
        """Return recent articles matching the GDELT query string.

        `timespan` accepts GDELT's relative window syntax (e.g. "24h",
        "7days", "1month"). `source_country` optionally narrows to one
        FIPS/country name (e.g. "US", "CH", "South Korea").
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "timespan": timespan,
            "maxrecords": max_records,
        }
        if source_country:
            params["sourcecountry"] = source_country

        data = self._fetch_json(params)
        raw_articles = data.get("articles", []) if isinstance(data, dict) else []

        out: list[GdeltArticle] = []
        for a in raw_articles:
            try:
                out.append(
                    GdeltArticle(
                        url=str(a.get("url", "")),
                        title=str(a.get("title", "")).strip(),
                        seen_date=str(a.get("seendate", "")),
                        domain=str(a.get("domain", "")),
                        language=str(a.get("language", "")),
                        source_country=str(a.get("sourcecountry", "")),
                    )
                )
            except Exception as e:
                logger.warning("Skipping malformed GDELT article: %s", e)
        return out

    def fetch_tone_timeline(
        self, query: str, timespan: str = "1month"
    ) -> list[GdeltTonePoint]:
        """Return daily average tone scores for a query over `timespan`.

        Response shape from timelinetone mode (JSON):
          {"timeline": [{"series": "...", "data": [{"date": "YYYYMMDDT...", "value": <float>}, ...]}]}
        """
        params: dict[str, Any] = {
            "query": query,
            "mode": "timelinetone",
            "format": "json",
            "timespan": timespan,
        }
        data = self._fetch_json(params)
        timeline = data.get("timeline", []) if isinstance(data, dict) else []
        if not timeline:
            return []

        # Take the first series — GDELT returns one tone series for
        # timelinetone by default.
        first_series = timeline[0] if timeline else {}
        points = first_series.get("data", []) if isinstance(first_series, dict) else []

        out: list[GdeltTonePoint] = []
        for p in points:
            try:
                raw_date = str(p.get("date", ""))
                if len(raw_date) >= 8:
                    iso_date = f"{raw_date[0:4]}-{raw_date[4:6]}-{raw_date[6:8]}"
                else:
                    iso_date = raw_date
                out.append(
                    GdeltTonePoint(date=iso_date, tone=float(p.get("value", 0.0)))
                )
            except Exception as e:
                logger.warning("Skipping malformed tone point: %s", e)
        return out

    def search_theme(
        self,
        theme: str,
        extra_query: str | None = None,
        timespan: str = "7days",
        max_records: int = 15,
    ) -> GdeltThemeResult:
        """Convenience: run `theme:<THEME>` (optionally AND'd with a keyword)
        and wrap the result in a GdeltThemeResult.
        """
        label = GEOPOLITICAL_THEMES.get(theme, theme)
        query = f"theme:{theme}"
        if extra_query:
            query = f"{query} {extra_query}"
        try:
            articles = self.search_articles(
                query=query, timespan=timespan, max_records=max_records
            )
            return GdeltThemeResult(theme=theme, label=label, articles=articles)
        except GdeltError as e:
            logger.warning("GDELT theme %s failed: %s", theme, e)
            return GdeltThemeResult(theme=theme, label=label, articles=[], error=str(e))


__all__ = [
    "BASE_URL",
    "GEOPOLITICAL_THEMES",
    "GdeltArticle",
    "GdeltClient",
    "GdeltError",
    "GdeltThemeResult",
    "GdeltTonePoint",
    "USER_AGENT",
]
