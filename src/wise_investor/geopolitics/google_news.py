"""Google News RSS fetcher — free, no API key required.

Base URL pattern:
  https://news.google.com/rss/search?q=<QUERY>&hl=en-US&gl=US&ceid=US:en

The feed returns headlines + source attribution only (no body text),
which is exactly what the Economist prompt needs — a short list of
"here's what's in the news" headlines with dates and sources. Full
article bodies are intentionally out of scope; if we need them later,
they should come from a dedicated scraper, not RSS.

Parsing uses Python's stdlib xml.etree (no extra deps).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import httpx


logger = logging.getLogger(__name__)


USER_AGENT = "MAFIS/0.1 personal-research (github.com/ccy5123/mafis)"


class GoogleNewsError(RuntimeError):
    pass


@dataclass
class GoogleNewsItem:
    """One news item parsed from a Google News RSS feed."""

    title: str
    link: str
    published: str  # ISO YYYY-MM-DD when possible, else raw pubDate
    source: str  # e.g. "Reuters", "CNBC"


def build_rss_url(query: str, hl: str = "en-US", gl: str = "US") -> str:
    """Build the Google News RSS URL for a given free-text query.

    Passes query through URL-encoding. Google News accepts Boolean OR and
    quoted phrases in the query string, matching our GDELT conventions.
    """
    q = quote_plus(query)
    return (
        f"https://news.google.com/rss/search?q={q}"
        f"&hl={hl}&gl={gl}&ceid={gl}:{hl.split('-')[0]}"
    )


def _to_iso_date(pub_date: str) -> str:
    """Convert an RFC-822 pubDate string to ISO YYYY-MM-DD when possible.

    Falls back to the raw input if parsing fails.
    """
    if not pub_date:
        return ""
    try:
        dt = parsedate_to_datetime(pub_date)
    except Exception:
        return pub_date
    if dt is None:
        return pub_date
    return dt.date().isoformat()


def parse_rss(xml_text: str) -> list[GoogleNewsItem]:
    """Parse a Google News RSS XML string into a list of items.

    Tolerates missing fields: items with no title AND no link are
    discarded, otherwise we best-effort fill in empty strings.
    """
    if not xml_text.strip():
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise GoogleNewsError(f"RSS XML parse failed: {e}") from e

    channel = root.find("channel")
    if channel is None:
        return []

    items: list[GoogleNewsItem] = []
    for item in channel.findall("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title and not link:
            continue

        pub = (item.findtext("pubDate") or "").strip()
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""

        items.append(
            GoogleNewsItem(
                title=title,
                link=link,
                published=_to_iso_date(pub),
                source=source,
            )
        )
    return items


def fetch_google_news(
    query: str,
    max_items: int = 10,
    timeout: float = 10.0,
    transport: httpx.BaseTransport | None = None,
) -> list[GoogleNewsItem]:
    """Fetch and parse a Google News RSS feed for a free-text query.

    Returns up to `max_items` items in feed order (newest first). Any
    HTTP / parse error raises GoogleNewsError — callers decide whether
    to degrade gracefully or surface the failure.
    """
    url = build_rss_url(query)
    client_kwargs = {"timeout": timeout, "headers": {"User-Agent": USER_AGENT}}
    if transport is not None:
        client_kwargs["transport"] = transport

    with httpx.Client(**client_kwargs) as client:
        try:
            r = client.get(url)
        except httpx.TransportError as e:
            raise GoogleNewsError(f"transport error on {url}: {e}") from e

    if r.status_code >= 400:
        raise GoogleNewsError(f"HTTP {r.status_code} on {url}")

    items = parse_rss(r.text)
    return items[:max_items]


__all__ = [
    "GoogleNewsError",
    "GoogleNewsItem",
    "USER_AGENT",
    "build_rss_url",
    "fetch_google_news",
    "parse_rss",
]
