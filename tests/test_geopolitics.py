"""Tests for the Phase 3E geopolitical data layer.

All HTTP boundaries are stubbed via httpx.MockTransport so the suite
runs fully offline. Real-network smoke tests are opt-in via
`pytest -m network`.
"""

from __future__ import annotations

import json

import httpx
import pytest

from wise_investor.geopolitics import gdelt, google_news, snapshot
from wise_investor.geopolitics.gdelt import (
    GEOPOLITICAL_THEMES,
    GdeltArticle,
    GdeltClient,
    GdeltError,
)
from wise_investor.geopolitics.google_news import (
    GoogleNewsError,
    GoogleNewsItem,
    build_rss_url,
    fetch_google_news,
    parse_rss,
)
from wise_investor.geopolitics.snapshot import (
    DEFAULT_THEMES,
    GeopoliticsSnapshot,
    build_google_news_query,
    format_geopolitics_snapshot,
    get_geopolitics_snapshot,
)


# ---------------------------------------------------------------------------
# GDELT client
# ---------------------------------------------------------------------------


def _gdelt_artlist_fixture() -> dict:
    return {
        "articles": [
            {
                "url": "https://www.reuters.com/world/us-china-tariffs-2026/",
                "url_mobile": "",
                "title": "US imposes new tariffs on Chinese semiconductors",
                "seendate": "20260420T101500Z",
                "socialimage": "https://...",
                "domain": "reuters.com",
                "language": "English",
                "sourcecountry": "United States",
            },
            {
                "url": "https://www.ft.com/content/some-story",
                "url_mobile": "",
                "title": "Export controls tighten on advanced chips",
                "seendate": "20260419T080000Z",
                "socialimage": "https://...",
                "domain": "ft.com",
                "language": "English",
                "sourcecountry": "United Kingdom",
            },
        ]
    }


def _gdelt_timeline_fixture() -> dict:
    return {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"date": "20260418T000000Z", "value": -2.5},
                    {"date": "20260419T000000Z", "value": -3.1},
                    {"date": "20260420T000000Z", "value": -1.8},
                ],
            }
        ]
    }


def _mock_gdelt_client(json_payload: dict | str = None) -> GdeltClient:
    """Build a GdeltClient whose transport returns the given JSON payload."""
    body = (
        json.dumps(json_payload)
        if isinstance(json_payload, dict)
        else (json_payload or "{}")
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    transport = httpx.MockTransport(handler)
    return GdeltClient(transport=transport)


def test_gdelt_search_articles_parses_artlist_fixture() -> None:
    client = _mock_gdelt_client(_gdelt_artlist_fixture())
    articles = client.search_articles(query="theme:ECON_TRADE_SANCTIONS")
    client.close()

    assert len(articles) == 2
    first = articles[0]
    assert isinstance(first, GdeltArticle)
    assert first.domain == "reuters.com"
    assert first.source_country == "United States"
    # ISO conversion drops the quirky GDELT "T" and inserts separators.
    assert first.iso_date == "2026-04-20T10:15:00Z"


def test_gdelt_empty_body_degrades_to_empty_list() -> None:
    # GDELT returns "{}" for requests with bad UA / zero matches — must
    # surface as an empty list, never as an exception.
    client = _mock_gdelt_client({})
    articles = client.search_articles(query="theme:NONE")
    assert articles == []
    client.close()


def test_gdelt_http_400_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad query")

    client = GdeltClient(transport=httpx.MockTransport(handler))
    with pytest.raises(GdeltError):
        client.search_articles(query="theme:BAD")
    client.close()


def test_gdelt_user_agent_is_polite() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["ua"] = request.headers.get("user-agent", "")
        return httpx.Response(200, text=json.dumps({"articles": []}))

    client = GdeltClient(transport=httpx.MockTransport(handler))
    client.search_articles(query="theme:ECON_TRADE_SANCTIONS")
    client.close()

    ua = captured.get("ua", "")
    assert "MAFIS" in ua
    assert "github.com" in ua


def test_gdelt_search_theme_labels_from_registry() -> None:
    client = _mock_gdelt_client(_gdelt_artlist_fixture())
    r = client.search_theme("ECON_TRADE_SANCTIONS")
    client.close()
    assert r.theme == "ECON_TRADE_SANCTIONS"
    assert r.label == GEOPOLITICAL_THEMES["ECON_TRADE_SANCTIONS"]
    assert len(r.articles) == 2
    assert r.error is None


def test_gdelt_tone_timeline_parses_fixture() -> None:
    client = _mock_gdelt_client(_gdelt_timeline_fixture())
    points = client.fetch_tone_timeline(query="theme:ECON_TRADE_SANCTIONS")
    client.close()

    assert len(points) == 3
    assert points[0].date == "2026-04-18"
    assert points[0].tone == -2.5
    assert points[-1].tone == -1.8


# ---------------------------------------------------------------------------
# Google News RSS
# ---------------------------------------------------------------------------


_RSS_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Google News</title>
    <item>
      <title>NVIDIA CEO defends export controls policy</title>
      <link>https://news.google.com/rss/articles/abc123</link>
      <pubDate>Mon, 21 Apr 2026 10:00:00 GMT</pubDate>
      <source url="https://www.reuters.com">Reuters</source>
      <description>Some snippet</description>
    </item>
    <item>
      <title>CHIPS Act funding details released</title>
      <link>https://news.google.com/rss/articles/def456</link>
      <pubDate>Sun, 20 Apr 2026 18:30:00 GMT</pubDate>
      <source url="https://www.cnbc.com">CNBC</source>
      <description>Snippet</description>
    </item>
  </channel>
</rss>
"""


def test_build_rss_url_encodes_query() -> None:
    url = build_rss_url("NVIDIA export controls")
    assert url.startswith("https://news.google.com/rss/search?q=")
    # Space encoded as plus-sign.
    assert "NVIDIA+export+controls" in url
    assert "hl=en-US" in url


def test_parse_rss_returns_items_in_feed_order() -> None:
    items = parse_rss(_RSS_FIXTURE)
    assert len(items) == 2
    assert items[0].title.startswith("NVIDIA CEO")
    assert items[0].source == "Reuters"
    assert items[0].published == "2026-04-21"  # RFC-822 → ISO
    assert items[1].source == "CNBC"


def test_parse_rss_empty_string_is_empty_list() -> None:
    assert parse_rss("") == []


def test_parse_rss_malformed_raises() -> None:
    with pytest.raises(GoogleNewsError):
        parse_rss("<not-xml")


def test_fetch_google_news_respects_max_items() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_RSS_FIXTURE)

    items = fetch_google_news(
        "query", max_items=1, transport=httpx.MockTransport(handler)
    )
    assert len(items) == 1


def test_fetch_google_news_http_error_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server down")

    with pytest.raises(GoogleNewsError):
        fetch_google_news("query", transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# Snapshot aggregator
# ---------------------------------------------------------------------------


def test_build_google_news_query_uses_registry() -> None:
    q = build_google_news_query("NVDA")
    assert '"NVIDIA"' in q
    assert '"NVDA"' in q
    assert "export controls" in q


def test_build_google_news_query_falls_back_to_symbol() -> None:
    q = build_google_news_query("ZZZZ")
    assert q == "ZZZZ"


def test_get_geopolitics_snapshot_happy_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stub Google News fetch
    def _fake_google(query, max_items=10, timeout=10.0, transport=None):
        return [
            GoogleNewsItem(
                title="Story 1",
                link="https://ex.com/1",
                published="2026-04-20",
                source="Reuters",
            )
        ]

    monkeypatch.setattr(snapshot, "fetch_google_news", _fake_google)

    mock_client = _mock_gdelt_client(_gdelt_artlist_fixture())
    snap = get_geopolitics_snapshot(
        symbol="NVDA",
        themes=("ECON_TRADE_SANCTIONS", "TRADE_WAR"),
        gdelt_client=mock_client,
    )
    mock_client.close()

    assert snap.symbol == "NVDA"
    assert snap.has_any_data
    assert len(snap.google_news) == 1
    assert len(snap.gdelt_themes) == 2
    for theme_result in snap.gdelt_themes:
        assert theme_result.articles  # fixture has 2 each call
    assert snap.errors == {}


def test_get_geopolitics_snapshot_google_news_failure_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*a, **k):
        raise GoogleNewsError("network down")

    monkeypatch.setattr(snapshot, "fetch_google_news", _boom)
    mock_client = _mock_gdelt_client(_gdelt_artlist_fixture())
    snap = get_geopolitics_snapshot(
        symbol="NVDA",
        themes=("ECON_TRADE_SANCTIONS",),
        gdelt_client=mock_client,
    )
    mock_client.close()

    assert snap.google_news == []
    assert "google_news" in snap.errors
    # GDELT still populated despite Google News failure.
    assert snap.gdelt_themes[0].articles


def test_get_geopolitics_snapshot_gdelt_failure_per_theme_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        snapshot, "fetch_google_news", lambda *a, **k: []
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream 500")

    mock_client = GdeltClient(
        transport=httpx.MockTransport(handler), retry_backoff_sec=0.0
    )
    snap = get_geopolitics_snapshot(
        symbol="NVDA",
        themes=("ECON_TRADE_SANCTIONS",),
        gdelt_client=mock_client,
    )
    mock_client.close()

    theme_result = snap.gdelt_themes[0]
    assert theme_result.articles == []
    assert theme_result.error  # captured per-theme, not snapshot-level


def test_format_geopolitics_snapshot_renders_all_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = GeopoliticsSnapshot(
        symbol="NVDA",
        generated_at="2026-04-23T10:00:00Z",
        google_news_query='"NVIDIA"',
        google_news=[
            GoogleNewsItem(
                title="Headline one",
                link="https://...",
                published="2026-04-21",
                source="Reuters",
            )
        ],
    )
    # Empty theme still shows up with "(no articles)" line.
    from wise_investor.geopolitics.gdelt import GdeltThemeResult

    snap.gdelt_themes.append(
        GdeltThemeResult(
            theme="ECON_TRADE_SANCTIONS",
            label="Trade sanctions / embargoes",
            articles=[],
        )
    )

    text = format_geopolitics_snapshot(snap)
    assert "NVDA" in text
    assert "Google News" in text
    assert "Reuters" in text
    assert "Headline one" in text
    assert "ECON_TRADE_SANCTIONS" in text
    assert "(no articles" in text


def test_default_themes_all_have_labels() -> None:
    """Every DEFAULT_THEMES key must be discoverable in the registry so
    the probe + format paths can render human-readable labels.
    """
    for theme in DEFAULT_THEMES:
        assert theme in GEOPOLITICAL_THEMES
