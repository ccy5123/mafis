"""Per-symbol geopolitical snapshot combining GDELT themes + Google News.

Given a ticker, we need two complementary views of the news landscape:

  1. Structured, theme-indexed event data (GDELT) so the Economist can
     say "ECON_TRADE_SANCTIONS coverage is elevated this week".
  2. Free-text recent headlines (Google News) so the Economist can cite
     specific stories with source + date.

The aggregator is resilient: any single source can fail and the rest
still populate. All failures are logged and recorded on the snapshot as
an `errors` dict so the Economist prompt can explicitly flag which
context is missing.

Ticker → company heuristics live in a small mapping that can grow over
time; for unknown tickers we fall back to the symbol itself as the
keyword.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from wise_investor.geopolitics.gdelt import (
    GEOPOLITICAL_THEMES,
    GdeltClient,
    GdeltError,
    GdeltThemeResult,
)
from wise_investor.geopolitics.google_news import (
    GoogleNewsError,
    GoogleNewsItem,
    fetch_google_news,
)


logger = logging.getLogger(__name__)


# Minimal symbol → (display name, extra keyword list) registry used when
# building Google News queries. Each ticker contributes its company name
# plus a small set of domain-relevant phrases (e.g. "export controls"
# for semiconductor names). Extend as more tickers join the universe.
SYMBOL_KEYWORDS: dict[str, tuple[str, list[str]]] = {
    "NVDA": (
        "NVIDIA",
        ["export controls", "CHIPS Act", "semiconductor sanctions"],
    ),
    "AMD": ("AMD", ["export controls", "CHIPS Act"]),
    "TSM": (
        "TSMC",
        ["Taiwan tension", "semiconductor sanctions", "export controls"],
    ),
    "ASML": ("ASML", ["EUV export controls", "Dutch chip export"]),
    "AVGO": ("Broadcom", ["export controls"]),
    "INTC": ("Intel", ["CHIPS Act", "export controls"]),
    "MU": ("Micron", ["memory export controls"]),
    "GEV": (
        "GE Vernova",
        ["power grid", "energy policy", "IRA tax credits"],
    ),
    "ETN": ("Eaton", ["power grid", "data center power"]),
    "AAPL": ("Apple", ["China supply chain", "tariff"]),
    "MSFT": ("Microsoft", ["EU regulation", "AI regulation"]),
    "GOOGL": ("Google", ["antitrust", "EU regulation"]),
    "META": ("Meta", ["EU regulation", "content moderation"]),
    "AMZN": ("Amazon", ["antitrust", "labor action"]),
    "TSLA": ("Tesla", ["EV tariff", "China EV"]),
}


# Which GDELT themes to probe by default. Keep this short — each entry
# is a separate network call, and the Economist does not need dozens.
# Economic themes only; MILITARY/TERROR were removed from defaults
# because theme-only queries return high-noise results unrelated to any
# specific equity (Serbian artillery drills, Czech army rankings, etc.).
# Users who want military/terror context can pass --themes explicitly to
# the probe script.
DEFAULT_THEMES: tuple[str, ...] = (
    "ECON_TRADE_SANCTIONS",
    "TRADE_WAR",
    "EPU_POLICY",
)


@dataclass
class GeopoliticsSnapshot:
    """Assembled geopolitical context for one symbol.

    Fields are all best-effort: any missing source is reflected in
    `errors` so the caller can annotate the report.
    """

    symbol: str
    generated_at: str  # ISO YYYY-MM-DDTHH:MM:SSZ
    gdelt_themes: list[GdeltThemeResult] = field(default_factory=list)
    google_news: list[GoogleNewsItem] = field(default_factory=list)
    google_news_query: str | None = None
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def has_any_data(self) -> bool:
        if self.google_news:
            return True
        for theme in self.gdelt_themes:
            if theme.articles:
                return True
        return False


def _now_iso_z() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_google_news_query(symbol: str) -> str:
    """Construct a Boolean OR query for Google News from the symbol registry.

    Unknown tickers fall back to just the symbol itself.
    """
    entry = SYMBOL_KEYWORDS.get(symbol.upper())
    if entry is None:
        return symbol.upper()
    company, extra = entry
    parts = [f'"{company}"', f'"{symbol.upper()}"']
    parts.extend(f'"{kw}"' for kw in extra)
    return " OR ".join(parts)


def get_geopolitics_snapshot(
    symbol: str,
    themes: tuple[str, ...] = DEFAULT_THEMES,
    gdelt_timespan: str = "7days",
    gdelt_max_per_theme: int = 8,
    google_max_items: int = 10,
    gdelt_client: GdeltClient | None = None,
) -> GeopoliticsSnapshot:
    """Build a full GeopoliticsSnapshot for one symbol.

    Fails soft per source: if GDELT is unreachable or Google News RSS
    returns garbage, the snapshot still contains whatever other source
    succeeded, with the failure text in `errors`.
    """
    symbol = symbol.upper()
    snapshot = GeopoliticsSnapshot(symbol=symbol, generated_at=_now_iso_z())

    # ---- Google News (one call, fast) -----------------------------------
    query = build_google_news_query(symbol)
    snapshot.google_news_query = query
    try:
        snapshot.google_news = fetch_google_news(query, max_items=google_max_items)
    except GoogleNewsError as e:
        logger.warning("Google News fetch failed for %s: %s", symbol, e)
        snapshot.errors["google_news"] = str(e)
    except Exception as e:
        logger.warning("Google News unexpected error for %s: %s", symbol, e)
        snapshot.errors["google_news"] = str(e)

    # ---- GDELT (one call per theme) -------------------------------------
    owned = False
    if gdelt_client is None:
        try:
            gdelt_client = GdeltClient()
            owned = True
        except Exception as e:
            snapshot.errors["gdelt_init"] = str(e)
            return snapshot

    try:
        for theme in themes:
            try:
                result = gdelt_client.search_theme(
                    theme=theme,
                    timespan=gdelt_timespan,
                    max_records=gdelt_max_per_theme,
                )
            except GdeltError as e:
                logger.warning("GDELT theme %s failed: %s", theme, e)
                result = GdeltThemeResult(
                    theme=theme,
                    label=GEOPOLITICAL_THEMES.get(theme, theme),
                    articles=[],
                    error=str(e),
                )
            snapshot.gdelt_themes.append(result)
    finally:
        if owned:
            gdelt_client.close()

    return snapshot


def format_geopolitics_snapshot(snapshot: GeopoliticsSnapshot) -> str:
    """Render the snapshot as a plain-text block for LLM consumption.

    Same shape as format_macro_snapshot() — a heading, then bulletted
    sub-sections. Each headline carries its own source and date so the
    model can copy-cite.
    """
    lines = [f"Geopolitical snapshot for {snapshot.symbol} (generated {snapshot.generated_at}):"]

    if snapshot.errors:
        lines.append("")
        lines.append("Data-source errors (context partial):")
        for src, msg in snapshot.errors.items():
            lines.append(f"  - {src}: {msg}")

    # Google News section
    lines.append("")
    lines.append(f'Google News headlines (query: {snapshot.google_news_query}):')
    if not snapshot.google_news:
        lines.append("  - (no items)")
    else:
        for item in snapshot.google_news:
            date = item.published or "undated"
            src = item.source or "unknown source"
            lines.append(f"  - [{date}] {item.title} — {src}")

    # GDELT themes
    for theme_result in snapshot.gdelt_themes:
        lines.append("")
        header = f"GDELT theme {theme_result.theme} ({theme_result.label}):"
        lines.append(header)
        if theme_result.error:
            lines.append(f"  - ERROR: {theme_result.error}")
            continue
        if not theme_result.articles:
            lines.append("  - (no articles this window)")
            continue
        for art in theme_result.articles:
            date = art.iso_date[:10] if art.iso_date else "undated"
            country = art.source_country or "?"
            lines.append(
                f"  - [{date}] {art.title} — {art.domain} ({country})"
            )

    return "\n".join(lines)


__all__ = [
    "DEFAULT_THEMES",
    "GeopoliticsSnapshot",
    "SYMBOL_KEYWORDS",
    "build_google_news_query",
    "format_geopolitics_snapshot",
    "get_geopolitics_snapshot",
]
