"""Surface user-submitted tips to crew agents as additional context.

The tip bot (`src/wise_investor/ingest/`) classifies inbound Telegram
messages into seven categories: ticker, macro, fx, sector, geopolitics,
commodity, none. This module is the read side — when a crew run starts,
it pulls the recent tips relevant to the analyzed symbol and the macro
context, formats them into a `<user_provided_tips>` block, and the
runner threads that block through every agent's user prompt.

Design principles:

  - Tips are HYPOTHESES, not facts. The block is appended outside the
    `<pre_gathered_tool_outputs>` boundary specifically so the
    Universal Citation Rule does not pull tips into the numeric-claim
    audit. Numbers still come from Python tools; tips only shape
    judgment.
  - Idempotent per run. `run_tag` (typically `<SYMBOL>_<YYYYMMDD>_<HHMM>`)
    is recorded on each tip after the run; the next crew on the same
    ticker will skip already-consumed tips. This prevents the same
    user message from re-injecting itself into every report.
  - Fail-soft. A TipStore that doesn't exist or fails to query results
    in an empty block — the crew run continues without tips. Users
    without the tip bot configured see no behavioral change.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Iterable

from wise_investor.ingest.tip_store import Tip, TipStore


logger = logging.getLogger(__name__)


# Categories we surface to the Economist (any agent can read them, but
# this is the bucket the Economist's framing is most likely to use).
_MACRO_CATEGORIES: tuple[str, ...] = (
    "macro",
    "fx",
    "commodity",
    "geopolitics",
    "sector",
)

# Default lookback window. Tips older than this are stale — the user's
# group-chat note from a month ago is unlikely to still be the right
# framing for today's analysis.
DEFAULT_MAX_AGE_DAYS: int = 7

# Hard cap so a noisy ingest period doesn't balloon the prompt.
DEFAULT_MAX_TIPS_PER_BUCKET: int = 8


@dataclass
class TipBundle:
    """All tips fetched for a single crew run."""

    ticker: list[Tip]
    macro: list[Tip]

    def all_tips(self) -> list[Tip]:
        """Flat list — used by mark_consumed_for_run."""
        return list(self.ticker) + list(self.macro)

    @property
    def is_empty(self) -> bool:
        return not self.ticker and not self.macro


def _since_iso(max_age_days: int) -> str:
    cutoff = dt.datetime.now() - dt.timedelta(days=max_age_days)
    return cutoff.strftime("%Y-%m-%dT%H:%M:%S")


def fetch_ticker_tips(
    symbol: str,
    run_tag: str,
    *,
    store: TipStore | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    limit: int = DEFAULT_MAX_TIPS_PER_BUCKET,
) -> list[Tip]:
    """Return ticker-specific tips matching `symbol` that haven't been
    consumed by ANY prior crew run.

    The dedup semantic is "any consumption blocks re-injection" — once
    a tip is fed to a crew, it shouldn't return on the next run for the
    same ticker. The `consumed_by` list still records *which* runs
    saw it (audit trail), but the gating is just "any tag present".

    `run_tag` is required so the caller signals intent to consume; we
    also use it as a guard against accidentally fetching tips when no
    consumption namespace was set up.

    Tips are ordered newest-first (matches list_tips contract) and
    capped at `limit` to keep the prompt bounded.
    """
    if not symbol or not run_tag:
        return []
    if store is None:
        store = TipStore()
    try:
        candidates = store.list_tips(
            ticker=symbol.upper(),
            since=_since_iso(max_age_days),
            limit=limit * 3,  # extra headroom for the consumed-filter
        )
    except Exception as e:
        logger.warning("fetch_ticker_tips failed (%s); returning empty.", e)
        return []
    fresh = [
        t for t in candidates
        if t.category == "ticker" and not t.consumed_by
    ]
    return fresh[:limit]


def fetch_macro_tips(
    run_tag: str,
    *,
    store: TipStore | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    limit: int = DEFAULT_MAX_TIPS_PER_BUCKET,
) -> list[Tip]:
    """Return macro-bucket tips (any non-ticker investment category)
    that haven't been consumed by any prior crew run. Same dedup
    semantic as `fetch_ticker_tips` — once consumed, never re-injected.
    """
    if not run_tag:
        return []
    if store is None:
        store = TipStore()
    try:
        candidates = store.list_tips(
            categories=list(_MACRO_CATEGORIES),
            since=_since_iso(max_age_days),
            limit=limit * 3,
        )
    except Exception as e:
        logger.warning("fetch_macro_tips failed (%s); returning empty.", e)
        return []
    fresh = [t for t in candidates if not t.consumed_by]
    return fresh[:limit]


def fetch_tips_for_run(
    symbol: str,
    run_tag: str,
    *,
    store: TipStore | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> TipBundle:
    """Convenience: return both ticker and macro buckets in one call.
    The runner uses this so it only opens TipStore once per run.
    """
    if store is None:
        store = TipStore()
    return TipBundle(
        ticker=fetch_ticker_tips(
            symbol, run_tag, store=store, max_age_days=max_age_days
        ),
        macro=fetch_macro_tips(
            run_tag, store=store, max_age_days=max_age_days
        ),
    )


def format_tips_block(bundle: TipBundle, symbol: str) -> str:
    """Render the bundle as the `<user_provided_tips>` block.

    Returns an empty string when both buckets are empty so the runner
    can skip the wrapper without conditional logic.

    The block carries its own usage rule — agents must not cite tips
    as numeric sources. We embed the rule alongside the data so the
    LLM sees the constraint right next to the content it constrains.
    """
    if bundle.is_empty:
        return ""

    lines: list[str] = [
        "<user_provided_tips>",
        "These are HUMAN-PROVIDED HYPOTHESES from the analyst's stock-",
        "discussion group, not facts. Treat them as hints to investigate.",
        "Numeric claims must still cite a `<tool_output>` source — do NOT",
        "cite `[Source: user_tip.telegram]` for any number. You may",
        "acknowledge a tip in narrative prose ('a user-submitted tip",
        "flagged X; checking the data, the relevant tool output shows Y').",
        "Tips older than 7 days are not surfaced.",
        "",
    ]

    if bundle.ticker:
        lines.append(f"## Ticker-specific tips ({symbol.upper()})")
        for tip in bundle.ticker:
            lines.append(_format_one_tip(tip, include_tickers=False))
        lines.append("")

    if bundle.macro:
        lines.append("## Macro / sector / geopolitics tips")
        for tip in bundle.macro:
            lines.append(_format_one_tip(tip, include_tickers=False))
        lines.append("")

    lines.append("</user_provided_tips>")
    return "\n".join(lines)


def _format_one_tip(tip: Tip, *, include_tickers: bool) -> str:
    """Render a single tip as one bullet line."""
    timestamp = (tip.received_at or "")[:16]  # YYYY-MM-DDTHH:MM
    sender = f" by @{tip.sender}" if tip.sender else ""
    body = (tip.raw_text or "").replace("\n", " ").strip()
    if len(body) > 280:
        body = body[:277].rstrip() + "…"

    suffix = ""
    if tip.category == "ticker" and include_tickers and tip.detected_tickers:
        suffix = f" [{', '.join(tip.detected_tickers)}]"
    elif tip.category in _MACRO_CATEGORIES:
        topics = ", ".join(tip.topics) if tip.topics else tip.category
        suffix = f" [{tip.category}: {topics}]"

    return f"- {timestamp}{sender}: \"{body}\"{suffix}"


def mark_consumed_for_run(
    tips: Iterable[Tip],
    run_tag: str,
    *,
    store: TipStore | None = None,
) -> int:
    """Append `run_tag` to each tip's consumed_by list. Idempotent.

    Returns the number of tips actually updated. Failures on individual
    tips are logged and skipped — partial failure must not poison the
    crew run that already produced a report.
    """
    if not run_tag:
        return 0
    if store is None:
        store = TipStore()
    n = 0
    for tip in tips:
        try:
            if store.mark_consumed(tip.id, run_tag):
                n += 1
        except Exception as e:
            logger.warning(
                "mark_consumed failed for tip #%s (%s); skipping.",
                tip.id,
                e,
            )
    return n


__all__ = [
    "DEFAULT_MAX_AGE_DAYS",
    "DEFAULT_MAX_TIPS_PER_BUCKET",
    "TipBundle",
    "fetch_macro_tips",
    "fetch_ticker_tips",
    "fetch_tips_for_run",
    "format_tips_block",
    "mark_consumed_for_run",
]
