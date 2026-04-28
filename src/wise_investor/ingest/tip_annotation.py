"""Read-only tip annotation surface (constitution Sec 7).

Constitution v2.0 § 7 redefined the Telegram tip channel: tips are
LOGGED for analytical posterity, but they NEVER trigger analysis and
they NEVER enter LLM context. The remaining purpose of the tip log is:

  1. **Annotation surface.** When a ticker survives screening on its
     own merits, the system attaches an annotation reading "user
     mentioned this N days ago" to the output. This is metadata for
     the *user* to read, not context delivered to any agent.

  2. **Gap analysis.** Compute the symmetric difference between
     "tickers the user has mentioned recently" and "tickers the
     system surfaced." A gap that's systematically biased one way is
     itself useful — the user's attention ↔ rubric mismatch is the
     calibration signal that the original tip-bot (which fed tips
     into prompts) destroyed by collapsing the two sides.

This module is a pure read-only consumer of `TipStore`. It writes
nothing back to the store. Tests stub the store; production code
hands in a real `TipStore` instance.

Importantly: nothing here returns text destined for an LLM prompt.
Callers that need to display annotations format them for the human
audience (terminal output, web UI, report generator). If a future
feature wants to feed "the user mentioned this" into a model prompt,
it must NOT do so via this module — see Commitment 1 (user
preferences must not influence universe membership) and § 7's
explicit rule that the annotation is metadata, NOT prompt context.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Protocol

from wise_investor.ingest.tip_store import Tip

logger = logging.getLogger(__name__)


# Default annotation lookback. 90 days is a compromise between "recent
# enough to be a current attention signal" and "long enough to capture
# tips the user may have forgotten about."
DEFAULT_WINDOW_DAYS: int = 90

# Length of the sample text snippet attached to each annotation.
SAMPLE_TEXT_MAX_CHARS: int = 200


# ---------------------------------------------------------------------------
# Storage protocol — minimal interface this module needs
# ---------------------------------------------------------------------------


class TipReader(Protocol):
    """Subset of `TipStore` used here.

    Production passes a real `TipStore`; tests pass an in-memory stub.
    Keeping the interface small means we don't accidentally couple to
    write-side methods that would let an annotator mutate the log.
    """

    def list_tips(
        self,
        ticker: str | None = None,
        category: str | None = None,
        categories: list[str] | None = None,
        topic: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Tip]: ...


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TipAnnotation:
    """Per-ticker tip metadata for the user to read."""

    ticker: str
    n_mentions: int
    first_mention_days_ago: int   # furthest-back mention in window
    last_mention_days_ago: int    # most recent mention in window
    sample_text: str              # short excerpt from the most recent mention

    def render(self) -> str:
        """Single-line human-readable summary.

        Example: "user mentioned 3× in last 60d (last 5d ago)"
        """
        if self.n_mentions == 1:
            return (
                f"user mentioned {self.last_mention_days_ago}d ago"
            )
        return (
            f"user mentioned {self.n_mentions}× in last "
            f"{self.first_mention_days_ago}d "
            f"(last {self.last_mention_days_ago}d ago)"
        )


@dataclass(frozen=True)
class GapReport:
    """Symmetric-difference report between system surfaces and user mentions.

    `mentioned_and_surfaced`: both sides agree these are interesting.
    `mentioned_only`: user attention pattern that the rubric DIDN'T
        confirm. Worth reviewing — either rubric blind spot or user noise.
    `surfaced_only`: system found these on merit, user hasn't noticed
        them. The constitution's main reason to exist (find ideas
        outside user attention).
    """

    mentioned_and_surfaced: tuple[str, ...]
    mentioned_only: tuple[str, ...]
    surfaced_only: tuple[str, ...]
    by_ticker_mentions: dict[str, int]
    window_days: int

    @property
    def n_mentioned(self) -> int:
        return len(self.mentioned_only) + len(self.mentioned_and_surfaced)

    @property
    def n_surfaced(self) -> int:
        return len(self.surfaced_only) + len(self.mentioned_and_surfaced)

    @property
    def overlap_ratio(self) -> float:
        """Fraction of mentioned tickers that the system also surfaced.

        Returns 0.0 when no mentions in window (ill-defined ratio).
        """
        if self.n_mentioned == 0:
            return 0.0
        return len(self.mentioned_and_surfaced) / self.n_mentioned


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def lookup_tip_annotations(
    tickers: list[str],
    store: TipReader,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    today: dt.date | None = None,
) -> dict[str, TipAnnotation]:
    """For each ticker, summarize tips received in the lookback window.

    Returns a dict keyed by ticker. Tickers with no mentions in the
    window are OMITTED (not included with empty annotations) so the
    caller can iterate the dict to render only those that have content.

    Args:
        tickers: Tickers to look up. Case is normalized to uppercase
            since TipStore stores detected_tickers uppercase.
        store: Anything matching `TipReader` (typically a `TipStore`).
        window_days: How far back to look. Default 90 days.
        today: Override for "today"; tests inject a fixed date.
    """
    today = today or dt.date.today()
    since = (today - dt.timedelta(days=window_days)).isoformat()

    out: dict[str, TipAnnotation] = {}
    for raw in tickers:
        ticker = raw.upper().strip()
        if not ticker:
            continue
        try:
            tips = store.list_tips(ticker=ticker, since=since)
        except Exception as e:
            logger.warning("tip lookup failed for %s: %s", ticker, e)
            continue
        if not tips:
            continue

        annotation = _summarize_mentions(ticker, tips, today=today)
        if annotation is not None:
            out[ticker] = annotation

    return out


def compute_gap_analysis(
    surfaced_tickers: list[str],
    store: TipReader,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    today: dt.date | None = None,
) -> GapReport:
    """Compare system-surfaced tickers against user-mentioned tickers.

    Args:
        surfaced_tickers: Tickers the screening pipeline surfaced
            (e.g., ADVANCE_TO_STAGE_3 / ADVANCE_TO_STAGE_4 / Stage 4
            survivors). Case-normalized to uppercase here.
        store: TipReader instance.
        window_days: Lookback window for "user mentioned recently."
        today: Test-injectable today.
    """
    today = today or dt.date.today()
    since = (today - dt.timedelta(days=window_days)).isoformat()

    surfaced_set = {t.upper().strip() for t in surfaced_tickers if t.strip()}

    # Pull every tip in the window with at least one detected ticker.
    # We deliberately do NOT filter by category="ticker": a macro tip
    # that incidentally mentions ('Fed cut → INTC and AMD') still
    # counts as "user attention on INTC/AMD" for gap purposes. This
    # matches the annotation surface, which also looks at all
    # categories — keeping the two consistent so the same ticker can't
    # appear in one report and not the other.
    try:
        recent_tips = store.list_tips(since=since)
    except Exception as e:
        logger.warning("gap analysis: list_tips failed: %s", e)
        recent_tips = []

    by_ticker: dict[str, int] = {}
    for tip in recent_tips:
        for raw in tip.detected_tickers:
            t = str(raw).upper().strip()
            if not t:
                continue
            by_ticker[t] = by_ticker.get(t, 0) + 1

    mentioned_set = set(by_ticker.keys())
    overlap = mentioned_set & surfaced_set
    mentioned_only = mentioned_set - surfaced_set
    surfaced_only = surfaced_set - mentioned_set

    return GapReport(
        mentioned_and_surfaced=tuple(sorted(overlap)),
        mentioned_only=tuple(sorted(mentioned_only)),
        surfaced_only=tuple(sorted(surfaced_only)),
        by_ticker_mentions=dict(sorted(by_ticker.items())),
        window_days=window_days,
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _summarize_mentions(
    ticker: str, tips: list[Tip], *, today: dt.date
) -> TipAnnotation | None:
    """Project a list of tips for one ticker into a TipAnnotation."""
    if not tips:
        return None

    days_ago_list: list[int] = []
    for tip in tips:
        d = _parse_received_date(tip.received_at)
        if d is None:
            continue
        days_ago_list.append((today - d).days)
    if not days_ago_list:
        return None

    # tips list is newest-first per TipStore convention; sample from [0].
    sample = tips[0].raw_text or ""
    if len(sample) > SAMPLE_TEXT_MAX_CHARS:
        sample = sample[:SAMPLE_TEXT_MAX_CHARS] + "…"

    return TipAnnotation(
        ticker=ticker,
        n_mentions=len(days_ago_list),
        first_mention_days_ago=max(days_ago_list),
        last_mention_days_ago=min(days_ago_list),
        sample_text=sample,
    )


def _parse_received_date(received_at: str) -> dt.date | None:
    """Parse the TipStore's ISO timestamp into a date.

    Tolerant of date-only ('2026-04-15') and datetime
    ('2026-04-15T11:37:00') variants, since the schema stores both.
    """
    if not received_at:
        return None
    try:
        return dt.date.fromisoformat(received_at[:10])
    except ValueError:
        return None


__all__ = [
    "DEFAULT_WINDOW_DAYS",
    "GapReport",
    "TipAnnotation",
    "TipReader",
    "compute_gap_analysis",
    "lookup_tip_annotations",
]
