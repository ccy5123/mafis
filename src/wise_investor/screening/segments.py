"""Segment-level resolution and the §13 multi-segment 30% rule.

The constitution evaluates a company through its *primary segment*
when one exists. We define primary as "the largest segment, when its
revenue share is at least 30% of total." Companies without any
segment crossing 30% are excluded from the universe before Stage 3.
Companies that don't report segments at all are treated as a single
de-facto segment occupying 100% of revenue (so they pass).

This module operates on `SegmentBreakdown` records produced by an
adapter. It is intentionally source-independent — caller hands us
the segment list, we apply the rule.
"""

from __future__ import annotations

from wise_investor.screening.types import Segment, SegmentBreakdown


PRIMARY_SEGMENT_THRESHOLD: float = 0.30


def resolve_primary_segment(
    segments: list[Segment],
    fiscal_year: int,
    source: str,
) -> SegmentBreakdown:
    """Apply §13: identify the primary segment if any crosses 30%.

    `segments` may be empty; the caller is responsible for deciding
    what that means. Empty input here returns a SegmentBreakdown with
    `primary_segment_exists=False` so the caller can fall back to a
    "single segment 100%" convention if desired.
    """
    if not segments:
        return SegmentBreakdown(
            primary_segment_exists=False,
            primary_segment_name=None,
            primary_segment_revenue_share=None,
            all_segments=tuple(),
            fiscal_year=fiscal_year,
            source=source,
        )

    # Sort by share descending, break ties by name for determinism so
    # two equal-revenue segments don't flip the "primary" designation
    # across runs.
    ranked = sorted(
        segments,
        key=lambda s: (-(s.share_of_total or 0.0), s.name),
    )
    top = ranked[0]
    primary_share = top.share_of_total or 0.0

    if primary_share >= PRIMARY_SEGMENT_THRESHOLD:
        return SegmentBreakdown(
            primary_segment_exists=True,
            primary_segment_name=top.name,
            primary_segment_revenue_share=primary_share,
            all_segments=tuple(ranked),
            fiscal_year=fiscal_year,
            source=source,
        )

    # No segment crosses the threshold → conglomerate, exclude.
    return SegmentBreakdown(
        primary_segment_exists=False,
        primary_segment_name=None,
        primary_segment_revenue_share=None,
        all_segments=tuple(ranked),
        fiscal_year=fiscal_year,
        source=source,
    )


def single_segment_default(
    company_name: str, fiscal_year: int, source: str = "single_segment_default"
) -> SegmentBreakdown:
    """Convention for tickers that don't disclose segments at all.

    A small/mid-cap with no segment reporting is, for our purposes,
    a single-segment company. Treating this as `not exists` would
    incorrectly exclude every such ticker. Instead we promote the
    company itself to a 100%-share segment.
    """
    placeholder = Segment(
        name=company_name or "Operations",
        revenue=None,
        share_of_total=1.0,
    )
    return SegmentBreakdown(
        primary_segment_exists=True,
        primary_segment_name=placeholder.name,
        primary_segment_revenue_share=1.0,
        all_segments=(placeholder,),
        fiscal_year=fiscal_year,
        source=source,
    )


__all__ = [
    "PRIMARY_SEGMENT_THRESHOLD",
    "resolve_primary_segment",
    "single_segment_default",
]
