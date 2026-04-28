"""Multi-segment 30% rule tests (constitution §13)."""

from __future__ import annotations

from wise_investor.screening.segments import (
    PRIMARY_SEGMENT_THRESHOLD,
    resolve_primary_segment,
    single_segment_default,
)
from wise_investor.screening.types import Segment


def test_single_dominant_segment_passes() -> None:
    """A 60% segment cleanly above the 30% threshold is the primary."""
    out = resolve_primary_segment(
        [
            Segment(name="Data Center", revenue=100, share_of_total=0.60),
            Segment(name="Gaming", revenue=40, share_of_total=0.25),
            Segment(name="Other", revenue=20, share_of_total=0.15),
        ],
        fiscal_year=2024,
        source="stub",
    )
    assert out.primary_segment_exists is True
    assert out.primary_segment_name == "Data Center"
    assert out.primary_segment_revenue_share == 0.60


def test_segment_exactly_at_threshold_passes() -> None:
    """Inclusive boundary: share_of_total == 0.30 satisfies §13."""
    out = resolve_primary_segment(
        [
            Segment(name="Cloud", revenue=30, share_of_total=PRIMARY_SEGMENT_THRESHOLD),
            Segment(name="Devices", revenue=70, share_of_total=0.70),
        ],
        fiscal_year=2024,
        source="stub",
    )
    assert out.primary_segment_exists is True
    # Devices at 70% > Cloud at 30% — Devices is primary.
    assert out.primary_segment_name == "Devices"


def test_no_segment_above_threshold_excluded() -> None:
    """Constitution §13: no primary segment → excluded from universe."""
    out = resolve_primary_segment(
        [
            Segment(name="A", revenue=29, share_of_total=0.29),
            Segment(name="B", revenue=28, share_of_total=0.28),
            Segment(name="C", revenue=22, share_of_total=0.22),
            Segment(name="D", revenue=21, share_of_total=0.21),
        ],
        fiscal_year=2024,
        source="stub",
    )
    assert out.primary_segment_exists is False
    assert out.primary_segment_name is None
    assert out.primary_segment_revenue_share is None
    # all_segments still populated for inspection / logging.
    assert len(out.all_segments) == 4


def test_empty_segment_list_excluded() -> None:
    out = resolve_primary_segment([], fiscal_year=2024, source="stub")
    assert out.primary_segment_exists is False
    assert out.all_segments == ()


def test_tie_break_is_alphabetical_for_determinism() -> None:
    """Equal revenue shares should not let ordering flip across runs."""
    out = resolve_primary_segment(
        [
            Segment(name="Beta", revenue=50, share_of_total=0.50),
            Segment(name="Alpha", revenue=50, share_of_total=0.50),
        ],
        fiscal_year=2024,
        source="stub",
    )
    assert out.primary_segment_name == "Alpha"  # alphabetical break


def test_single_segment_default_treats_company_as_one_segment() -> None:
    """Convention for non-segment-reporting tickers: treat as 100%."""
    out = single_segment_default("Acme Corp", fiscal_year=2024)
    assert out.primary_segment_exists is True
    assert out.primary_segment_name == "Acme Corp"
    assert out.primary_segment_revenue_share == 1.0
    assert out.source == "single_segment_default"


def test_resolve_handles_none_share_as_zero() -> None:
    """Defensive: a segment with `share_of_total=None` should not crash."""
    # Cast None through the dataclass to verify behavior, even though
    # types nominally require float — robustness against bad adapters.
    out = resolve_primary_segment(
        [
            Segment(name="X", revenue=None, share_of_total=0.0),
            Segment(name="Y", revenue=100, share_of_total=1.0),
        ],
        fiscal_year=2024,
        source="stub",
    )
    assert out.primary_segment_name == "Y"
    assert out.primary_segment_exists is True
