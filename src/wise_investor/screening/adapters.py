"""JSON-stub adapters for Stage 2 calibration.

The constitution's calibration phase (§23 Step 4) compares the
system's per-axis verdicts against the user's intuitive judgments on
30-50 hand-picked tickers. Doing that against live API data would
mix two failure modes — bad data vs bad rubric — and make the
calibration loop unreadable. Instead, calibration runs against
hand-curated JSON files where every fundamental is explicit and
auditable.

This module provides:

  - `load_fundamentals_from_json(path)` — read a fixture file into
    `TickerFundamentals` + a list of segment-history `SegmentBreakdown`s
  - `dump_fundamentals_template(symbol)` — emit a template JSON for
    seeding a new fixture, with all required fields stubbed to None

Real Finnhub / FMP / DART adapters are deliberately out of scope for
Step 1. They land in Step 5 (universe expansion), where rate limits
and caching policy become primary concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wise_investor.screening.types import (
    AnnualFinancials,
    QuarterlyMargin,
    Segment,
    SegmentBreakdown,
    TickerFundamentals,
)


def _segment_from_dict(d: dict[str, Any]) -> Segment:
    return Segment(
        name=str(d["name"]),
        revenue=_opt_float(d.get("revenue")),
        share_of_total=float(d["share_of_total"]),
    )


def _segment_breakdown_from_dict(d: dict[str, Any]) -> SegmentBreakdown:
    segs = tuple(_segment_from_dict(s) for s in d.get("all_segments", []))
    return SegmentBreakdown(
        primary_segment_exists=bool(d.get("primary_segment_exists", False)),
        primary_segment_name=d.get("primary_segment_name"),
        primary_segment_revenue_share=_opt_float(
            d.get("primary_segment_revenue_share")
        ),
        all_segments=segs,
        fiscal_year=int(d["fiscal_year"]),
        source=str(d.get("source", "stub")),
    )


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    return float(v)


def load_fundamentals_from_json(path: Path | str) -> TickerFundamentals:
    """Read a hand-curated JSON file into a TickerFundamentals.

    Schema: see `dump_fundamentals_template` for the canonical shape.
    The loader is permissive on missing optional fields (they default
    to None) and strict on required ones (`symbol`,
    `industry_classification`).
    """
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    annual = tuple(
        AnnualFinancials(
            fiscal_year=int(row["fiscal_year"]),
            revenue=_opt_float(row.get("revenue")),
            gross_profit=_opt_float(row.get("gross_profit")),
            operating_income=_opt_float(row.get("operating_income")),
            nopat=_opt_float(row.get("nopat")),
            invested_capital=_opt_float(row.get("invested_capital")),
            rd_expense=_opt_float(row.get("rd_expense")),
        )
        for row in raw.get("annual", [])
    )

    quarterly = tuple(
        QuarterlyMargin(
            quarter_id=str(q["quarter_id"]),
            gross_margin=_opt_float(q.get("gross_margin")),
        )
        for q in raw.get("quarterly_margins", [])
    )

    segments = tuple(
        _segment_breakdown_from_dict(sb)
        for sb in raw.get("segments_history", [])
    )

    return TickerFundamentals(
        symbol=str(raw["symbol"]),
        industry_classification=str(raw["industry_classification"]),
        annual=annual,
        quarterly_margins=quarterly,
        segments_history=segments,
        top5_customer_share=_opt_float(raw.get("top5_customer_share")),
        diversification_attempt_signals=int(
            raw.get("diversification_attempt_signals", 0)
        ),
        industry_roic_3y_median=_opt_float(raw.get("industry_roic_3y_median")),
        industry_gross_margin_3y_std=_opt_float(
            raw.get("industry_gross_margin_3y_std")
        ),
    )


def dump_fundamentals_template(symbol: str) -> dict[str, Any]:
    """Return a template dict suitable for seeding a fixture file.

    Calibration workflow:
      1. Run `dump_fundamentals_template("NVDA")`, write to a JSON file.
      2. Hand-fill the values from 10-K + Finnhub + your own peer set.
      3. Feed the JSON back through `load_fundamentals_from_json` and
         the prefilter.
      4. Compare output to your judgment.
    """
    return {
        "symbol": symbol,
        "industry_classification": "GICS Sub-Industry name (Level 3)",
        "annual": [
            {
                "fiscal_year": 2024,
                "revenue": None,
                "gross_profit": None,
                "operating_income": None,
                "nopat": None,
                "invested_capital": None,
                "rd_expense": None,
            },
            # add 2023, 2022, 2021, 2020 entries similarly
        ],
        "quarterly_margins": [
            {"quarter_id": "2024Q4", "gross_margin": None},
            # add 11 prior quarters
        ],
        "segments_history": [
            {
                "fiscal_year": 2024,
                "primary_segment_exists": True,
                "primary_segment_name": "Primary Segment",
                "primary_segment_revenue_share": 0.5,
                "all_segments": [
                    {"name": "Primary Segment", "revenue": None, "share_of_total": 0.5},
                ],
                "source": "stub",
            },
        ],
        "top5_customer_share": None,
        "diversification_attempt_signals": 0,
        "industry_roic_3y_median": None,
        "industry_gross_margin_3y_std": None,
    }


__all__ = [
    "dump_fundamentals_template",
    "load_fundamentals_from_json",
]
