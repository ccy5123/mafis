"""JSON-stub adapter tests."""

from __future__ import annotations

import json

from wise_investor.screening.adapters import (
    dump_fundamentals_template,
    load_fundamentals_from_json,
)


def test_template_roundtrips_through_loader(tmp_path) -> None:
    """The template emitted for seeding a fixture must itself be a
    valid TickerFundamentals when loaded back. Otherwise users hand-
    edit the template and the loader chokes on its own output.
    """
    template = dump_fundamentals_template("NVDA")
    path = tmp_path / "nvda_template.json"
    path.write_text(json.dumps(template), encoding="utf-8")

    funds = load_fundamentals_from_json(path)
    assert funds.symbol == "NVDA"
    assert funds.industry_classification.startswith("GICS")
    # Annual entries with all-None numbers still load.
    assert len(funds.annual) >= 1


def test_loader_handles_missing_optional_fields(tmp_path) -> None:
    """Required fields: symbol, industry_classification. Optional ones
    default cleanly without raising.
    """
    minimal = {
        "symbol": "TEST",
        "industry_classification": "Test Sub-Industry",
    }
    path = tmp_path / "minimal.json"
    path.write_text(json.dumps(minimal), encoding="utf-8")

    funds = load_fundamentals_from_json(path)
    assert funds.symbol == "TEST"
    assert funds.annual == ()
    assert funds.quarterly_margins == ()
    assert funds.segments_history == ()
    assert funds.top5_customer_share is None
    assert funds.diversification_attempt_signals == 0
    assert funds.industry_roic_3y_median is None


def test_loader_parses_full_fixture(tmp_path) -> None:
    """A realistic-shaped fixture (NVDA-like) loads with all fields
    populated correctly.
    """
    fixture = {
        "symbol": "NVDA",
        "industry_classification": "Semiconductors",
        "annual": [
            {
                "fiscal_year": 2022,
                "revenue": 26974.0,
                "gross_profit": 17475.0,
                "operating_income": 4224.0,
                "nopat": 3500.0,
                "invested_capital": 28000.0,
                "rd_expense": 7339.0,
            },
            {
                "fiscal_year": 2023,
                "revenue": 60922.0,
                "gross_profit": 44301.0,
                "operating_income": 32972.0,
                "nopat": 28000.0,
                "invested_capital": 35000.0,
                "rd_expense": 8675.0,
            },
            {
                "fiscal_year": 2024,
                "revenue": 130497.0,
                "gross_profit": 96566.0,
                "operating_income": 81453.0,
                "nopat": 70000.0,
                "invested_capital": 50000.0,
                "rd_expense": 12914.0,
            },
        ],
        "quarterly_margins": [
            {"quarter_id": "2024Q1", "gross_margin": 0.74},
            {"quarter_id": "2024Q2", "gross_margin": 0.76},
            {"quarter_id": "2024Q3", "gross_margin": 0.75},
            {"quarter_id": "2024Q4", "gross_margin": 0.74},
        ],
        "segments_history": [
            {
                "fiscal_year": 2022,
                "primary_segment_exists": True,
                "primary_segment_name": "Compute & Networking",
                "primary_segment_revenue_share": 0.55,
                "all_segments": [
                    {
                        "name": "Compute & Networking",
                        "revenue": 14755.0,
                        "share_of_total": 0.55,
                    },
                    {"name": "Graphics", "revenue": 12219.0, "share_of_total": 0.45},
                ],
                "source": "stub",
            },
        ],
        "top5_customer_share": 0.46,
        "diversification_attempt_signals": 0,
        "industry_roic_3y_median": 0.12,
        "industry_gross_margin_3y_std": 0.04,
    }
    path = tmp_path / "nvda.json"
    path.write_text(json.dumps(fixture), encoding="utf-8")

    funds = load_fundamentals_from_json(path)
    assert funds.symbol == "NVDA"
    assert len(funds.annual) == 3
    assert funds.annual[2].fiscal_year == 2024
    assert funds.annual[2].invested_capital == 50000.0
    assert funds.top5_customer_share == 0.46
    assert funds.industry_roic_3y_median == 0.12
    assert len(funds.segments_history) == 1
    assert funds.segments_history[0].primary_segment_name == "Compute & Networking"
