"""Construction tests for the Phase 2 Economist agent and FRED client.

Live FRED calls are opt-in via pytest -m network (like the Finnhub tests).
"""

from __future__ import annotations

import pytest

from wise_investor.agents.economist import (
    ECONOMIST_BACKSTORY,
    ECONOMIST_GOAL,
    economist_model,
    make_economist_system_prompt,
)
from wise_investor.agents.tasks import (
    ECONOMIST_REPORT_TEMPLATE,
    make_economist_user_prompt,
)
from wise_investor.config import settings
from wise_investor.data.fred import (
    MACRO_SERIES,
    FredObservation,
    MacroSnapshot,
    _one_year_earlier,
    _yoy_percent,
    format_macro_snapshot,
)


# ---------------------------------------------------------------------------
# Economist agent
# ---------------------------------------------------------------------------


def test_economist_goal_mentions_fred_macro() -> None:
    assert "FRED" in ECONOMIST_GOAL.upper() or "macro" in ECONOMIST_GOAL.lower()


def test_economist_backstory_forbids_stock_verdicts() -> None:
    text = ECONOMIST_BACKSTORY.lower()
    # Economist must not recommend specific stocks or forecast prices.
    assert "do not recommend" in text or "not opine" in text
    # Describes rate cycle state.
    assert "easing" in text and "hiking" in text and "holding" in text


def test_economist_system_prompt_has_discipline() -> None:
    prompt = make_economist_system_prompt().lower()
    assert "economist" in prompt
    assert "macro" in prompt
    # Universal Citation Rule should be referenced.
    assert "citation" in prompt


def test_economist_template_has_four_sections() -> None:
    for heading in [
        "## Rate Cycle",
        "## Inflation",
        "## Real Economy",
        "## FX and Geopolitical Backdrop",
    ]:
        assert heading in ECONOMIST_REPORT_TEMPLATE, f"missing: {heading}"


def test_economist_template_requires_fred_citations() -> None:
    t = ECONOMIST_REPORT_TEMPLATE
    assert "[Source: fred.FEDFUNDS]" in t
    assert "[Source: fred.CPIAUCSL]" in t
    assert "[Source: fred.DEXKOUS]" in t


def test_make_economist_user_prompt_injects_value_chain() -> None:
    prompt = make_economist_user_prompt("NVDA", "Value chain body")
    assert "NVDA" in prompt
    assert "<value_chain_brief>" in prompt
    assert "Value chain body" in prompt


def test_economist_model_not_empty() -> None:
    assert economist_model()


def test_economist_goal_length() -> None:
    assert len(ECONOMIST_GOAL) > 50


# ---------------------------------------------------------------------------
# FRED helpers (offline)
# ---------------------------------------------------------------------------


def test_one_year_earlier_basic() -> None:
    assert _one_year_earlier("2026-04-22") == "2025-04-22"
    assert _one_year_earlier("2024-01-15") == "2023-01-15"


def test_one_year_earlier_tolerates_bad_input() -> None:
    # Degrades to identity on malformed input rather than raising.
    assert _one_year_earlier("not-a-date") == "not-a-date"


def test_yoy_percent_normal_growth() -> None:
    latest = FredObservation(series_id="X", date="2026-01-01", value=110.0)
    prior = FredObservation(series_id="X", date="2025-01-01", value=100.0)
    assert _yoy_percent(latest, prior) == 10.0


def test_yoy_percent_handles_none_inputs() -> None:
    assert _yoy_percent(None, None) is None
    latest = FredObservation(series_id="X", date="2026-01-01", value=110.0)
    assert _yoy_percent(latest, None) is None
    assert _yoy_percent(None, latest) is None


def test_yoy_percent_handles_zero_base() -> None:
    latest = FredObservation(series_id="X", date="2026-01-01", value=5.0)
    prior = FredObservation(series_id="X", date="2025-01-01", value=0.0)
    assert _yoy_percent(latest, prior) is None


def test_macro_series_registry_has_core_entries() -> None:
    for series in ["FEDFUNDS", "CPIAUCSL", "UNRATE", "DEXKOUS", "GDPC1"]:
        assert series in MACRO_SERIES


def test_format_macro_snapshot_handles_all_none() -> None:
    # A completely empty snapshot should still render without raising.
    empty = MacroSnapshot(
        fed_funds_rate=None,
        cpi_latest=None,
        cpi_yoy_percent=None,
        unemployment_rate=None,
        real_gdp_latest=None,
        real_gdp_yoy_percent=None,
        usd_krw_rate=None,
        ten_year_treasury=None,
        ten_year_breakeven_inflation=None,
    )
    text = format_macro_snapshot(empty)
    assert "Macro snapshot" in text
    assert "N/A" in text


def test_format_macro_snapshot_with_values() -> None:
    snap = MacroSnapshot(
        fed_funds_rate=FredObservation(
            series_id="FEDFUNDS", date="2026-03-01", value=4.5, units="Percent"
        ),
        cpi_latest=FredObservation(
            series_id="CPIAUCSL", date="2026-02-01", value=320.0, units="Index 1982-84=100"
        ),
        cpi_yoy_percent=2.7,
        unemployment_rate=None,
        real_gdp_latest=None,
        real_gdp_yoy_percent=None,
        usd_krw_rate=FredObservation(
            series_id="DEXKOUS", date="2026-04-22", value=1380.0, units="Korean won per 1 USD"
        ),
        ten_year_treasury=None,
        ten_year_breakeven_inflation=None,
    )
    text = format_macro_snapshot(snap)
    assert "4.5" in text
    assert "2.70%" in text
    assert "1380" in text


# ---------------------------------------------------------------------------
# Network (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_network_fred_macro_snapshot_fetches_fed_funds() -> None:
    if not settings.fred_api_key or settings.fred_api_key == "your_fred_api_key_here":
        pytest.skip("FRED_API_KEY not set")

    from wise_investor.data.fred import FredClient

    with FredClient() as c:
        obs = c.latest_observation("FEDFUNDS")
    assert obs is not None
    assert obs.series_id == "FEDFUNDS"
    assert obs.value is not None and obs.value > 0
    assert obs.date  # ISO date string
