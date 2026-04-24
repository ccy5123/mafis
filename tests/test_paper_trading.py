"""Tests for the Phase 4 paper-trading ledger + report parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.paper_trading.ledger import (
    PaperTrade,
    PaperTradeLedger,
)
from wise_investor.paper_trading.report_parser import (
    CrewReportSummary,
    parse_crew_report,
)


# ---------------------------------------------------------------------------
# Fixture reports
# ---------------------------------------------------------------------------


_CLEAN_BUY_REPORT = """\
# NVDA — Equity Research Note

# Part 6 · Steward

## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
- **NEUTRALIZED**: Cash flow strong [Source: fetch.free_cash_flow].
- **NEUTRALIZED**: Moat durable [Source: edgar.moat_signals].
"""


_DOWNGRADED_REPORT = """\
# NVDA — Equity Research Note

# Part 6 · Steward

## Verdict
BUY

## Conviction Level
Conviction: 4

## Rationale
The LLM speculated.

- **NEUTRALIZED**: Market may support higher multiples.
- **SURVIVED**: Growth unsustainability unchallenged.
"""


_PASS_REPORT = """\
# GEV — Equity Research Note

# Part 6 · Steward

## Verdict
PASS

## Conviction Level
Conviction: 1

## Rationale
- **SURVIVED**: A.
- **SURVIVED**: B.
"""


# ---------------------------------------------------------------------------
# Report parser
# ---------------------------------------------------------------------------


def test_parse_clean_buy_report() -> None:
    s = parse_crew_report(_CLEAN_BUY_REPORT)
    assert s.symbol == "NVDA"
    assert s.verdict == "BUY"
    assert s.conviction == 4
    assert s.original_verdict == "BUY"
    assert s.original_conviction == 4
    assert s.audit_downgraded is False


def test_parse_downgraded_buy_flags_audit() -> None:
    s = parse_crew_report(_DOWNGRADED_REPORT)
    # Raw LLM: BUY / C4. Audit reclassifies the speculative NEUTRALIZED
    # ("may support") to SURVIVED → 0N + 2S → PASS / C1 per the
    # graduated matrix (bear majority).
    assert s.original_verdict == "BUY"
    assert s.original_conviction == 4
    assert s.verdict == "PASS"
    assert s.conviction == 1
    assert s.audit_downgraded is True


def test_parse_pass_report_no_audit_action() -> None:
    s = parse_crew_report(_PASS_REPORT)
    assert s.symbol == "GEV"
    assert s.verdict == "PASS"
    assert s.audit_downgraded is False


def test_parse_symbol_hint_overrides_detection() -> None:
    # Empty title block — the symbol would otherwise be blank; hint fills it.
    text = "# Part 6 · Steward\n\n## Verdict\nBUY\n\n## Conviction Level\nConviction: 3\n"
    s = parse_crew_report(text, symbol_hint="AMD")
    assert s.symbol == "AMD"


def test_parse_missing_verdict_returns_none() -> None:
    text = "# NVDA — Report\n\nNo Steward section.\n"
    s = parse_crew_report(text)
    assert s.symbol == "NVDA"
    assert s.verdict is None


# ---------------------------------------------------------------------------
# PaperTradeLedger CRUD
# ---------------------------------------------------------------------------


def test_record_trade_populates_all_fields(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    trade = ledger.record_trade(
        symbol="NVDA",
        verdict="BUY",
        original_verdict="BUY",
        conviction=4,
        original_conviction=4,
        audit_downgraded=False,
        price_at_verdict=500.0,
        report_path="reports/NVDA_20260424.crew.md",
    )
    assert trade.id >= 1
    assert trade.symbol == "NVDA"
    assert trade.verdict == "BUY"
    assert trade.conviction == 4
    assert trade.audit_downgraded is False
    assert trade.price_at_verdict == 500.0
    # Verdict date defaults to today.
    assert trade.verdict_date  # non-empty


def test_record_trade_rejects_invalid_verdict(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    with pytest.raises(ValueError, match="verdict"):
        ledger.record_trade(symbol="NVDA", verdict="MAYBE", original_verdict="BUY")


def test_record_trade_rejects_invalid_conviction(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    with pytest.raises(ValueError, match="conviction"):
        ledger.record_trade(
            symbol="NVDA", verdict="BUY", original_verdict="BUY", conviction=6
        )


def test_record_trade_preserves_downgrade_flag(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    t = ledger.record_trade(
        symbol="NVDA", verdict="HOLD", original_verdict="BUY",
        conviction=2, original_conviction=4, audit_downgraded=True,
        price_at_verdict=500.0,
    )
    assert t.audit_downgraded is True
    round_trip = ledger.get_trade(t.id)
    assert round_trip is not None
    assert round_trip.audit_downgraded is True


def test_list_trades_filters_by_symbol(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(symbol="NVDA", verdict="BUY", original_verdict="BUY")
    ledger.record_trade(symbol="GEV", verdict="PASS", original_verdict="PASS")
    nvda = ledger.list_trades(symbol="NVDA")
    assert len(nvda) == 1
    assert nvda[0].symbol == "NVDA"


def test_list_trades_filters_by_verdict(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(symbol="NVDA", verdict="BUY", original_verdict="BUY")
    ledger.record_trade(symbol="GEV", verdict="PASS", original_verdict="PASS")
    buys = ledger.list_trades(verdict="BUY")
    assert len(buys) == 1
    assert buys[0].symbol == "NVDA"


def test_delete_trade(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    t = ledger.record_trade(symbol="NVDA", verdict="BUY", original_verdict="BUY")
    assert ledger.delete_trade(t.id) is True
    assert ledger.get_trade(t.id) is None


def test_delete_missing_id_returns_false(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    assert ledger.delete_trade(9999) is False


# ---------------------------------------------------------------------------
# Derived views — returns + performance
# ---------------------------------------------------------------------------


def test_current_returns_computes_pct(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(
        symbol="NVDA", verdict="BUY", original_verdict="BUY",
        price_at_verdict=400.0,
    )
    ret = ledger.current_returns({"NVDA": 500.0})
    assert len(ret) == 1
    # +25% gain
    assert ret[0].return_pct == 25.0


def test_current_returns_handles_missing_price(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(
        symbol="NVDA", verdict="BUY", original_verdict="BUY",
        price_at_verdict=400.0,
    )
    ret = ledger.current_returns({"NVDA": None})
    assert ret[0].return_pct is None


def test_current_returns_handles_missing_entry_price(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(
        symbol="NVDA", verdict="BUY", original_verdict="BUY",
        price_at_verdict=None,
    )
    ret = ledger.current_returns({"NVDA": 500.0})
    assert ret[0].return_pct is None


def test_performance_summary_aggregates_by_verdict(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    # BUY that gained +20%
    ledger.record_trade(
        symbol="A", verdict="BUY", original_verdict="BUY",
        conviction=4, price_at_verdict=100.0,
    )
    # BUY that lost -10%
    ledger.record_trade(
        symbol="B", verdict="BUY", original_verdict="BUY",
        conviction=3, price_at_verdict=100.0,
    )
    # PASS that would have lost -25%
    ledger.record_trade(
        symbol="C", verdict="PASS", original_verdict="PASS",
        conviction=1, price_at_verdict=100.0,
    )

    prices = {"A": 120.0, "B": 90.0, "C": 75.0}
    summary = ledger.performance_summary(prices)

    # BUYs: avg (20 + -10) / 2 = 5%; win rate = 1/2.
    assert summary.by_verdict["BUY"]["n"] == 2
    assert summary.by_verdict["BUY"]["avg_return_pct"] == 5.0
    assert summary.by_verdict["BUY"]["win_rate"] == 0.5
    # PASS: "avoided" a -25% loss.
    assert summary.by_verdict["PASS"]["avg_return_pct"] == -25.0


def test_performance_summary_audit_effect(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    # Original BUY, downgraded by audit → effectively HOLD. Would've gained +15%.
    ledger.record_trade(
        symbol="D", verdict="HOLD", original_verdict="BUY",
        conviction=2, original_conviction=4,
        audit_downgraded=True, price_at_verdict=100.0,
    )
    # Clean BUY that passed audit, lost -5%.
    ledger.record_trade(
        symbol="E", verdict="BUY", original_verdict="BUY",
        conviction=4, original_conviction=4,
        audit_downgraded=False, price_at_verdict=100.0,
    )
    prices = {"D": 115.0, "E": 95.0}
    summary = ledger.performance_summary(prices)
    # Downgraded BUYs averaged +15% (audit "cost" us opportunity cost);
    # clean BUYs averaged -5% (audit would've helped here? no).
    assert summary.audit_effect["downgraded_avg_return_pct"] == 15.0
    assert summary.audit_effect["clean_avg_return_pct"] == -5.0


def test_performance_summary_by_conviction(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    ledger.record_trade(
        symbol="A", verdict="BUY", original_verdict="BUY",
        conviction=5, price_at_verdict=100.0,
    )
    ledger.record_trade(
        symbol="B", verdict="BUY", original_verdict="BUY",
        conviction=2, price_at_verdict=100.0,
    )
    prices = {"A": 130.0, "B": 105.0}
    summary = ledger.performance_summary(prices)
    # C5 should outperform C2 if calibration holds.
    assert summary.by_conviction[5]["avg_return_pct"] == 30.0
    assert summary.by_conviction[2]["avg_return_pct"] == 5.0


def test_performance_summary_zero_trades(tmp_path: Path) -> None:
    ledger = PaperTradeLedger(tmp_path / "pt.sqlite")
    summary = ledger.performance_summary({})
    assert summary.n_trades == 0
    assert summary.by_verdict == {}
    assert summary.by_conviction == {}
