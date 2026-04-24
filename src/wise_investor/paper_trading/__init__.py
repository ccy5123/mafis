"""Phase 4 paper trading — validate Steward verdicts against market returns.

Closes the MVP Q1 loop: Phase 1 evaluation answered "is the report
useful for investment decisions?" structurally (citations, refusals,
discipline). Paper trading answers it empirically — if BUY verdicts
don't outperform PASS verdicts over N weeks, the structural quality
doesn't matter. This package records every Steward verdict at its
issue date and price, then lets you re-price later to compute the
hypothetical P&L.

SQLite tables live in the same `data/portfolio.sqlite` file as the
positions ledger — separate concern, shared storage.
"""

from wise_investor.paper_trading.ledger import (
    PaperTrade,
    PaperTradeLedger,
    TradeReturn,
    PerformanceSummary,
)
from wise_investor.paper_trading.report_parser import (
    CrewReportSummary,
    parse_crew_report,
)

__all__ = [
    "CrewReportSummary",
    "PaperTrade",
    "PaperTradeLedger",
    "PerformanceSummary",
    "TradeReturn",
    "parse_crew_report",
]
