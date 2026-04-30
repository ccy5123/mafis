"""Back-validation: score the rubric against historical outcomes.

For each (ticker, calibration_date) pair, run the Stage 2 pre-filter
on the as-of-date fundamentals, then compare to outcomes 5 years
later. Two outcome metrics:

  1. **Stock total return vs benchmark.** Did the ticker outperform
     the S&P 500 over the calibration_date → calibration_date+5y
     window? This is a unified signal across all axes — if the
     rubric BUYs and the stock underperforms, the rubric is at
     least partly wrong somewhere.

  2. **Per-axis fundamental persistence.** For tickers that PASSED
     a specific axis at calibration_date, did the axis's
     quantitative claim still hold 5 years later?
        - Moat axis: ROIC advantage at T+5 still ≥ 5pp threshold
        - Bottleneck axis: top-5 customer share still elevated
          (constitution §15 also defines `hhi`, `top3_market_share_sum`,
          `each_top3_member_share`; not computed in v1 — see
          `BottleneckProxies` docstring for the implementation gap)
        - New Frontier axis: harder to verify automatically
          (imitation evidence requires news/analyst data); skipped
          in v1, reported as "verification deferred"

The rubric's correctness shows up in the joint distribution of these
metrics: high BUY rate combined with low outperform rate flags a
permissive rubric; low BUY rate combined with strong "passed-axis-
persisted" rate suggests the rubric is well-calibrated but maybe
too strict on recall.

Outputs are written to a back-validation ledger (JSON) so successive
runs can compare the same constitution version against the same
calibration window — trends in rubric accuracy across constitution
versions become measurable.

This module is intentionally NOT user-intuition-driven. The
calibration loop in §23 Step 4 was rewritten to back-validation
because user-intuition calibration violates Commitment 1 (user
preferences must not influence universe membership) — running the
rubric and then revising it on user-judgment-driven mismatches
re-introduces the bias the constitution is supposed to exclude.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.historical_adapter import (
    fetch_historical_fundamentals,
)
from wise_investor.screening.prefilter import (
    MOAT_ROIC_ADVANTAGE_MIN,
    evaluate_ticker,
)
from wise_investor.screening.proxies import (
    compute_bottleneck_proxies,
    compute_moat_proxies,
)
from wise_investor.screening.types import (
    PrefilterResult,
    TickerFundamentals,
)

logger = logging.getLogger(__name__)


# Default validation horizon — 5 calendar years from calibration_date
# to the outcome measurement point.
DEFAULT_HORIZON_YEARS: int = 5


# Type aliases for injectable fetchers (tests stub these).
PriceReturnFetcher = Callable[[str, dt.date, dt.date], float | None]
"""(symbol, start, end) → cumulative total return as a decimal (e.g. 0.34 = +34%)."""


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StockReturnOutcome:
    """Stock-return outcome over the validation window."""

    symbol: str
    start_date: dt.date
    end_date: dt.date
    ticker_return: float | None
    benchmark_return: float | None
    excess_return: float | None  # ticker_return - benchmark_return


@dataclass(frozen=True)
class AxisPersistenceOutcome:
    """Did a passed axis's quantitative claim still hold at T+horizon?"""

    axis: str
    passed_at_calibration: bool
    persisted_at_horizon: bool | None  # None = couldn't verify
    detail: str  # e.g. "ROIC advantage T+5: 0.07 (>= 0.05 threshold)"


@dataclass(frozen=True)
class TickerBackValidation:
    """Per-ticker back-validation record."""

    symbol: str
    calibration_date: dt.date
    horizon_date: dt.date
    constitution_version: str
    prefilter_result: PrefilterResult
    return_outcome: StockReturnOutcome
    axis_persistence: tuple[AxisPersistenceOutcome, ...]


@dataclass
class BackValidationSummary:
    """Aggregate stats across a back-validation run."""

    constitution_version: str
    calibration_date: dt.date
    horizon_date: dt.date
    n_tickers: int = 0
    n_advanced: int = 0  # Stage 2 cleared the gate
    n_rejected: int = 0
    advanced_avg_excess_return: float | None = None
    rejected_avg_excess_return: float | None = None
    moat_persistence_rate: float | None = None  # of moat-passed, % persisted
    bottleneck_persistence_rate: float | None = None
    per_ticker_records: list[TickerBackValidation] = field(default_factory=list)

    def add(self, record: TickerBackValidation) -> None:
        self.per_ticker_records.append(record)
        self.n_tickers += 1
        if record.prefilter_result.hierarchy_decision == "ADVANCE_TO_STAGE_3":
            self.n_advanced += 1
        else:
            self.n_rejected += 1

    def finalize(self) -> BackValidationSummary:
        """Compute aggregate stats from the per-ticker records."""
        advanced_excess = [
            r.return_outcome.excess_return
            for r in self.per_ticker_records
            if r.prefilter_result.hierarchy_decision == "ADVANCE_TO_STAGE_3"
            and r.return_outcome.excess_return is not None
        ]
        rejected_excess = [
            r.return_outcome.excess_return
            for r in self.per_ticker_records
            if r.prefilter_result.hierarchy_decision == "REJECT"
            and r.return_outcome.excess_return is not None
        ]
        if advanced_excess:
            self.advanced_avg_excess_return = sum(advanced_excess) / len(advanced_excess)
        if rejected_excess:
            self.rejected_avg_excess_return = sum(rejected_excess) / len(rejected_excess)

        # Per-axis persistence.
        moat_records = [
            o
            for r in self.per_ticker_records
            for o in r.axis_persistence
            if o.axis == "moat" and o.passed_at_calibration
        ]
        if moat_records:
            verified = [o for o in moat_records if o.persisted_at_horizon is not None]
            if verified:
                self.moat_persistence_rate = sum(
                    1 for o in verified if o.persisted_at_horizon
                ) / len(verified)

        bn_records = [
            o
            for r in self.per_ticker_records
            for o in r.axis_persistence
            if o.axis == "bottleneck" and o.passed_at_calibration
        ]
        if bn_records:
            verified = [o for o in bn_records if o.persisted_at_horizon is not None]
            if verified:
                self.bottleneck_persistence_rate = sum(
                    1 for o in verified if o.persisted_at_horizon
                ) / len(verified)

        return self


# ---------------------------------------------------------------------------
# Per-axis persistence verifiers
# ---------------------------------------------------------------------------


def _verify_moat_persistence(
    funds_at_horizon: TickerFundamentals,
) -> AxisPersistenceOutcome:
    """ROIC advantage at T+horizon should still be ≥ MOAT_ROIC_ADVANTAGE_MIN."""
    if not funds_at_horizon.annual:
        return AxisPersistenceOutcome(
            axis="moat",
            passed_at_calibration=True,
            persisted_at_horizon=None,
            detail="No annual data available at T+horizon",
        )

    proxies = compute_moat_proxies(funds_at_horizon)
    if proxies.roic_advantage is None:
        return AxisPersistenceOutcome(
            axis="moat",
            passed_at_calibration=True,
            persisted_at_horizon=None,
            detail="Could not compute ROIC advantage at T+horizon",
        )

    persisted = proxies.roic_advantage >= MOAT_ROIC_ADVANTAGE_MIN
    return AxisPersistenceOutcome(
        axis="moat",
        passed_at_calibration=True,
        persisted_at_horizon=persisted,
        detail=(
            f"ROIC advantage at T+horizon: {proxies.roic_advantage:.4f} "
            f"({'>=' if persisted else '<'} {MOAT_ROIC_ADVANTAGE_MIN:.2f})"
        ),
    )


def _verify_bottleneck_persistence(
    funds_at_horizon: TickerFundamentals,
) -> AxisPersistenceOutcome:
    """Top-5 customer share at T+horizon stays elevated.

    yfinance doesn't provide customer concentration, so this verifier
    is mostly going to return "couldn't verify" until the real
    Finnhub/EDGAR adapter lands. Kept here so the contract is
    visible even when v1 can't satisfy it.

    Note: constitution §15 also lists `hhi >= 2500` (concentrated market)
    and oligopoly metrics (`top3_market_share_sum`, `each_top3_member_share`)
    as PASS conditions. None are computed quantitatively in v1 — the
    market-concentration angle is a Stage 3 LLM qualitative check.
    """
    proxies = compute_bottleneck_proxies(funds_at_horizon)
    if proxies.top5_customer_share is None:
        return AxisPersistenceOutcome(
            axis="bottleneck",
            passed_at_calibration=True,
            persisted_at_horizon=None,
            detail=(
                "top-5 customer share unavailable at T+horizon "
                "(yfinance doesn't surface; needs EDGAR Risk Factors parser)"
            ),
        )
    persisted = proxies.top5_customer_share >= 0.40  # mirror the prefilter threshold
    return AxisPersistenceOutcome(
        axis="bottleneck",
        passed_at_calibration=True,
        persisted_at_horizon=persisted,
        detail=(
            f"Top-5 customer share at T+horizon: "
            f"{proxies.top5_customer_share:.2f}"
        ),
    )


def _verify_frontier_persistence() -> AxisPersistenceOutcome:
    """Imitation evidence verification needs news / analyst data we
    don't have automated access to. v1 reports "deferred."
    """
    return AxisPersistenceOutcome(
        axis="new_frontier",
        passed_at_calibration=True,
        persisted_at_horizon=None,
        detail=(
            "imitation-evidence verification requires news / analyst "
            "feeds; deferred until those adapters land"
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def back_validate_ticker(
    symbol: str,
    calibration_date: dt.date,
    *,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    benchmark_symbol: str = "^GSPC",  # S&P 500
    fetcher: Callable | None = None,
    price_return_fetcher: PriceReturnFetcher | None = None,
    cache: bool = True,
    source: str = "finnhub",
    with_industry_aggregates: bool = True,
    with_rag_signals: bool = False,
) -> TickerBackValidation:
    """Run the full back-validation flow on one ticker.

    1. Fetch fundamentals as of calibration_date → run Stage 2 prefilter.
    2. Fetch fundamentals as of horizon_date → axis-persistence checks
       for axes that passed at step 1.
    3. Fetch (ticker, benchmark) returns over [calibration_date, horizon_date].
    4. Build a TickerBackValidation record.

    The `source` argument selects which historical adapter to use:
      - "finnhub" (default): Finnhub /stock/financials-reported with
        exact filed_date filtering. Returns ~15 years of US history;
        recommended for any 5-year horizon back-validation.
      - "yfinance": legacy yfinance-backed adapter. Free of API keys
        but limited to ~4 years of history; insufficient for the
        constitution's §22 5-year horizon. Kept here for tests that
        rely on the original adapter behavior.

    `with_industry_aggregates`: when True (default for source='finnhub'),
    runs the peer aggregator with as_of_date=calibration_date so the
    industry ROIC median used in §10 advantage calculation reflects
    historical peer state, not today's. Set False when running tests
    that don't need network calls or when source='yfinance' (which
    has no peer endpoint).

    `with_rag_signals` (P1a-Full 2026-04): when True (default False),
    runs `extract_rag_signals(symbol)` against the indexed 10-K
    collection and feeds the result into the historical adapter so
    `top5_customer_share` and `diversification_attempt_signals` are
    populated. NOTE — this introduces a known lookahead bias: RAG
    content comes from the *currently indexed* 10-K, not the filing
    available on `calibration_date`. The bias is small (top-5
    customer share is structurally stable across 5 years for most
    issuers) but not zero — diversification signals especially
    reflect events that may not yet have happened at the calibration
    date. Calibration runs that flip this on must record it in the
    ledger entry's `with_rag_signals=True` field for traceability.
    """
    horizon_date = dt.date(
        calibration_date.year + horizon_years,
        calibration_date.month,
        calibration_date.day,
    )

    # KR symbols skip peer aggregation — Finnhub /stock/peers returns
    # 403 on KRX tickers. (Even if it didn't, peer ROIC computation
    # would still fail for KR peers because /stock/financials-reported
    # is also 403 on the same tickers — see _fetch_for_source's KR
    # dispatcher branch.)
    from wise_investor.screening.live_adapter_kr import is_korean_symbol
    industry_aggregates = None
    if (
        with_industry_aggregates
        and source == "finnhub"
        and fetcher is None
        and not is_korean_symbol(symbol)
    ):
        industry_aggregates = _fetch_industry_aggregates_at(
            symbol, calibration_date,
        )

    # KR also skips RAG enrichment — DART filings aren't indexed in the
    # ChromaDB collection (different structure, see rag_signals.py
    # docstring). top5_customer_share/diversification_attempt_signals
    # stay at the default None/0 for KR tickers, exactly as before P1a.
    rag_signals_obj: Any = None
    if (
        with_rag_signals
        and source == "finnhub"
        and fetcher is None
        and not is_korean_symbol(symbol)
    ):
        try:
            from wise_investor.screening.rag_signals import extract_rag_signals
            rag_signals_obj = extract_rag_signals(symbol)
            logger.info(
                "RAG signals for %s: top5=%s, diversif=%s",
                symbol,
                getattr(rag_signals_obj, "top5_customer_share", None),
                getattr(rag_signals_obj, "diversification_attempt_signals", None),
            )
        except Exception as e:
            logger.warning(
                "extract_rag_signals failed for %s: %s — proceeding "
                "without RAG enrichment",
                symbol, e,
            )

    funds_calib = _fetch_for_source(
        source, symbol, calibration_date,
        fetcher=fetcher, cache=cache,
        industry_aggregates=industry_aggregates,
        rag_signals=rag_signals_obj,
    )
    primary_seg = (
        funds_calib.segments_history[-1]
        if funds_calib.segments_history
        else None
    )
    prefilter = evaluate_ticker(funds_calib, primary_seg)

    funds_horizon = _fetch_for_source(
        source, symbol, horizon_date, fetcher=fetcher, cache=cache,
    )
    persistence: list[AxisPersistenceOutcome] = []

    if prefilter.moat.verdict in ("PASS", "NEED_LLM"):
        persistence.append(_verify_moat_persistence(funds_horizon))
    else:
        persistence.append(
            AxisPersistenceOutcome(
                axis="moat",
                passed_at_calibration=False,
                persisted_at_horizon=None,
                detail=f"did not pass moat at calibration ({prefilter.moat.verdict})",
            )
        )

    if prefilter.bottleneck.verdict in ("PASS", "NEED_LLM"):
        persistence.append(_verify_bottleneck_persistence(funds_horizon))
    else:
        persistence.append(
            AxisPersistenceOutcome(
                axis="bottleneck",
                passed_at_calibration=False,
                persisted_at_horizon=None,
                detail=(
                    f"did not pass bottleneck at calibration "
                    f"({prefilter.bottleneck.verdict})"
                ),
            )
        )

    if prefilter.new_frontier.verdict in ("PASS", "NEED_LLM"):
        persistence.append(_verify_frontier_persistence())
    else:
        persistence.append(
            AxisPersistenceOutcome(
                axis="new_frontier",
                passed_at_calibration=False,
                persisted_at_horizon=None,
                detail=(
                    f"did not pass new_frontier at calibration "
                    f"({prefilter.new_frontier.verdict})"
                ),
            )
        )

    # Stock return outcome.
    if price_return_fetcher is None:
        price_return_fetcher = _default_price_return_fetcher

    try:
        ticker_return = price_return_fetcher(symbol, calibration_date, horizon_date)
    except Exception as e:
        logger.warning("ticker return fetch failed for %s: %s", symbol, e)
        ticker_return = None

    try:
        benchmark_return = price_return_fetcher(
            benchmark_symbol, calibration_date, horizon_date
        )
    except Exception as e:
        logger.warning("benchmark return fetch failed: %s", e)
        benchmark_return = None

    excess = (
        ticker_return - benchmark_return
        if ticker_return is not None and benchmark_return is not None
        else None
    )

    return TickerBackValidation(
        symbol=symbol.upper(),
        calibration_date=calibration_date,
        horizon_date=horizon_date,
        constitution_version=CONSTITUTION_VERSION,
        prefilter_result=prefilter,
        return_outcome=StockReturnOutcome(
            symbol=symbol.upper(),
            start_date=calibration_date,
            end_date=horizon_date,
            ticker_return=ticker_return,
            benchmark_return=benchmark_return,
            excess_return=excess,
        ),
        axis_persistence=tuple(persistence),
    )


def back_validate_universe(
    symbols: list[str],
    calibration_date: dt.date,
    *,
    horizon_years: int = DEFAULT_HORIZON_YEARS,
    benchmark_symbol: str = "^GSPC",
    fetcher: Callable | None = None,
    price_return_fetcher: PriceReturnFetcher | None = None,
    cache: bool = True,
    source: str = "finnhub",
    with_rag_signals: bool = False,
) -> BackValidationSummary:
    """Run back_validate_ticker across a list of symbols and aggregate.

    `source` and `with_rag_signals` are forwarded to each ticker (see
    `back_validate_ticker` for valid values).
    """
    horizon_date = dt.date(
        calibration_date.year + horizon_years,
        calibration_date.month,
        calibration_date.day,
    )
    summary = BackValidationSummary(
        constitution_version=CONSTITUTION_VERSION,
        calibration_date=calibration_date,
        horizon_date=horizon_date,
    )
    for s in symbols:
        try:
            record = back_validate_ticker(
                s,
                calibration_date,
                horizon_years=horizon_years,
                benchmark_symbol=benchmark_symbol,
                fetcher=fetcher,
                price_return_fetcher=price_return_fetcher,
                cache=cache,
                source=source,
                with_rag_signals=with_rag_signals,
            )
            summary.add(record)
        except Exception as e:
            logger.warning("back_validate_ticker failed for %s: %s", s, e)
    return summary.finalize()


def _fetch_for_source(
    source: str,
    symbol: str,
    as_of_date: dt.date,
    *,
    fetcher: Callable | None,
    cache: bool,
    industry_aggregates: object | None = None,
    rag_signals: Any = None,
) -> TickerFundamentals:
    """Dispatch to the right historical adapter.

    Tests that pass a custom `fetcher` get yfinance-style behavior
    (the fetcher hook is part of that adapter's API). Production runs
    leave `fetcher=None` and route to Finnhub by default for richer
    history.
    """
    if source == "finnhub" and fetcher is None:
        # P3-1 (2026-04): KR symbols dispatch to the DART historical
        # adapter — Finnhub /stock/financials-reported returns 403 for
        # KRX tickers. The historical KR wrapper synthesizes a
        # `today` so the live adapter's history window aligns with
        # `as_of_date`. Peer aggregation isn't wired for KR yet, so
        # `industry_aggregates` is silently dropped on this path
        # (matches the US historical adapter for yfinance source).
        from wise_investor.screening.live_adapter_kr import (
            fetch_historical_fundamentals_kr,
            is_korean_symbol,
        )
        if is_korean_symbol(symbol):
            return fetch_historical_fundamentals_kr(
                symbol, as_of_date,
                industry_aggregates=industry_aggregates,
            )
        from wise_investor.screening.historical_adapter_finnhub import (
            fetch_historical_fundamentals_finnhub,
        )
        return fetch_historical_fundamentals_finnhub(
            symbol, as_of_date,
            industry_aggregates=industry_aggregates,
            rag_signals=rag_signals,
        )
    # yfinance path (or any custom fetcher injection): the legacy
    # adapter has no industry_aggregates parameter, so the value is
    # silently dropped. Tests that exercise the yfinance path don't
    # care about peer medians anyway.
    return fetch_historical_fundamentals(
        symbol, as_of_date, fetcher=fetcher, cache=cache,
    )


def _fetch_industry_aggregates_at(
    symbol: str, as_of_date: dt.date,
) -> object | None:
    """Run the peer aggregator for a single back-validation ticker.

    Returns an `IndustryAggregates` (or None on failure). Errors are
    swallowed — peer aggregation is an enrichment, not a hard
    requirement. A missing median just means the moat axis stays in
    NEED_LLM territory exactly as it would without this enrichment.
    """
    try:
        from wise_investor.screening.peer_aggregator import (
            compute_industry_aggregates,
        )
        result = compute_industry_aggregates(
            symbol, as_of_date=as_of_date,
        )
        return result.industry_aggregates
    except Exception as e:
        logger.warning(
            "peer_aggregator failed for %s @ %s: %s — proceeding without "
            "industry median",
            symbol, as_of_date, e,
        )
        return None


# ---------------------------------------------------------------------------
# Default price-return fetcher
# ---------------------------------------------------------------------------


def _default_price_return_fetcher(
    symbol: str, start: dt.date, end: dt.date
) -> float | None:
    """Cumulative total return from yfinance over [start, end].

    Returns None when yfinance has no data in the window (delisted,
    too far back, or simply missing). The back-validation summary
    treats None as "couldn't measure" rather than "no return."
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "yfinance is required for back-validation price fetches."
        ) from e

    # yfinance accepts string YYYY-MM-DD; pad end by one day for inclusion.
    history = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=(end + dt.timedelta(days=1)).isoformat(),
        auto_adjust=True,  # adjust for splits and dividends → total return
    )
    if history is None or history.empty or len(history) < 2:
        return None

    first_close = float(history["Close"].iloc[0])
    last_close = float(history["Close"].iloc[-1])
    if first_close <= 0:
        return None
    return (last_close / first_close) - 1.0


__all__ = [
    "AxisPersistenceOutcome",
    "BackValidationSummary",
    "DEFAULT_HORIZON_YEARS",
    "StockReturnOutcome",
    "TickerBackValidation",
    "back_validate_ticker",
    "back_validate_universe",
]
