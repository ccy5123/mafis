"""yfinance wrapper — minimal cross-validation source for Phase 1.

Per design-v2.2 §3.2, yfinance is the fallback/cross-check for FMP, not the
primary source. It scrapes Yahoo Finance so it can break without warning —
this wrapper therefore isolates the quirks and always returns None on failure
rather than raising, so the caller can decide how to handle missing data.

Only fields actually used for cross-validation are exposed.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel


logger = logging.getLogger(__name__)


class YFQuote(BaseModel):
    symbol: str
    price: float | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    shares_outstanding: float | None = None


def _first_not_none(mapping: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = mapping.get(k)
        if v is not None:
            return v
    return None


def get_quote_snapshot(symbol: str) -> YFQuote:
    """Return a best-effort snapshot from yfinance.

    yfinance is imported lazily so tests that mock it can run without the
    heavyweight import triggering first.
    """
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except Exception as e:
        logger.warning("yfinance import failed: %s", e)
        return YFQuote(symbol=symbol)

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:
        logger.warning("yfinance lookup failed for %s: %s", symbol, e)
        return YFQuote(symbol=symbol)

    price = _first_not_none(info, "currentPrice", "regularMarketPrice", "previousClose")
    return YFQuote(
        symbol=symbol,
        price=float(price) if price is not None else None,
        market_cap=_coerce_float(info.get("marketCap")),
        pe_ratio=_coerce_float(info.get("trailingPE")),
        shares_outstanding=_coerce_float(info.get("sharesOutstanding")),
    )


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # yfinance sometimes returns NaN via pandas; reject these so cross-validation
    # doesn't get confused.
    if f != f:  # NaN check
        return None
    return f
