"""DART-backed pre-gather adapter for Korean tickers.

Produces facts dict entries in the SAME shape the existing
`pre_gather_facts` pipeline emits for US tickers (Finnhub path), so
the downstream agent prompts / Universal Citation Rule / quality
metrics don't need to change.

Ticker conventions supported:
  - "005930"       — 6-digit KRX stock code (canonical)
  - "005930.KS"    — Yahoo / Finnhub style suffix; stripped
  - "005930.KQ"    — KOSDAQ suffix; stripped

KRW → USD: DART returns amounts in KRW. For comparability with the
US/Finnhub facts block (dollar values) we include the native KRW
value AND a USD-equivalent using the FRED DEXKOUS rate. Both are
cited so the Skeptic can verify either direction.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from wise_investor.data.dart import (
    DartClient,
    DartError,
    REPORT_ANNUAL,
    extract_account_value,
)


logger = logging.getLogger(__name__)


_KOREAN_SUFFIX_RE = re.compile(r"\.(?:KS|KQ)$", re.IGNORECASE)
_SIX_DIGIT_RE = re.compile(r"^\d{6}$")


# Korean account name → (display_label, candidate_names).
# DART returns account names in Korean; we probe common variants.
_ACCOUNTS: list[tuple[str, list[str]]] = [
    ("revenue", ["매출액", "수익(매출액)", "영업수익"]),
    ("operating_income", ["영업이익", "영업이익(손실)"]),
    ("net_income", ["당기순이익", "당기순이익(손실)"]),
    ("gross_profit", ["매출총이익", "매출총이익(손실)"]),
    ("total_assets", ["자산총계"]),
    ("total_liabilities", ["부채총계"]),
    ("total_equity", ["자본총계"]),
    ("cash_and_cash_equivalents", ["현금및현금성자산"]),
    ("operating_cash_flow", ["영업활동현금흐름", "영업활동으로인한현금흐름"]),
]


def is_korean_ticker(symbol: str) -> bool:
    """True when the symbol looks like a KRX listing (6-digit code
    with or without a .KS / .KQ suffix).
    """
    stripped = _KOREAN_SUFFIX_RE.sub("", symbol).strip()
    return bool(_SIX_DIGIT_RE.match(stripped))


def normalize_korean_symbol(symbol: str) -> str:
    """Strip any .KS / .KQ suffix; zero-pad to 6 digits."""
    stripped = _KOREAN_SUFFIX_RE.sub("", symbol).strip()
    return stripped.zfill(6)


def _krw_to_usd(krw_amount: float | None, usd_krw_rate: float | None) -> float | None:
    """Convert KRW to USD using the KRW-per-USD rate. Returns None if
    either input is missing or rate is non-positive.
    """
    if krw_amount is None or usd_krw_rate is None or usd_krw_rate <= 0:
        return None
    return krw_amount / usd_krw_rate


def _format_value(
    label: str,
    krw: float | None,
    usd: float | None,
) -> str:
    """One fact line: display BOTH KRW (native) and USD (comparable).

    Example output:
        revenue: 300,870,903,000,000 KRW (≈ $205.8B USD at 1461.66 KRW/USD)
    """
    if krw is None:
        return f"{label}: N/A (not found in DART filing)"
    krw_str = f"{krw:,.0f} KRW"
    if usd is None:
        return f"{label}: {krw_str}"
    # Format USD in billions / millions for readability.
    usd_str: str
    abs_usd = abs(usd)
    if abs_usd >= 1e9:
        usd_str = f"${usd/1e9:,.2f}B USD"
    elif abs_usd >= 1e6:
        usd_str = f"${usd/1e6:,.2f}M USD"
    else:
        usd_str = f"${usd:,.0f} USD"
    return f"{label}: {krw_str} (≈ {usd_str})"


def pre_gather_dart_facts(
    symbol: str,
    year: int | str = 2024,
    reprt_code: str = REPORT_ANNUAL,
    usd_krw_rate: float | None = None,
    client: DartClient | None = None,
) -> dict[str, str]:
    """Return a facts dict with `dart.*` keys for a Korean ticker.

    Schema mirrors the Finnhub-path entries (revenue / net_income /
    total_debt etc.) so the agents can consume them without shape
    awareness. A `dart.metadata` entry carries the company identity
    (corp_code, corp_name) for citation traceability.

    `usd_krw_rate` is the KRW-per-USD figure (e.g. 1461.66). When
    provided, every dollar-valued line shows both KRW and USD. Pass
    None to skip the USD conversion.
    """
    normalized = normalize_korean_symbol(symbol)
    facts: dict[str, str] = {}

    owned = False
    if client is None:
        try:
            client = DartClient()
            owned = True
        except DartError as e:
            facts["dart.metadata"] = f"ERROR: {e}"
            return facts

    try:
        # Resolve stock_code → corp_code.
        corp_code = client.corp_code_from_stock_code(normalized)
        if corp_code is None:
            facts["dart.metadata"] = (
                f"ERROR: stock code {normalized} not found in DART registry. "
                "Unlisted entity or wrong market."
            )
            return facts

        # Find company name from the mapping (already loaded via
        # corp_code_from_stock_code).
        mappings = client.load_corp_mapping()
        entry = next(
            (m for m in mappings if m.stock_code == normalized), None
        )
        corp_name = entry.corp_name if entry else "(unknown)"

        facts["dart.metadata"] = (
            f"Symbol: {normalized} / Corp code: {corp_code} / "
            f"Name: {corp_name} / Filing: {year} {reprt_code}"
        )

        # Pull full statements (consolidated).
        try:
            response = client.financials(
                corp_code=corp_code, year=year, reprt_code=reprt_code
            )
        except DartError as e:
            facts["dart.metadata"] += f"\nERROR fetching statements: {e}"
            return facts

        if not response.ok:
            facts["dart.metadata"] += (
                f"\nDART error {response.status}: {response.message}"
            )
            return facts

        # Extract each account. We fill entries even when missing so
        # the agent prompt sees a stable schema.
        for logical_key, candidates in _ACCOUNTS:
            value_krw: float | None = None
            for name in candidates:
                value_krw = extract_account_value(response, account_nm=name)
                if value_krw is not None:
                    break
            usd = _krw_to_usd(value_krw, usd_krw_rate)
            facts[f"dart.{logical_key}"] = _format_value(
                logical_key, value_krw, usd
            )
    finally:
        if owned and client is not None:
            client.close()

    return facts


__all__ = [
    "is_korean_ticker",
    "normalize_korean_symbol",
    "pre_gather_dart_facts",
]
