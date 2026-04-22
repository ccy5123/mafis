"""verify_number — the Skeptic's number-checking tool (Finnhub-backed).

Post Phase 1B migration: raw dollar values come from /stock/financials-reported
via XBRL concept extraction; computed metrics call our own calculate_per /
calculate_ev_ebitda / reverse_dcf functions (which also use Finnhub).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from wise_investor.data.finnhub import (
    FinnhubClient,
    derive_ebitda,
    derive_free_cash_flow,
    extract_field,
    total_debt as fn_total_debt,
)
from wise_investor.tools.dcf import reverse_dcf
from wise_investor.tools.valuation import calculate_ev_ebitda, calculate_per


DEFAULT_TOLERANCE_PCT = 1.0


class VerificationResult(BaseModel):
    field: str
    claim: float
    source_value: float | None
    matches: bool | None
    diff_pct: float | None
    tolerance_pct: float
    source_citation: str
    warnings: list[str] = Field(default_factory=list)


FIELD_ALIASES = {
    "pe": "per",
    "p/e": "per",
    "price_to_earnings": "per",
    "ev/ebitda": "ev_ebitda",
    "opcf": "operating_cash_flow",
    "ocf": "operating_cash_flow",
    "capex": "capital_expenditure",
    "fcf": "free_cash_flow",
    "reverse_dcf_growth": "implied_growth_rate",
}

_QUOTE_FIELDS = {"price", "market_cap"}
_INCOME_FIELDS = {
    "revenue",
    "net_income",
    "operating_income",
    "gross_profit",
    "eps_diluted",
    "eps_basic",
    "depreciation_and_amortization",
}
_BALANCE_FIELDS = {
    "total_debt",
    "total_assets",
    "total_stockholders_equity",
    "cash_and_cash_equivalents",
}
_CASHFLOW_FIELDS = {
    "free_cash_flow",
    "operating_cash_flow",
    "capital_expenditure",
}
_DERIVED_INCOME_FIELDS = {"ebitda"}
_EV_FIELDS = {"enterprise_value"}
_COMPUTED_FIELDS = {"per", "ev_ebitda", "implied_growth_rate"}


_SUPPORTED = (
    _QUOTE_FIELDS
    | _INCOME_FIELDS
    | _BALANCE_FIELDS
    | _CASHFLOW_FIELDS
    | _DERIVED_INCOME_FIELDS
    | _EV_FIELDS
    | _COMPUTED_FIELDS
)


def _fetch_source(
    field: str, symbol: str, client: FinnhubClient
) -> tuple[float | None, str, list[str]]:
    warnings: list[str] = []
    sym = symbol.upper()

    if field == "price":
        q = client.quote(sym)
        return q.price, f"Finnhub /quote current for {sym} — price", warnings

    if field == "market_cap":
        p = client.profile(sym)
        return p.market_cap_usd, f"Finnhub /stock/profile2 marketCapitalization for {sym}", warnings

    if field == "enterprise_value":
        m = client.metric(sym)
        return (
            m.metric.enterprise_value_usd,
            f"Finnhub /stock/metric enterpriseValue for {sym}",
            warnings,
        )

    if field in _INCOME_FIELDS | _BALANCE_FIELDS | _CASHFLOW_FIELDS:
        latest = client.latest_annual_financials(sym)
        if latest is None:
            warnings.append("no annual financials")
            return None, f"Finnhub /stock/financials-reported for {sym} (no annual entry)", warnings

        if field == "total_debt":
            v = fn_total_debt(latest)
        elif field == "free_cash_flow":
            v = derive_free_cash_flow(latest)
        else:
            v = extract_field(latest, field)

        citation = (
            f"Finnhub /stock/financials-reported 10-K end_date={latest.end_date} "
            f"— {field}"
        )
        return v, citation, warnings

    if field == "ebitda":
        latest = client.latest_annual_financials(sym)
        if latest is None:
            warnings.append("no annual financials")
            return None, f"Finnhub /stock/financials-reported for {sym} (no annual)", warnings
        v = derive_ebitda(latest)
        return (
            v,
            f"Finnhub /stock/financials-reported 10-K end_date={latest.end_date} "
            f"— EBITDA derived (OperatingIncome + D&A)",
            warnings,
        )

    if field == "per":
        r = calculate_per(sym, client=client)
        warnings.extend(r.warnings)
        citation = (
            f"calculate_per({sym}) = price/{r.inputs.get('eps_diluted_latest_annual')} "
            f"as of {r.as_of}"
        )
        return r.computed, citation, warnings

    if field == "ev_ebitda":
        r = calculate_ev_ebitda(sym, client=client)
        warnings.extend(r.warnings)
        citation = (
            f"calculate_ev_ebitda({sym}) = "
            f"{r.inputs.get('enterprise_value')}/{r.inputs.get('ebitda_latest_annual')} "
            f"as of {r.as_of}"
        )
        return r.computed, citation, warnings

    if field == "implied_growth_rate":
        r = reverse_dcf(sym, client=client)
        warnings.extend(r.warnings)
        citation = (
            f"reverse_dcf({sym}) with "
            f"discount={r.inputs.get('discount_rate')}, "
            f"terminal_growth={r.inputs.get('terminal_growth')}, "
            f"years={r.inputs.get('high_growth_years')}"
        )
        return r.implied_growth_rate, citation, warnings

    raise ValueError(f"unsupported field '{field}'")


def _diff_pct(claim: float, source: float) -> float | None:
    if source == 0:
        return None
    return round(abs(claim - source) / abs(source) * 100.0, 3)


def verify_number(
    claim: float,
    field: str,
    symbol: str,
    client: FinnhubClient | None = None,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> VerificationResult:
    normalized = FIELD_ALIASES.get(field.lower(), field.lower())
    if normalized not in _SUPPORTED:
        raise ValueError(
            f"unsupported field '{field}'. supported: {sorted(_SUPPORTED)}"
        )

    owned = False
    if client is None:
        client = FinnhubClient()
        owned = True

    try:
        source_value, citation, warnings = _fetch_source(normalized, symbol, client)
    finally:
        if owned:
            client.close()

    if source_value is None:
        return VerificationResult(
            field=normalized,
            claim=claim,
            source_value=None,
            matches=None,
            diff_pct=None,
            tolerance_pct=tolerance_pct,
            source_citation=citation,
            warnings=warnings + ["source value unavailable — cannot verify"],
        )

    if source_value == 0:
        matches = claim == 0
        return VerificationResult(
            field=normalized,
            claim=claim,
            source_value=0.0,
            matches=matches,
            diff_pct=None,
            tolerance_pct=tolerance_pct,
            source_citation=citation,
            warnings=warnings + ["source value is zero; compared by absolute equality"],
        )

    diff = _diff_pct(claim, source_value)
    return VerificationResult(
        field=normalized,
        claim=claim,
        source_value=source_value,
        matches=diff is not None and diff <= tolerance_pct,
        diff_pct=diff,
        tolerance_pct=tolerance_pct,
        source_citation=citation,
        warnings=warnings,
    )


def list_supported_fields() -> list[str]:
    return sorted(_SUPPORTED)


# ---------------------------------------------------------------------------
# Read-only value fetching (used by pre-gather)
# ---------------------------------------------------------------------------


class FetchResult(BaseModel):
    """A value-only fetch result, without MATCH/MISMATCH verification semantics.

    Used by pre-gather when we just need the authoritative source value for
    a field, not a comparison. Keeps the verify_number tool's MATCH/MISMATCH
    verdicts out of reports that aren't actually verifying claims.
    """

    field: str
    value: float | None
    source_citation: str
    warnings: list[str] = Field(default_factory=list)


def fetch_source_value(
    field: str,
    symbol: str,
    client: FinnhubClient | None = None,
) -> FetchResult:
    """Return the authoritative value for a field with source attribution.

    Equivalent to "the 'source value' column of verify_number" but without
    the claim/diff/match apparatus. Prefer this over `verify_number(claim=0.0)`
    when the caller just wants the number.
    """
    normalized = FIELD_ALIASES.get(field.lower(), field.lower())
    if normalized not in _SUPPORTED:
        raise ValueError(
            f"unsupported field '{field}'. supported: {sorted(_SUPPORTED)}"
        )

    owned = False
    if client is None:
        client = FinnhubClient()
        owned = True

    try:
        value, citation, warnings = _fetch_source(normalized, symbol, client)
    finally:
        if owned:
            client.close()

    return FetchResult(
        field=normalized,
        value=value,
        source_citation=citation,
        warnings=warnings,
    )
