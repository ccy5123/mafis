"""Tests for finnhub data-layer enhancements (#4 alias, #3-deep debt)."""

from __future__ import annotations

import pytest

from wise_investor.data.finnhub import (
    FinancialLineItem,
    FinancialReport,
    FinancialsEntry,
    _financials_symbol,
    extract_field,
    total_debt,
)


def test_goog_aliases_to_googl() -> None:
    """Calibration finding (#4, 2026-04): post-2015 Alphabet reorg
    moved 10-K filings to GOOGL. Querying GOOG returns 4 stale
    pre-2015 entries; GOOGL returns the current 11 entries."""
    assert _financials_symbol("GOOG") == "GOOGL"
    assert _financials_symbol("goog") == "GOOGL"  # case-insensitive


def test_default_passthrough() -> None:
    """Most tickers map to themselves — the alias is the exception."""
    assert _financials_symbol("NVDA") == "NVDA"
    assert _financials_symbol("MSFT") == "MSFT"
    assert _financials_symbol("BRK-B") == "BRK-B"


def test_googl_already_correct_passthrough() -> None:
    """GOOGL itself shouldn't be mangled — the alias only fires on
    GOOG (the Class C symbol)."""
    assert _financials_symbol("GOOGL") == "GOOGL"


def test_lowercase_normalized_to_upper() -> None:
    assert _financials_symbol("nvda") == "NVDA"


# ---------------------------------------------------------------------------
# Debt extraction — capital-lease-inclusive variants (#3-deep)
# ---------------------------------------------------------------------------


def _entry_with_bs(items: list[FinancialLineItem]) -> FinancialsEntry:
    """Shorthand: build a stub entry whose balance-sheet has the given items."""
    return FinancialsEntry(
        symbol="TEST", year=2017, form="10-K",
        report=FinancialReport(bs=items),
    )


def test_long_term_debt_picks_capital_lease_variant() -> None:
    """Calibration finding (#3-deep, 2026-04): retailers like HD report
    long-term debt under `LongTermDebtAndCapitalLeaseObligations`
    instead of `LongTermDebtNoncurrent`. The candidate list now
    accepts the AndCapitalLease variant first.

    HD FY2017 was the canonical case: $24.27B in
    LongTermDebtAndCapitalLeaseObligations was missed entirely
    before this fix, leaving total_debt at $1.56B (CommercialPaper
    only) and IC computing as -$0.58B.
    """
    entry = _entry_with_bs([
        FinancialLineItem(
            concept="us-gaap_LongTermDebtAndCapitalLeaseObligations",
            value=24_267_000_000,
        ),
    ])
    assert extract_field(entry, "long_term_debt") == 24_267_000_000


def test_long_term_debt_falls_through_to_plain_noncurrent() -> None:
    """Filers without capital leases use bare LongTermDebtNoncurrent;
    the fallback chain still picks them up."""
    entry = _entry_with_bs([
        FinancialLineItem(
            concept="us-gaap_LongTermDebtNoncurrent",
            value=50_000_000_000,
        ),
    ])
    assert extract_field(entry, "long_term_debt") == 50_000_000_000


def test_long_term_debt_prefers_capital_lease_when_both_present() -> None:
    """When both tags coexist the capital-lease variant wins (it's
    the more inclusive figure). The candidate list ordering is the
    documented contract — flipping it would silently change the
    debt figure for filers that report both."""
    entry = _entry_with_bs([
        FinancialLineItem(
            concept="us-gaap_LongTermDebtAndCapitalLeaseObligations",
            value=24_000_000_000,
        ),
        FinancialLineItem(
            concept="us-gaap_LongTermDebtNoncurrent",
            value=20_000_000_000,
        ),
    ])
    # AndCapitalLease (24B) wins over plain Noncurrent (20B)
    assert extract_field(entry, "long_term_debt") == 24_000_000_000


def test_short_term_debt_picks_capital_lease_current_variant() -> None:
    """`LongTermDebtAndCapitalLeaseObligationsCurrent` (the 1-year-
    maturity portion of long-term debt-with-leases) added as a
    short-term debt candidate."""
    entry = _entry_with_bs([
        FinancialLineItem(
            concept="us-gaap_LongTermDebtAndCapitalLeaseObligationsCurrent",
            value=1_202_000_000,
        ),
    ])
    assert extract_field(entry, "short_term_debt") == 1_202_000_000


def test_total_debt_sums_long_and_short_post_fix() -> None:
    """End-to-end: HD FY2017 had CommercialPaper $1.56B (short) +
    LongTermDebtAndCapitalLeaseObligations $24.27B (long). Pre-fix
    total_debt returned only $1.56B. Post-fix it returns the sum.

    Note: CommercialPaper still wins in the short_term candidate
    chain because LongTermDebtAndCapitalLeaseObligationsCurrent
    isn't present in this synthetic entry; HD's real entry has both
    and the AndCapitalLease..Current variant wins by being first.
    """
    entry = _entry_with_bs([
        FinancialLineItem(
            concept="us-gaap_CommercialPaper", value=1_559_000_000,
        ),
        FinancialLineItem(
            concept="us-gaap_LongTermDebtAndCapitalLeaseObligations",
            value=24_267_000_000,
        ),
    ])
    assert total_debt(entry) == 1_559_000_000 + 24_267_000_000


# ---------------------------------------------------------------------------
# P3-3 (2026-04): tolerant numeric coercion for noisy Finnhub responses
# ---------------------------------------------------------------------------


def test_value_validator_accepts_plain_number() -> None:
    """Numeric values pass through unchanged (regression for the legacy
    sentinel-only validator)."""
    li = FinancialLineItem(concept="us-gaap_Revenues", value=1_234_567.0)
    assert li.value == pytest.approx(1_234_567.0)


def test_value_validator_coerces_legacy_sentinels_to_none() -> None:
    """The original validator behavior — sentinel strings → None."""
    for sentinel in ("", "-", "—", "N/A", "NA", "null", "none"):
        li = FinancialLineItem(concept="us-gaap_X", value=sentinel)
        assert li.value is None, f"sentinel {sentinel!r} should become None"


def test_value_validator_parses_comma_formatted_string() -> None:
    """Finnhub occasionally returns '1,234,000' — parse, don't reject."""
    li = FinancialLineItem(concept="us-gaap_X", value="1,234,000")
    assert li.value == pytest.approx(1_234_000.0)


def test_value_validator_treats_html_noise_as_missing() -> None:
    """The CDNS/F failure mode — HTML body fragments leaking into a
    numeric field. Must not raise; returns None so the rest of the
    response stays parseable."""
    html_noise = '<!--DOCTYPE html PUBLIC ...reak End -->\n   </div>'
    li = FinancialLineItem(concept="us-gaap_X", value=html_noise)
    assert li.value is None


def test_value_validator_treats_arbitrary_prose_as_missing() -> None:
    """Defense in depth: any non-numeric string should land as None."""
    li = FinancialLineItem(concept="us-gaap_X", value="see footnote 7")
    assert li.value is None
