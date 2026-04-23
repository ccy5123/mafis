"""Tests for the peer sanity filter and the '## Peer Override' parser.

The valuation.py peer sanity filter drops peers whose multiples are so
extreme that any comparison is misleading (see the GEV post-mortem where
Finnhub returned Bloom Energy at EV/EBITDA=2423.95). The override parser
lets a value chain brief inject hand-picked comparables that Finnhub's
auto-peer algorithm missed.
"""

from __future__ import annotations

from wise_investor.agents.runner import parse_peer_override
from wise_investor.tools.valuation import (
    SANITY_MAX_EV_EBITDA,
    SANITY_MAX_PER,
    PeerMultipleRow,
    _is_sane_peer,
)


# ---------------------------------------------------------------------------
# Sanity filter
# ---------------------------------------------------------------------------


def test_sane_peer_typical_megacap() -> None:
    row = PeerMultipleRow(symbol="MSFT", per=35.0, ev_ebitda=25.0)
    keep, reason = _is_sane_peer(row)
    assert keep is True
    assert reason == ""


def test_excluded_when_both_multiples_missing() -> None:
    row = PeerMultipleRow(symbol="XYZ", per=None, ev_ebitda=None)
    keep, reason = _is_sane_peer(row)
    assert keep is False
    assert "unavailable" in reason


def test_excluded_when_per_exceeds_threshold() -> None:
    # Forgent-Power-like case from the GEV run.
    row = PeerMultipleRow(symbol="FPS", per=712.71, ev_ebitda=50.0)
    keep, reason = _is_sane_peer(row)
    assert keep is False
    assert "PER" in reason
    # Threshold is formatted as an integer ("300x") — check that magnitude appears.
    assert f"{int(SANITY_MAX_PER)}x" in reason


def test_excluded_when_ev_ebitda_exceeds_threshold() -> None:
    # Literal Bloom Energy case.
    row = PeerMultipleRow(symbol="BE", per=None, ev_ebitda=2423.95)
    keep, reason = _is_sane_peer(row)
    assert keep is False
    assert "EV/EBITDA" in reason


def test_excluded_when_per_negative() -> None:
    row = PeerMultipleRow(symbol="LOSS", per=-15.0, ev_ebitda=10.0)
    keep, reason = _is_sane_peer(row)
    assert keep is False
    assert "negative" in reason


def test_excluded_when_ev_ebitda_negative() -> None:
    row = PeerMultipleRow(symbol="LOSS", per=None, ev_ebitda=-8.0)
    keep, reason = _is_sane_peer(row)
    assert keep is False
    assert "negative" in reason


def test_kept_with_one_multiple_present() -> None:
    # Many real peers have only PER (loss on EBITDA but profitable) or only
    # EV/EBITDA. Keep them — they still contribute to the comparison.
    row = PeerMultipleRow(symbol="A", per=30.0, ev_ebitda=None)
    keep, _ = _is_sane_peer(row)
    assert keep is True


def test_boundary_exactly_at_threshold_is_kept() -> None:
    row_per = PeerMultipleRow(symbol="X", per=SANITY_MAX_PER, ev_ebitda=10.0)
    row_ev = PeerMultipleRow(symbol="Y", per=20.0, ev_ebitda=SANITY_MAX_EV_EBITDA)
    assert _is_sane_peer(row_per)[0] is True
    assert _is_sane_peer(row_ev)[0] is True


# ---------------------------------------------------------------------------
# Peer override parser
# ---------------------------------------------------------------------------


def test_parse_override_extracts_tickers() -> None:
    text = """# MOCK Value Chain

## Peer Override

Finnhub peers are incomplete. Add real comparables:

- SMNEY — Siemens Energy ADR
- ETN — Eaton Corporation
- ABBNY — ABB ADR

## Upstream — Suppliers

- Some supplier
"""
    tickers = parse_peer_override(text)
    assert tickers == ["SMNEY", "ETN", "ABBNY"]


def test_parse_override_handles_asterisk_bullets() -> None:
    text = """## Peer Override
* AAA — first
* BBB — second
## Next
"""
    assert parse_peer_override(text) == ["AAA", "BBB"]


def test_parse_override_stops_at_next_heading() -> None:
    text = """## Peer Override
- GOOD — valid
## Unrelated Section
- NOISE — should be ignored
"""
    assert parse_peer_override(text) == ["GOOD"]


def test_parse_override_missing_section_returns_empty() -> None:
    text = """## Some Other Heading
- NOT_A_PEER — prose line
"""
    assert parse_peer_override(text) == []


def test_parse_override_empty_section_returns_empty() -> None:
    text = """## Peer Override

No overrides needed; Finnhub auto-peers are adequate.

- (none)

## Next
"""
    # "(none)" does not match ticker pattern; no tickers extracted.
    assert parse_peer_override(text) == []


def test_parse_override_ignores_non_ticker_words() -> None:
    text = """## Peer Override

- NONE — not a ticker
- AAPL — real ticker
"""
    assert parse_peer_override(text) == ["AAPL"]


def test_parse_override_deduplicates() -> None:
    text = """## Peer Override

- DUP — first mention
- DUP — duplicate
- NEW — different
"""
    assert parse_peer_override(text) == ["DUP", "NEW"]


def test_parse_override_heading_case_insensitive() -> None:
    text = """## peer override
- CASE — should still work
"""
    assert parse_peer_override(text) == ["CASE"]


def test_parse_override_tolerates_symbols_with_dot() -> None:
    # ADR / foreign listing notation like "BRK.B" or "005930.KS".
    text = """## Peer Override
- BRK.B — Berkshire class B
"""
    assert parse_peer_override(text) == ["BRK.B"]
