"""Extract the analytical sections from a 10-K HTML document.

We care about four sections that drive most qualitative analysis:

  Item 1.  Business                  — what the company does
  Item 1A. Risk Factors              — what could go wrong (Skeptic fuel)
  Item 7.  MD&A                      — management's own narrative
  Item 7A. Quantitative market risk  — rate / FX / commodity exposure

Strategy: strip HTML to text via BeautifulSoup, then use case-insensitive
regex to find each section heading and slice to the next one. 10-K
templates vary across companies; the regex tolerates "ITEM 1", "Item 1.",
"ITEM 1.", and so on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup


SECTION_MARKERS: dict[str, list[str]] = {
    # key -> list of regex fragments matching that section's heading.
    "business": [r"Item\s*1[.\s]+Business"],
    "risk_factors": [r"Item\s*1A[.\s]+Risk\s*Factors"],
    "mdna": [
        r"Item\s*7[.\s]+Management'?s?\s*Discussion",
        r"Item\s*7[.\s]+Management",
    ],
    "quant_market_risk": [r"Item\s*7A[.\s]+Quantitative"],
}

# Per-section stop patterns. Each section key lists headings that come
# AFTER it in the canonical 10-K outline — this matters because MD&A
# (Item 7) routinely cross-references Item 1A Risk Factors, and a naive
# union-of-all-stops list would terminate the MD&A body at that cross-
# reference rather than at the real next section.
#
# Outline: Item 1, 1A, 1B, 1C, 2, 3, 4, Part II, Item 5, 6, 7, 7A, 8, ...
SECTION_STOPS: dict[str, list[str]] = {
    "business": [
        r"Item\s*1A[.\s]+Risk\s*Factors",
        r"Item\s*1B[.\s]+",
        r"Item\s*2[.\s]+Properties",
        r"Item\s*3[.\s]+Legal",
        r"Item\s*4[.\s]+Mine\s*Safety",
        r"Item\s*5[.\s]+Market\s*for",
    ],
    "risk_factors": [
        r"Item\s*1B[.\s]+",
        r"Item\s*1C[.\s]+",
        r"Item\s*2[.\s]+Properties",
        r"Item\s*3[.\s]+Legal",
        r"Item\s*4[.\s]+Mine\s*Safety",
        r"Item\s*5[.\s]+Market\s*for",
    ],
    "mdna": [
        r"Item\s*7A[.\s]+Quantitative",
        r"Item\s*8[.\s]+Financial",
        r"Item\s*9[.\s]+",
    ],
    "quant_market_risk": [
        r"Item\s*8[.\s]+Financial",
        r"Item\s*9[.\s]+",
        r"Item\s*10[.\s]+",
    ],
}

# Legacy alias kept for backward-compat in tests that imported this name.
# New code should use SECTION_STOPS[key] per section.
STOP_MARKERS: list[str] = sorted(
    {pat for patterns in SECTION_STOPS.values() for pat in patterns}
)


@dataclass
class ParsedSections:
    business: str | None
    risk_factors: str | None
    mdna: str | None
    quant_market_risk: str | None

    def as_dict(self) -> dict[str, str]:
        return {
            name: text
            for name, text in {
                "business": self.business,
                "risk_factors": self.risk_factors,
                "mdna": self.mdna,
                "quant_market_risk": self.quant_market_risk,
            }.items()
            if text
        }


def html_to_plain_text(html: str) -> str:
    """Strip HTML tags, collapse whitespace, leave only readable body."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script/style/tables' formatting noise.
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Normalize whitespace: collapse runs of blank lines, trim each line.
    lines = [line.strip() for line in text.splitlines()]
    # Drop very short lines that are likely page-number / TOC debris (<3 chars).
    cleaned = "\n".join(line for line in lines if line)
    # Squash 3+ consecutive newlines to a single blank line.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _find_first(
    patterns: list[str], text: str, start: int = 0
) -> tuple[int, int] | None:
    """Return (start, end) of the earliest regex match across `patterns`
    at or after `start`, or None if no pattern matches.
    """
    earliest: tuple[int, int] | None = None
    for pat in patterns:
        m = re.search(pat, text[start:], re.IGNORECASE)
        if not m:
            continue
        abs_start = start + m.start()
        abs_end = start + m.end()
        if earliest is None or abs_start < earliest[0]:
            earliest = (abs_start, abs_end)
    return earliest


def _find_all(patterns: list[str], text: str) -> list[tuple[int, int]]:
    """Return every (start, end) match across all `patterns`, sorted by
    position. Used to locate the TOC entry vs the body heading, where
    the body is always the last match.
    """
    hits: list[tuple[int, int]] = []
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            hits.append((m.start(), m.end()))
    hits.sort()
    # Deduplicate overlapping matches from different patterns at the same
    # position (e.g. two regex variants both match the MD&A heading).
    deduped: list[tuple[int, int]] = []
    for start, end in hits:
        if deduped and abs(start - deduped[-1][0]) < 50:
            continue
        deduped.append((start, end))
    return deduped


def _find_nearest_stop(
    text: str, start: int, stop_patterns: list[str] | None = None
) -> int | None:
    """Find the earliest stop-pattern position strictly after `start`.

    `stop_patterns` defaults to the union STOP_MARKERS for backward
    compatibility, but _slice_section passes the section-specific list
    from SECTION_STOPS so cross-references to earlier Items do not
    terminate the body prematurely.

    A 5-char self-match guard skips stops that overlap the heading we
    just consumed.
    """
    patterns = stop_patterns if stop_patterns is not None else STOP_MARKERS
    stop_idx: int | None = None
    for stop_pat in patterns:
        match = re.search(stop_pat, text[start:], re.IGNORECASE)
        if match is None:
            continue
        candidate = start + match.start()
        if candidate - start < 5:
            continue
        if stop_idx is None or candidate < stop_idx:
            stop_idx = candidate
    return stop_idx


def _slice_section(text: str, key: str) -> str | None:
    """Find the section whose heading matches SECTION_MARKERS[key] and
    return its body up to the next stop marker.

    10-K filings can contain the same "Item N." string many times:
      - Once in the Table of Contents (tight cluster, body ~20 chars).
      - Once as the actual body heading (body runs thousands of chars
        until the next Item heading).
      - Multiple times as cross-references inside other sections
        ("see Item 1A. Risk Factors for details", body ~50 chars).

    We pick the match whose candidate body — distance to the next stop
    marker — is LARGEST. That uniquely identifies the real body heading
    in every 10-K shape we've tested (NVDA 2026, AAPL 2024, etc.).
    """
    patterns = SECTION_MARKERS[key]
    all_matches = _find_all(patterns, text)
    if not all_matches:
        return None

    stop_patterns = SECTION_STOPS.get(key)
    best_body_size = 0
    best_end = 0
    best_stop: int | None = None
    for m_start, m_end in all_matches:
        stop = _find_nearest_stop(text, m_end, stop_patterns)
        body_size = (stop if stop is not None else len(text)) - m_end
        if body_size > best_body_size:
            best_body_size = body_size
            best_end = m_end
            best_stop = stop

    if best_body_size == 0:
        return None

    body = text[best_end:best_stop] if best_stop else text[best_end:]
    body = body.strip()
    if len(body) < 100:
        return None
    return body


def extract_sections(html: str) -> ParsedSections:
    """Parse the four target sections from 10-K HTML. Missing sections are None."""
    text = html_to_plain_text(html)
    return ParsedSections(
        business=_slice_section(text, "business"),
        risk_factors=_slice_section(text, "risk_factors"),
        mdna=_slice_section(text, "mdna"),
        quant_market_risk=_slice_section(text, "quant_market_risk"),
    )
