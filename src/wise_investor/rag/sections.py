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

# Sections we slice up to (heading of the next part). Ordering matters —
# we try each in order and pick the earliest match after the current
# section's start.
STOP_MARKERS: list[str] = [
    r"Item\s*1A[.\s]+Risk\s*Factors",
    r"Item\s*1B[.\s]+",
    r"Item\s*2[.\s]+Properties",
    r"Item\s*3[.\s]+Legal",
    r"Item\s*4[.\s]+Mine\s*Safety",
    r"Item\s*5[.\s]+Market\s*for",
    r"Item\s*6[.\s]+",
    r"Item\s*7[.\s]+Management",
    r"Item\s*7A[.\s]+Quantitative",
    r"Item\s*8[.\s]+Financial",
    r"Item\s*9[.\s]+",
    r"Item\s*10[.\s]+",
    r"Item\s*11[.\s]+",
    r"Item\s*12[.\s]+",
    r"Item\s*13[.\s]+",
    r"Item\s*14[.\s]+",
    r"Item\s*15[.\s]+",
    r"Item\s*16[.\s]+",
    r"Signatures?\s*$",
]


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


def _slice_section(text: str, key: str) -> str | None:
    """Find the section whose heading matches SECTION_MARKERS[key] and
    return its body up to the next stop marker.
    """
    patterns = SECTION_MARKERS[key]
    m = _find_first(patterns, text)
    if m is None:
        return None

    # The text of many 10-K filings includes the heading both in the table
    # of contents AND in the body. The TOC copy is usually first and very
    # short (< 200 chars to the next item). Skip it and take the second
    # occurrence when present.
    first_start = m[0]
    first_body_candidate = text[m[1] : m[1] + 2000]
    # If the "body" right after first match contains another occurrence of
    # the same pattern within the first 1500 chars, it's likely TOC.
    second_m = _find_first(patterns, text, start=m[1])
    if second_m is not None and (second_m[0] - m[1]) < 1500:
        start = second_m[1]
    else:
        start = m[1]

    # Find the nearest stop marker after `start`.
    stop_idx: int | None = None
    for stop_pat in STOP_MARKERS:
        match = re.search(stop_pat, text[start:], re.IGNORECASE)
        if match:
            candidate = start + match.start()
            # The stop must not be identical to our own heading regex.
            # Simple guard: require at least 200 chars of body before stop.
            if candidate - start < 200:
                continue
            if stop_idx is None or candidate < stop_idx:
                stop_idx = candidate

    body = text[start:stop_idx] if stop_idx else text[start:]
    body = body.strip()
    if len(body) < 100:
        # Section exists but is too short to be useful — likely a TOC-only hit.
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
