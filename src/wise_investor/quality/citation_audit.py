"""Citation grounding audit — verify that `[Source: edgar.*]` citations
are actually supported by the indexed 10-K passages.

Problem this solves: observed in NVDA_20260423_1401.crew.md, the Valuer
wrote "semiconductor company growth rates, which have averaged around
15-20% [Source: edgar.mdna_highlights]". That number is NOT in NVDA's
10-K MD&A — verified via scripts/search_10k.py with distance > 1.4.
The LLM attached a plausible-looking edgar citation to a hallucinated
number, which defeats the entire RAG provenance system.

This module runs AFTER the crew produces a combined report. It:
  1. Extracts every sentence containing a `[Source: edgar.*]` citation.
  2. Extracts the quantitative claims in each sentence (numbers with
     units — percentages, dollar amounts, multiples).
  3. Queries ChromaDB for the relevant passages (symbol + section
     from the citation).
  4. Checks whether each numeric claim appears verbatim (or as a close
     textual match) inside the top-k retrieved passages.
  5. Flags ungrounded claims — numbers that appear in the report with
     an edgar citation but not in any indexed passage.

The audit is ADDITIVE: it returns a list of flagged sentences plus an
optional markdown block ready to append to the combined report. The
original text is not modified.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable


logger = logging.getLogger(__name__)


# `[Source: edgar.<label>]` — captures the query label used in
# rag.integration.DEFAULT_QUERIES. Labels (e.g. "mdna_highlights") are
# NOT the same as the ChromaDB section keys ("mdna"); we map them below.
_EDGAR_CITATION_RE = re.compile(
    r"\[Source:\s*edgar\.(?P<label>[a-z_]+)\s*(?:,[^]]+)?\]",
    re.IGNORECASE,
)


# edgar.<label> → underlying 10-K section key used by rag.index.
# `None` means "search all sections" (labels like moat_signals don't
# map cleanly to a single 10-K section because moat evidence is spread
# across Business and MD&A).
_EDGAR_LABEL_TO_SECTION: dict[str, str | None] = {
    "business_segments": "business",
    "moat_signals": None,
    "risk_factors": "risk_factors",
    "mdna_highlights": "mdna",
}


def _label_to_section(label: str) -> str | None:
    return _EDGAR_LABEL_TO_SECTION.get(label.lower(), None)

# Numeric-claim patterns — what counts as a "specific quantitative claim".
# Order matters for deduplication: more specific patterns first.
_NUMERIC_CLAIM_PATTERNS: list[re.Pattern[str]] = [
    # Dollar amounts: $1.2B, $1,200M, $500 million
    re.compile(
        r"\$\s*\d{1,3}(?:[,\d]*)?(?:\.\d+)?\s*(?:billion|million|thousand|B\b|M\b|K\b)?",
        re.IGNORECASE,
    ),
    # Percent ranges: 15-20%, 3 to 5%
    re.compile(r"\d+(?:\.\d+)?\s*(?:-|\s*to\s*)\s*\d+(?:\.\d+)?\s*%"),
    # Single percentages: 20.4%, 40%
    re.compile(r"\d+(?:\.\d+)?\s*%"),
    # Multiples: 40.56x, 34.87 EV/EBITDA — these often come from Python
    # tools and shouldn't be checked against edgar. Leave to caller to
    # strip if desired. For now include.
    re.compile(r"\d+(?:\.\d+)?\s*x\b", re.IGNORECASE),
]


@dataclass
class UngroundedClaim:
    """One numeric claim that was cited via edgar.* but not found in the
    retrieved passages.
    """

    sentence: str
    section: str
    claim_number: str
    nearest_distance: float  # top-1 Chroma distance for the sentence query
    reason: str


@dataclass
class CitationAuditResult:
    symbol: str
    citations_checked: int = 0
    ungrounded: list[UngroundedClaim] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skeptic_missing_edgar_risk: bool = False

    @property
    def violation(self) -> bool:
        return len(self.ungrounded) > 0 or self.skeptic_missing_edgar_risk


# A typical edgar passage doesn't contain year-on-year percent values
# (the 10-K rarely quantifies industry averages). When the report mixes
# numbers from Python tools (fetch.*) with edgar citations in the same
# sentence, we should not penalise numbers that a Python tool already
# supports. This set captures common tool-attribution tokens that
# mean "this number is from Python, not edgar".
_PYTHON_TOOL_CITATION_RE = re.compile(
    r"\[Source:\s*(?:fetch\.[a-z_]+|calculate_[a-z_]+|reverse_dcf|"
    r"cross_validate_quote|get_peer_multiples|fred\.[A-Z0-9_]+)\s*[,\]]",
    re.IGNORECASE,
)


def _extract_sentences_with_edgar_citations(text: str) -> list[tuple[str, str]]:
    """Return (sentence, edgar_label) for every sentence referencing
    `[Source: edgar.<label>]`. A "sentence" is approximated as the
    text between preceding and trailing newline / period boundaries —
    good enough for the report shapes we produce.
    """
    out: list[tuple[str, str]] = []
    # Iterate over citation matches, then slice back to the nearest
    # sentence boundary.
    for m in _EDGAR_CITATION_RE.finditer(text):
        label = m.group("label").lower()
        cite_start = m.start()
        cite_end = m.end()

        # Walk backward to find the sentence start: previous "\n\n", or
        # a line start, or a period that ends a different sentence.
        before = text[:cite_start]
        line_start = before.rfind("\n")
        prev_period = max(
            before.rfind(". "), before.rfind(".\n"), before.rfind("! "),
            before.rfind("? "), before.rfind(":\n")
        )
        start = max(line_start, prev_period)
        if start < 0:
            start = 0

        # Walk forward for sentence end. Prefer the first newline after
        # the citation, else a period.
        after = text[cite_end:]
        forward_dot = after.find(". ")
        forward_nl = after.find("\n")
        candidates = [x for x in (forward_dot, forward_nl) if x >= 0]
        end_offset = min(candidates) if candidates else len(after)
        end = cite_end + end_offset

        sentence = text[start:end].strip(" \t\n-*")
        if sentence:
            out.append((sentence, label))
    return out


def _extract_numeric_claims(sentence: str) -> list[str]:
    """Return every numeric claim in `sentence`, deduplicated, preserving
    order. Numbers tagged with a Python-tool citation on the same
    sentence are NOT excluded here — the caller can choose to skip
    those, since the question "is this number in edgar?" is not
    meaningful when the number is already sourced from Python tools.
    """
    claims: list[str] = []
    seen: set[str] = set()
    for pat in _NUMERIC_CLAIM_PATTERNS:
        for m in pat.finditer(sentence):
            val = m.group(0).strip()
            key = re.sub(r"\s+", "", val.lower())
            if key in seen:
                continue
            seen.add(key)
            claims.append(val)
    return claims


def _normalize_for_match(s: str) -> str:
    """Collapse whitespace + lowercase for substring matching."""
    return re.sub(r"\s+", " ", s.strip().lower())


def _claim_variants(claim: str) -> list[str]:
    """Return match variants for a numeric claim — handles common
    formatting differences (percent with/without space, dollar sign
    presence, ranges split into endpoints).
    """
    out = {claim, claim.replace(" ", ""), claim.lower()}
    # For percent ranges "15-20%", also try each endpoint alone.
    range_m = re.match(
        r"(\d+(?:\.\d+)?)\s*(?:-|\s*to\s*)\s*(\d+(?:\.\d+)?)\s*%", claim
    )
    if range_m:
        out.add(f"{range_m.group(1)}%")
        out.add(f"{range_m.group(2)}%")
        out.add(f"{range_m.group(1)}.{range_m.group(2)}")
    # Dollar amounts like "$1.2B" should also match "1.2 billion".
    dollar_m = re.match(
        r"\$\s*(\d+(?:\.\d+)?)\s*B", claim, re.IGNORECASE
    )
    if dollar_m:
        out.add(f"{dollar_m.group(1)} billion")
        out.add(f"{dollar_m.group(1)}B")
    return [v for v in out if v]


def audit_edgar_citations(
    combined_text: str,
    symbol: str,
    search_fn: Callable[..., list] | None = None,
    k: int = 5,
) -> CitationAuditResult:
    """Scan `combined_text` for `[Source: edgar.*]` citations and
    verify their numeric claims against indexed passages.

    `search_fn` defaults to `wise_investor.rag.index.search`; tests can
    inject a stub to return fixture passages without touching Chroma.
    """
    if search_fn is None:
        from wise_investor.rag.index import search as default_search
        search_fn = default_search

    result = CitationAuditResult(symbol=symbol.upper())
    sentences = _extract_sentences_with_edgar_citations(combined_text)
    result.citations_checked = len(sentences)

    # Skeptic compliance check runs regardless of whether any edgar.*
    # citations exist elsewhere — absence of an edgar.risk_factors
    # citation in the Skeptic section is the violation.
    skeptic_section = _extract_skeptic_section(combined_text)
    if skeptic_section is not None:
        cites_risk = re.search(
            r"\[Source:\s*(?:edgar\.risk_factors|10-K\s+risk_factors)",
            skeptic_section,
            re.IGNORECASE,
        )
        if not cites_risk:
            result.skeptic_missing_edgar_risk = True
            result.notes.append(
                "Skeptic section contains no [Source: 10-K risk_factors, ...] "
                "citation — the Phase 3D mandate was not honored."
            )

    if not sentences:
        result.notes.append("No edgar.* citations found in report.")
        return result

    for sentence, label in sentences:
        claims = _extract_numeric_claims(sentence)
        if not claims:
            continue

        # If the sentence already attributes the numbers to a Python
        # tool via another citation, skip — those claims aren't edgar-
        # sourced even if an edgar citation also appears.
        if _PYTHON_TOOL_CITATION_RE.search(sentence):
            continue

        chroma_section = _label_to_section(label)

        # Retrieve passages from Chroma, filtering by section when the
        # label maps cleanly. Fall back to an unfiltered search when
        # the first attempt returns nothing — some claims may be
        # elsewhere in the filing even when the LLM picked the wrong
        # edgar.<label> for the citation.
        try:
            hits = search_fn(
                query=sentence,
                symbol=symbol,
                section=chroma_section,
                k=k,
            )
            if not hits and chroma_section is not None:
                hits = search_fn(
                    query=sentence, symbol=symbol, section=None, k=k
                )
        except Exception as e:
            logger.warning("citation audit search failed: %s", e)
            continue

        if not hits:
            for claim in claims:
                result.ungrounded.append(
                    UngroundedClaim(
                        sentence=sentence,
                        section=label,
                        claim_number=claim,
                        nearest_distance=float("inf"),
                        reason="no passages retrieved for this query",
                    )
                )
            continue

        combined_passage_text = _normalize_for_match(
            " ".join(getattr(h, "text", "") for h in hits)
        )
        best_distance = min(getattr(h, "distance", 0.0) for h in hits)

        for claim in claims:
            variants = _claim_variants(claim)
            grounded = any(
                _normalize_for_match(v) in combined_passage_text for v in variants
            )
            if not grounded:
                result.ungrounded.append(
                    UngroundedClaim(
                        sentence=sentence,
                        section=label,
                        claim_number=claim,
                        nearest_distance=best_distance,
                        reason=(
                            "claim number not found in top-k retrieved "
                            "passages for the cited section"
                        ),
                    )
                )

    if result.ungrounded:
        result.notes.append(
            f"{len(result.ungrounded)} ungrounded numeric claim(s) found "
            f"among {result.citations_checked} edgar.* citation(s)."
        )
    else:
        result.notes.append(
            f"All {result.citations_checked} edgar.* citation(s) verified "
            f"against indexed passages."
        )
    return result


def _extract_skeptic_section(text: str) -> str | None:
    """Return the text of the `# Part N · Skeptic` section, or None if
    not detected. Matches any Part number so this works for the
    5-agent layout (Part 4) and the 6-agent debate layout (still Part 4
    but with Defender as Part 5 and Steward as Part 6).
    """
    m = re.search(
        r"^#\s*Part\s*\d+\s*·\s*Skeptic\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if m is None:
        return None
    start = m.end()
    next_part = re.search(
        r"^#\s*Part\s*\d+\s*·", text[start:], re.IGNORECASE | re.MULTILINE
    )
    end = start + (next_part.start() if next_part else len(text) - start)
    return text[start:end]


def render_citation_audit_section(result: CitationAuditResult) -> str:
    """Render the audit as a markdown block ready to append to the
    combined report. Returns an empty string when there are no
    violations (so the caller can unconditionally concatenate).
    """
    if not result.violation:
        return ""

    lines = [
        "",
        "---",
        "",
        "## System Audit — Citation Grounding",
        "",
    ]

    if result.ungrounded:
        lines.extend(
            [
                f"Scanned {result.citations_checked} `[Source: edgar.*]` "
                f"citation(s) in the report for {result.symbol}. "
                f"**{len(result.ungrounded)}** numeric claim(s) could not be "
                "grounded in the indexed 10-K passages:",
                "",
            ]
        )
        for u in result.ungrounded:
            lines.append(
                f"- **Claim `{u.claim_number}`** in section `{u.section}` "
                f"(nearest passage distance {u.nearest_distance:.3f}): "
                f"{u.reason}."
            )
            lines.append(f"  - Sentence: _{u.sentence.strip()}_")
        lines.append("")
        lines.append(
            "Readers should treat ungrounded edgar-cited numbers as "
            "unverified — the LLM attached the citation but the specific "
            "value does not appear in the retrieved passages."
        )

    if result.skeptic_missing_edgar_risk:
        if lines[-1] != "":
            lines.append("")
        lines.append(
            f"**Skeptic mandate violation:** the Skeptic section for "
            f"{result.symbol} did not cite any `[Source: 10-K "
            f"risk_factors, ...]` passage. Phase 3D's template requires at "
            f"least ONE of the 5 rebuttals to ground in the filing's Risk "
            f"Factors. Readers should regard the Skeptic's attacks as "
            f"grounded only in the curated value chain brief, not in "
            f"EDGAR-disclosed risks — a weaker form of red-team."
        )

    return "\n".join(lines) + "\n"


__all__ = [
    "CitationAuditResult",
    "UngroundedClaim",
    "audit_edgar_citations",
    "render_citation_audit_section",
]
