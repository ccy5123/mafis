"""Automated report-quality metrics (Phase 1D).

Each metric is a pure function: it takes a report's markdown text (and
optionally the facts cache used to generate it) and returns a MetricResult
with a numeric value, a pass/fail verdict against a threshold, and a
supporting `details` dict for debugging and regression tracking.

Design philosophy: metrics are cheap, deterministic, and composable.
Imperfect numerical readings are acceptable when they move in the right
direction over time — the point is regression detection, not absolute
truth.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Single-metric result with provenance for regression tracking."""

    name: str
    value: float
    unit: str  # "rate", "count", "ratio", "percent"
    passed: bool | None = None  # None if threshold not applicable
    threshold: float | None = None
    explanation: str
    details: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared regex building blocks
# ---------------------------------------------------------------------------


# Money with a magnitude suffix: $215.94B, 215.9380B (facts cache format),
# $7.47M, 1,127.56. The $ is optional. If $ is absent, a B/M/K suffix is
# required to distinguish from a bare decimal — otherwise "40.79" (a PER
# multiple) would be indistinguishable from "$40.79" (dollars).
_MONEY_WITH_UNIT_PATTERN = r"\$?\d[\d,]*(?:\.\d+)?[BMKbmk]"
# Money with $ and no magnitude suffix: "$1,127.56", "$40.00"
_DOLLAR_NO_UNIT_PATTERN = r"\$[\d,]+(?:\.\d+)?"
# Percentages: 20.27%, 54.72%, 5%
_PERCENT_PATTERN = r"\d+(?:\.\d+)?\s*%"
# Multiples: 35.68x, 27.30x
_MULTIPLE_PATTERN = r"\b\d+(?:\.\d+)?\s*x\b"
# Standalone decimals with a unit-like context: "PER 40.79", "EV/EBITDA 35.01"
_BARE_NUMBER_PATTERN = r"\b\d+\.\d+\b"

# Combined: all "numeric tokens" we consider citation-worthy.
# Order matters — more specific patterns first so "$215.94B" is not
# half-matched by the bare-decimal pattern.
_NUMERIC_TOKEN_PATTERN = re.compile(
    "|".join(
        [
            _MONEY_WITH_UNIT_PATTERN,
            _DOLLAR_NO_UNIT_PATTERN,
            _PERCENT_PATTERN,
            _MULTIPLE_PATTERN,
            _BARE_NUMBER_PATTERN,
        ]
    )
)

# A citation appears as "[Source: ...]" or "— [Source: ...]". We want to
# know whether a numeric token has one WITHIN ~120 characters after it on
# the same line or the immediately-following bullet-continuation line.
_CITATION_PATTERN = re.compile(r"\[Source:\s*[^\]]+\]")


# ---------------------------------------------------------------------------
# #5 Refusal phrase count — easiest
# ---------------------------------------------------------------------------


# Exact refusal phrases the Skeptic agent is prompted to emit when a claim
# cannot be quantified from sourced facts. Count these directly.
REFUSAL_PHRASES = [
    "Downside not quantifiable from current facts",
    "Unknown from current facts",
    "Not computable from current facts",
    "not quantifiable from current facts",  # case-fold minor variant
]


def refusal_count(report: str) -> MetricResult:
    """Count occurrences of the Skeptic's sanctioned refusal phrases.

    Higher = stronger epistemic humility. Zero means either no impossible-
    to-quantify claims arose (uncommon) OR the agent is inventing numbers
    instead of refusing. Benchmark established from Phase 1C NVDA run: the
    Skeptic's 5 rebuttals plus 2 Reverse-DCF stress-test questions yield
    ~7 refusals when epistemically honest. Threshold set conservatively
    at 3.
    """
    counts_by_phrase: dict[str, int] = {}
    total = 0
    text_lc = report.lower()
    for phrase in REFUSAL_PHRASES:
        c = text_lc.count(phrase.lower())
        counts_by_phrase[phrase] = c
        total += c

    # Deduplicate: the two case variants of "not quantifiable from current facts"
    # double-count. Subtract the minor variant, it's a subset of the first.
    total -= counts_by_phrase.get("not quantifiable from current facts", 0)

    threshold = 3
    return MetricResult(
        name="refusal_count",
        value=float(total),
        unit="count",
        threshold=float(threshold),
        passed=total >= threshold,
        explanation=(
            f"Skeptic emitted {total} sanctioned refusal phrase(s) "
            f"(threshold ≥ {threshold}). Higher = more epistemic humility."
        ),
        details={"counts_by_phrase": counts_by_phrase},
    )


# ---------------------------------------------------------------------------
# #1 Citation rate — numeric tokens with [Source: ...] nearby
# ---------------------------------------------------------------------------


def _numeric_tokens_in_line(line: str) -> list[str]:
    return _NUMERIC_TOKEN_PATTERN.findall(line)


def citation_rate(report: str) -> MetricResult:
    """Fraction of numeric tokens in the report that are followed by a
    `[Source: ...]` citation on the same line.

    Implementation: for each non-empty line, count numeric tokens and check
    whether the line contains a [Source: ...] citation. Lines without any
    number are ignored. Lines with a number but no citation count as
    uncited; lines with both count all their numbers as cited.

    This is a lower-bound estimator — it assumes one citation covers every
    number on its line, which matches the actual Phase 1B/1C output format
    ("$215.94B — [Source: fetch.revenue]"). Numbers split across multiple
    lines in a fenced code block (peer tables) are not penalised because
    they're explicitly labelled by the table header, not by per-row
    citations.
    """
    # Strip fenced code blocks — the peer-multiples table is whitelisted;
    # its numbers are labelled by column headers, not per-row citations.
    text_no_code = re.sub(r"```[\s\S]*?```", "", report)

    total_numbers = 0
    cited_numbers = 0
    uncited_samples: list[str] = []

    for line in text_no_code.splitlines():
        nums = _numeric_tokens_in_line(line)
        if not nums:
            continue
        total_numbers += len(nums)
        if _CITATION_PATTERN.search(line):
            cited_numbers += len(nums)
        else:
            if len(uncited_samples) < 5:
                uncited_samples.append(line.strip()[:120])

    rate = (cited_numbers / total_numbers) if total_numbers > 0 else 1.0
    threshold = 0.80  # 80% of numeric claims should have inline citations
    return MetricResult(
        name="citation_rate",
        value=round(rate, 4),
        unit="rate",
        threshold=threshold,
        passed=rate >= threshold,
        explanation=(
            f"{cited_numbers} of {total_numbers} numeric tokens "
            f"({rate * 100:.1f}%) are on a line with [Source: ...]. "
            f"Threshold ≥ {threshold * 100:.0f}%."
        ),
        details={
            "total_numbers": total_numbers,
            "cited_numbers": cited_numbers,
            "uncited_line_samples": uncited_samples,
        },
    )


# ---------------------------------------------------------------------------
# #6 Vulnerable-link grounding
# ---------------------------------------------------------------------------


_VULNERABLE_LINK_REF_PATTERN = re.compile(
    r"Vulnerable\s+link\s*#?\s*(\d+)", re.IGNORECASE
)


def _find_skeptic_section(report: str) -> str:
    """Return the Skeptic portion of the combined report, or the whole
    report if the Part 3 header is absent.
    """
    m = re.search(
        r"^\s*#+\s*Part\s*3\s*[·\-]\s*Skeptic", report, re.MULTILINE
    )
    if not m:
        return report
    return report[m.start():]


def vulnerable_link_grounding(report: str) -> MetricResult:
    """Count references to 'Vulnerable link #N' inside the Skeptic section.

    Phase 1C design (§7.4 strengthening) requires at least 3 of the 5
    Skeptic rebuttals to ground in a value-chain Vulnerable link. We measure
    this by counting unique link numbers referenced in Skeptic's output.
    """
    skeptic = _find_skeptic_section(report)
    matches = _VULNERABLE_LINK_REF_PATTERN.findall(skeptic)
    unique_links = sorted({int(m) for m in matches})
    total_refs = len(matches)

    threshold = 3
    return MetricResult(
        name="vulnerable_link_grounding",
        value=float(len(unique_links)),
        unit="count",
        threshold=float(threshold),
        passed=len(unique_links) >= threshold,
        explanation=(
            f"Skeptic referenced {len(unique_links)} distinct Vulnerable "
            f"link(s) with {total_refs} total mentions. Threshold ≥ "
            f"{threshold} distinct links."
        ),
        details={
            "distinct_link_numbers": unique_links,
            "total_references": total_refs,
        },
    )


# ---------------------------------------------------------------------------
# #4 Hard numbers vs scenario-hedging ratio
# ---------------------------------------------------------------------------


# Scenario / hedging words that indicate a conditional statement rather
# than a factual claim. Counted alongside numbers to assess how much of
# the report is "locked in" facts vs scenario narrative.
_SCENARIO_WORDS = [
    r"\bcould\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bwould\b",
    r"\bif\b",
    r"\bassume[sd]?\b",
    r"\bassumption\b",
    r"\bimply(?:ing)?\b",
    r"\bimplies\b",
    r"\bimplied\b",
    r"\bscenario\b",
    r"\bpotential(?:ly)?\b",
]
_SCENARIO_PATTERN = re.compile("|".join(_SCENARIO_WORDS), re.IGNORECASE)


def hard_vs_scenario(report: str) -> MetricResult:
    """Ratio of hard numeric tokens to scenario/hedging words.

    High ratio = fact-dense, sparse hedging. Low ratio = heavy scenario
    narrative relative to facts. For an equity research note we prefer
    >= 0.5 (at least one number per two hedging words). Code fences are
    excluded so the peer table doesn't inflate the numerator.
    """
    text_no_code = re.sub(r"```[\s\S]*?```", "", report)

    numbers = _NUMERIC_TOKEN_PATTERN.findall(text_no_code)
    scenario_words = _SCENARIO_PATTERN.findall(text_no_code)

    n_num = len(numbers)
    n_sc = len(scenario_words)
    if n_sc == 0:
        ratio = float("inf") if n_num > 0 else 0.0
    else:
        ratio = n_num / n_sc

    threshold = 0.5
    # Handle inf for display
    display_ratio = 99.0 if ratio == float("inf") else round(ratio, 3)
    return MetricResult(
        name="hard_vs_scenario",
        value=display_ratio,
        unit="ratio",
        threshold=threshold,
        passed=(ratio == float("inf")) or (ratio >= threshold),
        explanation=(
            f"Report has {n_num} numeric tokens and {n_sc} hedging/scenario "
            f"words. Ratio {display_ratio}. Threshold ≥ {threshold}."
        ),
        details={
            "numbers": n_num,
            "scenario_words": n_sc,
        },
    )


# ---------------------------------------------------------------------------
# #2 Invention audit — every number in report must be in facts cache
# ---------------------------------------------------------------------------


# Canonical representation of a numeric token: strip $, commas, spaces; keep
# the trailing unit suffix (B/M/K) and the % or x marker. Small precision
# tolerance: 0.5% so a number rounded to two decimals in the report still
# matches the raw source value.
_NUM_CANON_PATTERN = re.compile(
    r"(\-?)(?:\$)?([\d,]+(?:\.\d+)?)([BMKbmk%x])?"
)


def _normalise_number(token: str) -> float | None:
    """Convert '$215.94B' → 215940000000.0; '40.79x' → 40.79; '20.27%' → 0.2027.

    Returns None if the token can't be interpreted as a single number.
    """
    t = token.strip()
    m = _NUM_CANON_PATTERN.fullmatch(t)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    digits = m.group(2).replace(",", "")
    try:
        v = float(digits)
    except ValueError:
        return None
    unit = (m.group(3) or "").lower()
    if unit == "b":
        v *= 1e9
    elif unit == "m":
        v *= 1e6
    elif unit == "k":
        v *= 1e3
    elif unit == "%":
        v /= 100.0
    # 'x' (multiple) stays as-is
    return sign * v


def _extract_numbers_from_facts(facts: dict[str, str]) -> set[float]:
    """Pull every numeric value out of the pre-gathered facts cache."""
    values: set[float] = set()
    for txt in facts.values():
        for tok in _NUMERIC_TOKEN_PATTERN.findall(txt):
            v = _normalise_number(tok)
            if v is not None:
                values.add(round(v, 4))
    return values


def invention_audit(
    report: str,
    facts: dict[str, str],
    tolerance_pct: float = 0.5,
    value_chain_text: str | None = None,
) -> MetricResult:
    """Every number in the report (outside code fences) should trace to a
    number in the facts cache OR the value chain brief, within
    `tolerance_pct` percent.

    Reports any number that has no near-match as a suspected invention.
    Small amounts of drift are tolerated because narrative sometimes
    paraphrases (e.g. "~20%" summarising 20.27%). The value chain brief is
    included in the source pool so hand-curated facts (like "40% of data
    center revenue") are not mis-flagged.

    Header lines generated by the runner (starting with "_Models:") are
    stripped before scanning so model-version strings like "7b-16k" don't
    appear as suspected numbers.
    """
    # Strip runner header lines that carry model version strings.
    text_filtered = "\n".join(
        line for line in report.splitlines()
        if not line.strip().startswith("_Models:")
    )
    text_no_code = re.sub(r"```[\s\S]*?```", "", text_filtered)
    report_tokens = _NUMERIC_TOKEN_PATTERN.findall(text_no_code)

    # Pool numbers from both the facts cache and (optionally) the value chain
    # brief — the Analyst legitimately cites curated qualitative data.
    fact_values = _extract_numbers_from_facts(facts)
    if value_chain_text:
        for tok in _NUMERIC_TOKEN_PATTERN.findall(value_chain_text):
            v = _normalise_number(tok)
            if v is not None:
                fact_values.add(round(v, 4))
    fact_list = sorted(fact_values)

    suspected: list[tuple[str, float]] = []
    audited = 0
    for tok in report_tokens:
        v = _normalise_number(tok)
        if v is None:
            continue
        audited += 1
        v_abs = abs(v)
        # Check match against every fact value. For small values, use
        # absolute tolerance of 0.01; for large, percent-based.
        matched = False
        # Build list of candidate target values to try — the token's
        # normalized value PLUS its percent<->decimal equivalents. This
        # covers facts stored as "3.64" that the report emits as "3.64%"
        # (and the reverse).
        candidates = [v, v * 100.0, v / 100.0]
        for cand in candidates:
            for fv in fact_values:
                if fv == 0:
                    if abs(cand) < 0.01:
                        matched = True
                        break
                    continue
                if abs(cand - fv) / abs(fv) * 100.0 <= tolerance_pct:
                    matched = True
                    break
            if matched:
                break
        if not matched:
            # Second chance: small integers like "5", "10 years", "100" are
            # not data numbers and would fail an audit. Skip tokens whose
            # integer value is one of these structural markers.
            if v.is_integer() and v_abs in {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 30, 100, 365, 2024, 2025, 2026, 2027, 2028,
            }:
                continue
            suspected.append((tok, v))

    threshold = 3  # allow up to 3 drifted/uncatalogued numbers
    n_suspect = len(suspected)
    return MetricResult(
        name="invention_audit",
        value=float(n_suspect),
        unit="count",
        threshold=float(threshold),
        passed=n_suspect <= threshold,
        explanation=(
            f"{n_suspect} number(s) in the report did not match any value "
            f"in the facts cache within ±{tolerance_pct}%. Threshold ≤ "
            f"{threshold} drifted or uncatalogued numbers."
        ),
        details={
            "audited_tokens": audited,
            "suspected_inventions": [tok for tok, _ in suspected[:10]],
            "fact_pool_size": len(fact_list),
        },
    )


# ---------------------------------------------------------------------------
# #3 Skeptic coverage — Skeptic targets Bull claims
# ---------------------------------------------------------------------------


def skeptic_coverage(report: str) -> MetricResult:
    """Count how many of the Skeptic's rebuttals name a target claim that
    is attributable to the Analyst or Valuer.

    Uses the Phase 1C template marker 'Target claim (Analyst|Valuer)' or
    'Target claim (Analyst)' / 'Target claim (Valuer)'. A well-formed
    Phase 1C report should have exactly 5 such attributions.
    """
    skeptic = _find_skeptic_section(report)
    matches = re.findall(
        r"Target claim\s*\(\s*(Analyst|Valuer|Analyst\|Valuer|Valuer\|Analyst)\s*\)",
        skeptic,
        re.IGNORECASE,
    )
    count = len(matches)
    analyst_count = sum(1 for m in matches if "analyst" in m.lower())
    valuer_count = sum(1 for m in matches if "valuer" in m.lower())

    threshold = 5  # Phase 1C template mandates exactly 5 rebuttals
    return MetricResult(
        name="skeptic_coverage",
        value=float(count),
        unit="count",
        threshold=float(threshold),
        # Both 5 exactly and "more" pass, but the template says exactly 5;
        # under-production fails, over-production is technically off-spec
        # but we don't fail it.
        passed=count >= threshold,
        explanation=(
            f"Skeptic produced {count} rebuttal(s) with an explicit "
            f"'Target claim (...)' attribution (Analyst: {analyst_count}, "
            f"Valuer: {valuer_count}). Template mandates {threshold}."
        ),
        details={
            "total_targeted_rebuttals": count,
            "targets_analyst": analyst_count,
            "targets_valuer": valuer_count,
        },
    )


# ---------------------------------------------------------------------------
# Aggregate scoring
# ---------------------------------------------------------------------------


ALL_METRICS = [
    "refusal_count",
    "citation_rate",
    "vulnerable_link_grounding",
    "hard_vs_scenario",
    "invention_audit",
    "skeptic_coverage",
]


def score_report(
    report: str,
    facts: dict[str, str] | None = None,
    value_chain_text: str | None = None,
) -> list[MetricResult]:
    """Run every metric on a report. Metrics that need the facts cache are
    skipped with a None `passed` when it is not supplied.
    """
    results: list[MetricResult] = [
        refusal_count(report),
        citation_rate(report),
        vulnerable_link_grounding(report),
        hard_vs_scenario(report),
        skeptic_coverage(report),
    ]
    if facts is not None:
        results.insert(
            4,
            invention_audit(report, facts, value_chain_text=value_chain_text),
        )
    return results
