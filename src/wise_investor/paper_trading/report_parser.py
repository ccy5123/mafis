"""Extract the verdict + conviction + audit status from a crew report.

Combines two existing parsers we already have:
  - `notify.summary.extract_verdict_summary` reads the LLM's
    reported Verdict / Conviction off the Steward section.
  - `agents.steward_audit.audit_steward_section` re-runs the
    discipline matrix on the saved markdown (picks up both the raw
    labels and any audit-corrected ceiling).

The paper-trading ledger needs both: we want to record (a) what the
LLM said, (b) what the audit corrected it to, and (c) whether the
audit downgraded the verdict. That last bit lets us later measure
"how much alpha did the audit discipline actually contribute?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from wise_investor.agents.steward_audit import audit_steward_section
from wise_investor.notify.summary import extract_verdict_summary


@dataclass
class CrewReportSummary:
    """What the paper-trading ledger records per report."""

    symbol: str
    verdict: str | None          # post-audit (what we treat as the "real" call)
    conviction: int | None       # post-audit
    original_verdict: str | None # LLM's raw Verdict heading
    original_conviction: int | None
    audit_downgraded: bool       # True if audit changed the verdict


def _extract_symbol_from_report(text: str) -> str | None:
    """Pick up the ticker from the report title, e.g. `# NVDA — Equity ...`."""
    m = re.search(r"^#\s+([A-Z][A-Z0-9.-]{0,9})\s+—", text, re.MULTILINE)
    if m:
        return m.group(1)
    return None


def parse_crew_report(text: str, symbol_hint: str | None = None) -> CrewReportSummary:
    """Return a CrewReportSummary for a full combined-report markdown.

    `symbol_hint` overrides the title-bar extraction when the caller
    already knows the ticker (e.g., reading back a file named
    NVDA_20260424_1015.crew.md).
    """
    symbol = (symbol_hint or _extract_symbol_from_report(text) or "").upper()

    # What the LLM wrote (before audit correction).
    raw = extract_verdict_summary(symbol, text)
    original_verdict = raw.verdict
    original_conviction = raw.conviction

    # What the discipline audit corrected (or left alone).
    audit = audit_steward_section(text)

    if audit.violation and audit.corrected_verdict is not None:
        final_verdict = audit.corrected_verdict
        final_conviction = audit.corrected_conviction
        audit_downgraded = True
    else:
        final_verdict = audit.verdict or original_verdict
        final_conviction = audit.conviction or original_conviction
        audit_downgraded = False

    return CrewReportSummary(
        symbol=symbol,
        verdict=final_verdict,
        conviction=final_conviction,
        original_verdict=original_verdict,
        original_conviction=original_conviction,
        audit_downgraded=audit_downgraded,
    )


__all__ = ["CrewReportSummary", "parse_crew_report"]
