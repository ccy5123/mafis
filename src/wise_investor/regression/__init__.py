"""Phase 4 regression test harness — compare crew reports over time.

Use case: you tune a prompt, or upgrade from Qwen 7B to Qwen 14B, or
bump the audit thresholds. The question becomes "did my change
improve or regress the output against a known baseline?". This
package provides a deterministic compare:

  baseline.crew.md  vs  new.crew.md
  ───────────────────────────────────
  Verdict:        BUY C4  →  HOLD C2     (regressed or corrected?)
  citation_rate:  84.9%   →  92.1%       (improved)
  refusal_count:  8       →  12          (Skeptic tighter)
  audit violations: 1     →  0           (discipline gap closed)

Intentionally narrow scope: we compare the parsed artifacts
(verdict, quality metrics, audit outcomes), NOT raw markdown text
diffs — the narrative differs run-to-run by design, and a git-style
diff would drown real signal in paragraph re-orderings.
"""

from wise_investor.regression.compare import (
    MetricDelta,
    ReportDiff,
    compare_reports,
    extract_report_signals,
)

__all__ = [
    "MetricDelta",
    "ReportDiff",
    "compare_reports",
    "extract_report_signals",
]
