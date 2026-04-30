"""Calibration ledger — persist back-validation summaries across constitution versions.

Each calibration run produces a `BackValidationSummary` (constitution
version, calibration date, per-ticker records, aggregate stats). The
ledger writes one JSON file per run at:

    data/calibration_ledger/v{version}_{calibration_date}_{run_timestamp}.json

Append-only by design: never overwrite. The version+date prefix makes
it trivial to compare rubric accuracy across constitution versions on
the same calibration window — e.g. did v2.1's threshold tweak actually
improve recall on the same 30-ticker manifest from 2018, or did it
just shift errors around?

Manifest loading lives in this module too because manifest + ledger
are the two persistent artifacts the calibration loop needs.

Constitution alignment:
  - §22 mandates back-validation against objective historical outcomes.
  - Commitment 1: the manifest itself MUST not be a user-preference
    list. The `selection_principle` field on the loaded manifest is
    surfaced verbatim in the ledger entry so reviewers can audit
    *why* each ticker is on the list.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# `back_validation` is imported lazily inside helpers to avoid a circular
# import path when the ledger is consumed from inside the same package
# (e.g. from a script or test that also pulls the back-validation
# orchestrator). The dataclasses we touch live in plain modules already.
from wise_investor.screening.back_validation import (
    AxisPersistenceOutcome,
    BackValidationSummary,
    StockReturnOutcome,
    TickerBackValidation,
)
from wise_investor.screening.types import AxisVerdict, PrefilterResult

# Resolve repo root from this file's location: src/wise_investor/screening/<file>
# → 3 parents up = repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER_DIR = REPO_ROOT / "data" / "calibration_ledger"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "calibration" / "manifest.yaml"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestEntry:
    """One ticker on the calibration manifest."""

    symbol: str
    rationale: str
    market: str  # "US", "US-ADR", "KR", or whatever convention the manifest uses


@dataclass(frozen=True)
class CalibrationManifest:
    """Loaded calibration manifest."""

    selection_principle: str
    default_calibration_date: dt.date
    default_horizon_years: int
    entries: tuple[ManifestEntry, ...]

    @property
    def symbols(self) -> list[str]:
        return [e.symbol for e in self.entries]


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> CalibrationManifest:
    """Load and validate a calibration manifest YAML.

    Raises:
        FileNotFoundError: when the path doesn't exist.
        ValueError: when the manifest has no tickers.
    """
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    metadata = raw.get("metadata", {}) or {}
    selection_principle = str(metadata.get("selection_principle", "")).strip()
    cal_date_raw = metadata.get("default_calibration_date")
    horizon_years = int(metadata.get("default_horizon_years", 5))

    if cal_date_raw is None:
        default_calibration_date = dt.date(2018, 6, 30)
    elif isinstance(cal_date_raw, dt.date):
        default_calibration_date = cal_date_raw
    else:
        default_calibration_date = dt.date.fromisoformat(str(cal_date_raw))

    entries = tuple(
        ManifestEntry(
            symbol=str(t["symbol"]),
            rationale=str(t.get("rationale", "")),
            market=str(t.get("market", "US")),
        )
        for t in (raw.get("tickers") or [])
        if t.get("symbol")
    )

    if not entries:
        raise ValueError(f"manifest at {path} contains no tickers")

    return CalibrationManifest(
        selection_principle=selection_principle,
        default_calibration_date=default_calibration_date,
        default_horizon_years=horizon_years,
        entries=entries,
    )


# ---------------------------------------------------------------------------
# Ledger I/O
# ---------------------------------------------------------------------------


def write_ledger_entry(
    summary: BackValidationSummary,
    *,
    ledger_dir: Path = DEFAULT_LEDGER_DIR,
    manifest: CalibrationManifest | None = None,
    run_timestamp: dt.datetime | None = None,
) -> Path:
    """Persist one back-validation run to the ledger.

    The optional `manifest` is recorded for audit so the ledger entry
    is self-contained: the rationale list and selection principle are
    captured at the moment of the run, even if the manifest file is
    later edited.
    """
    ledger_dir.mkdir(parents=True, exist_ok=True)
    if run_timestamp is None:
        run_timestamp = dt.datetime.now(tz=dt.UTC)

    fname = (
        f"v{summary.constitution_version}_"
        f"{summary.calibration_date.isoformat()}_"
        f"{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    path = ledger_dir / fname

    payload: dict[str, Any] = {
        "schema_version": 1,
        "constitution_version": summary.constitution_version,
        "calibration_date": summary.calibration_date.isoformat(),
        "horizon_date": summary.horizon_date.isoformat(),
        "run_timestamp_utc": run_timestamp.isoformat(),
        "n_tickers": summary.n_tickers,
        "n_advanced": summary.n_advanced,
        "n_rejected": summary.n_rejected,
        "advanced_avg_excess_return": summary.advanced_avg_excess_return,
        "rejected_avg_excess_return": summary.rejected_avg_excess_return,
        "moat_persistence_rate": summary.moat_persistence_rate,
        "bottleneck_persistence_rate": summary.bottleneck_persistence_rate,
        "per_ticker_records": [
            _serialize_record(r) for r in summary.per_ticker_records
        ],
    }

    if manifest is not None:
        payload["manifest"] = {
            "selection_principle": manifest.selection_principle,
            "default_calibration_date": manifest.default_calibration_date.isoformat(),
            "default_horizon_years": manifest.default_horizon_years,
            "entries": [
                {"symbol": e.symbol, "rationale": e.rationale, "market": e.market}
                for e in manifest.entries
            ],
        }

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return path


def list_ledger_entries(ledger_dir: Path = DEFAULT_LEDGER_DIR) -> list[Path]:
    """Return all ledger files, sorted by name (which sorts chronologically
    because the run timestamp is encoded in the filename)."""
    if not ledger_dir.exists():
        return []
    return sorted(ledger_dir.glob("v*.json"))


def load_ledger_entry(path: Path) -> dict[str, Any]:
    """Read one ledger entry as a JSON dict."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Internal serialization helpers
# ---------------------------------------------------------------------------


def _serialize_record(record: TickerBackValidation) -> dict[str, Any]:
    """Convert a TickerBackValidation into a JSON-friendly dict.

    The frozen dataclasses are nested deep enough that asdict() would
    dump the full proxy detail dicts; we explicitly project only the
    fields a calibration reviewer actually needs to read. Raw proxies
    are reachable through the prefilter result's `details` field if
    a future audit needs them — kept here as `prefilter.moat.details`.
    """
    payload: dict[str, Any] = {
        "symbol": record.symbol,
        "calibration_date": record.calibration_date.isoformat(),
        "horizon_date": record.horizon_date.isoformat(),
        "constitution_version": record.constitution_version,
        "prefilter": _serialize_prefilter(record.prefilter_result),
        "return_outcome": _serialize_return(record.return_outcome),
        "axis_persistence": [
            _serialize_persistence(o) for o in record.axis_persistence
        ],
    }
    # P3-5 (2026-04): include the Stage 3 LLM result when present.
    # Absent when calibration ran with --with-stage3=False (legacy
    # flow) — distinguishable from null-LLM-output by presence of
    # the key.
    if record.stage3_result is not None:
        payload["stage3"] = _serialize_stage3(record.stage3_result)
    return payload


def _serialize_prefilter(p: PrefilterResult) -> dict[str, Any]:
    return {
        "hierarchy_decision": p.hierarchy_decision,
        "passed_axes": list(p.passed_axes),
        "need_llm_axes": list(p.need_llm_axes),
        "excluded_reason": p.excluded_reason,
        "moat": _serialize_axis_verdict(p.moat),
        "new_frontier": _serialize_axis_verdict(p.new_frontier),
        "bottleneck": _serialize_axis_verdict(p.bottleneck),
    }


def _serialize_axis_verdict(v: AxisVerdict) -> dict[str, Any]:
    return {
        "axis": v.axis,
        "verdict": v.verdict,
        "reason": v.reason,
        "details": v.details,
    }


def _serialize_return(r: StockReturnOutcome) -> dict[str, Any]:
    return {
        "symbol": r.symbol,
        "start_date": r.start_date.isoformat(),
        "end_date": r.end_date.isoformat(),
        "ticker_return": r.ticker_return,
        "benchmark_return": r.benchmark_return,
        "excess_return": r.excess_return,
    }


def _serialize_persistence(o: AxisPersistenceOutcome) -> dict[str, Any]:
    return {
        "axis": o.axis,
        "passed_at_calibration": o.passed_at_calibration,
        "persisted_at_horizon": o.persisted_at_horizon,
        "detail": o.detail,
    }


def _serialize_stage3(s: Any) -> dict[str, Any]:
    """Serialize a Stage3Result. Inputs come via record.stage3_result
    so the type is `Stage3Result` at the call site; we type Any here
    to keep the import lazy (Stage3Result lives in `screening.types`
    which is already imported by this module, so we just use it).
    """
    def _axis(o: Any) -> dict[str, Any]:
        return {
            "axis": o.axis,
            "verdict": o.verdict,
            "qualifier": o.qualifier,
            "reasoning": o.reasoning,
        }
    return {
        "constitution_version": s.constitution_version,
        "hierarchy_decision": s.hierarchy_decision,
        "rejection_reason": s.rejection_reason,
        "llm_reported_decision": s.llm_reported_decision,
        "moat": _axis(s.moat),
        "new_frontier": _axis(s.new_frontier),
        "bottleneck": _axis(s.bottleneck),
        # raw_llm_output omitted intentionally — it can be very long
        # and we already capture the parsed verdicts. A future audit
        # tool can re-prompt with the same proxies if needed.
    }


__all__ = [
    "CalibrationManifest",
    "DEFAULT_LEDGER_DIR",
    "DEFAULT_MANIFEST_PATH",
    "ManifestEntry",
    "list_ledger_entries",
    "load_ledger_entry",
    "load_manifest",
    "write_ledger_entry",
]
