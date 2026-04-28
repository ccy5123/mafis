"""Calibration ledger + manifest tests.

Manifest loading is a pure file-parse round-trip. Ledger writes are
exercised by constructing a synthetic BackValidationSummary and
asserting the JSON output preserves every field a downstream
calibration reviewer needs.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from wise_investor.screening import CONSTITUTION_VERSION
from wise_investor.screening.back_validation import (
    AxisPersistenceOutcome,
    BackValidationSummary,
    StockReturnOutcome,
    TickerBackValidation,
)
from wise_investor.screening.calibration_ledger import (
    DEFAULT_MANIFEST_PATH,
    CalibrationManifest,
    ManifestEntry,
    list_ledger_entries,
    load_ledger_entry,
    load_manifest,
    write_ledger_entry,
)
from wise_investor.screening.types import (
    AxisVerdict,
    PrefilterResult,
    Segment,
    SegmentBreakdown,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _axis_verdict(axis, verdict, reason="ok"):
    return AxisVerdict(axis=axis, verdict=verdict, reason=reason, details={})


def _segment_breakdown(year=2018):
    return SegmentBreakdown(
        primary_segment_exists=True,
        primary_segment_name="Core",
        primary_segment_revenue_share=1.0,
        all_segments=(Segment(name="Core", revenue=None, share_of_total=1.0),),
        fiscal_year=year,
        source="stub",
    )


def _prefilter_result(symbol="TEST", decision="ADVANCE_TO_STAGE_3"):
    return PrefilterResult(
        symbol=symbol,
        constitution_version=CONSTITUTION_VERSION,
        moat=_axis_verdict("moat", "PASS"),
        new_frontier=_axis_verdict("new_frontier", "FAIL", "no frontier signals"),
        bottleneck=_axis_verdict("bottleneck", "PASS"),
        primary_segment=_segment_breakdown(),
        excluded_reason=None,
        hierarchy_decision=decision,
        passed_axes=("moat", "bottleneck"),
        need_llm_axes=(),
    )


def _record(symbol="TEST", decision="ADVANCE_TO_STAGE_3", excess=0.10):
    cal = dt.date(2018, 6, 30)
    horizon = dt.date(2023, 6, 30)
    return TickerBackValidation(
        symbol=symbol,
        calibration_date=cal,
        horizon_date=horizon,
        constitution_version=CONSTITUTION_VERSION,
        prefilter_result=_prefilter_result(symbol, decision),
        return_outcome=StockReturnOutcome(
            symbol=symbol,
            start_date=cal,
            end_date=horizon,
            ticker_return=0.30 + (excess or 0),
            benchmark_return=0.30,
            excess_return=excess,
        ),
        axis_persistence=(
            AxisPersistenceOutcome(
                axis="moat",
                passed_at_calibration=True,
                persisted_at_horizon=True,
                detail="ROIC advantage at T+5: 0.07 (>= 0.05)",
            ),
            AxisPersistenceOutcome(
                axis="new_frontier",
                passed_at_calibration=False,
                persisted_at_horizon=None,
                detail="did not pass new_frontier at calibration (FAIL)",
            ),
            AxisPersistenceOutcome(
                axis="bottleneck",
                passed_at_calibration=True,
                persisted_at_horizon=None,
                detail="top-5 customer share unavailable at T+horizon",
            ),
        ),
    )


def _summary(records):
    s = BackValidationSummary(
        constitution_version=CONSTITUTION_VERSION,
        calibration_date=dt.date(2018, 6, 30),
        horizon_date=dt.date(2023, 6, 30),
    )
    for r in records:
        s.add(r)
    return s.finalize()


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------


def test_default_manifest_exists_and_loads() -> None:
    """The committed manifest at data/calibration/manifest.yaml must
    parse cleanly — calibration runs depend on it.
    """
    assert DEFAULT_MANIFEST_PATH.exists(), (
        f"Default manifest missing at {DEFAULT_MANIFEST_PATH}"
    )
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    assert isinstance(manifest, CalibrationManifest)
    assert len(manifest.entries) >= 30
    # Every entry must carry a rationale — Commitment 1 audit trail.
    for entry in manifest.entries:
        assert entry.rationale.strip(), f"{entry.symbol} has empty rationale"


def test_default_manifest_includes_axis_diversity() -> None:
    """The manifest should cover all three axes plus negative controls;
    enforce a minimum count by market so a future regression that
    accidentally trims the list to one market gets caught.
    """
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    markets = {e.market for e in manifest.entries}
    assert "US" in markets
    assert "KR" in markets


def test_manifest_with_no_tickers_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("metadata: {}\ntickers: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(p)


def test_manifest_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "does_not_exist.yaml")


def test_manifest_metadata_defaults_when_unset(tmp_path: Path) -> None:
    p = tmp_path / "minimal.yaml"
    p.write_text(
        "tickers:\n  - {symbol: AAA, rationale: 'x', market: US}\n",
        encoding="utf-8",
    )
    manifest = load_manifest(p)
    # Default calibration date and horizon should fall back without error.
    assert isinstance(manifest.default_calibration_date, dt.date)
    assert manifest.default_horizon_years == 5
    assert manifest.entries[0].symbol == "AAA"


def test_manifest_handles_yaml_native_date(tmp_path: Path) -> None:
    """When YAML auto-parses a date (no quotes), it lands as dt.date.
    The loader must accept it without re-parsing.
    """
    p = tmp_path / "native_date.yaml"
    p.write_text(
        "metadata:\n"
        "  default_calibration_date: 2017-06-30\n"
        "  default_horizon_years: 4\n"
        "tickers:\n"
        "  - {symbol: AAA, rationale: 'x', market: US}\n",
        encoding="utf-8",
    )
    manifest = load_manifest(p)
    assert manifest.default_calibration_date == dt.date(2017, 6, 30)
    assert manifest.default_horizon_years == 4


def test_manifest_symbols_property() -> None:
    manifest = CalibrationManifest(
        selection_principle="test",
        default_calibration_date=dt.date(2018, 6, 30),
        default_horizon_years=5,
        entries=(
            ManifestEntry(symbol="AAA", rationale="r", market="US"),
            ManifestEntry(symbol="BBB", rationale="r", market="KR"),
        ),
    )
    assert manifest.symbols == ["AAA", "BBB"]


# ---------------------------------------------------------------------------
# Ledger write tests
# ---------------------------------------------------------------------------


def test_ledger_write_creates_file_with_correct_name(tmp_path: Path) -> None:
    summary = _summary([_record()])
    ts = dt.datetime(2026, 4, 27, 12, 30, 0, tzinfo=dt.UTC)
    path = write_ledger_entry(summary, ledger_dir=tmp_path, run_timestamp=ts)
    assert path.exists()
    assert path.name == f"v{CONSTITUTION_VERSION}_2018-06-30_20260427T123000Z.json"


def test_ledger_payload_round_trip(tmp_path: Path) -> None:
    summary = _summary([_record(symbol="AAA", excess=0.20)])
    path = write_ledger_entry(summary, ledger_dir=tmp_path)
    payload = load_ledger_entry(path)

    assert payload["constitution_version"] == CONSTITUTION_VERSION
    assert payload["calibration_date"] == "2018-06-30"
    assert payload["horizon_date"] == "2023-06-30"
    assert payload["n_tickers"] == 1
    assert payload["n_advanced"] == 1
    assert payload["n_rejected"] == 0
    assert abs(payload["advanced_avg_excess_return"] - 0.20) < 1e-9

    record = payload["per_ticker_records"][0]
    assert record["symbol"] == "AAA"
    assert record["prefilter"]["hierarchy_decision"] == "ADVANCE_TO_STAGE_3"
    assert record["prefilter"]["moat"]["verdict"] == "PASS"
    assert record["return_outcome"]["excess_return"] == pytest.approx(0.20)
    assert len(record["axis_persistence"]) == 3


def test_ledger_records_manifest_for_audit(tmp_path: Path) -> None:
    """When a manifest is passed, the ledger entry MUST include it
    so reviewers can audit which selection principle was in effect at
    the time of the run, even if the manifest is later edited.
    """
    summary = _summary([_record()])
    manifest = CalibrationManifest(
        selection_principle="axis-diversity, no user preference",
        default_calibration_date=dt.date(2018, 6, 30),
        default_horizon_years=5,
        entries=(ManifestEntry(symbol="TEST", rationale="r", market="US"),),
    )
    path = write_ledger_entry(summary, ledger_dir=tmp_path, manifest=manifest)
    payload = load_ledger_entry(path)
    assert "manifest" in payload
    assert payload["manifest"]["selection_principle"].startswith("axis-diversity")
    assert payload["manifest"]["entries"][0]["symbol"] == "TEST"


def test_ledger_omits_manifest_when_not_provided(tmp_path: Path) -> None:
    summary = _summary([_record()])
    path = write_ledger_entry(summary, ledger_dir=tmp_path)
    payload = load_ledger_entry(path)
    assert "manifest" not in payload


def test_ledger_aggregates_advance_and_reject(tmp_path: Path) -> None:
    summary = _summary([
        _record(symbol="GOOD1", decision="ADVANCE_TO_STAGE_3", excess=0.30),
        _record(symbol="GOOD2", decision="ADVANCE_TO_STAGE_3", excess=0.10),
        _record(symbol="BAD1", decision="REJECT", excess=-0.20),
    ])
    path = write_ledger_entry(summary, ledger_dir=tmp_path)
    payload = load_ledger_entry(path)
    assert payload["n_tickers"] == 3
    assert payload["n_advanced"] == 2
    assert payload["n_rejected"] == 1
    assert abs(payload["advanced_avg_excess_return"] - 0.20) < 1e-9
    assert abs(payload["rejected_avg_excess_return"] - (-0.20)) < 1e-9


def test_ledger_creates_directory_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nonexistent_subdir" / "ledger"
    summary = _summary([_record()])
    path = write_ledger_entry(summary, ledger_dir=target)
    assert path.parent == target
    assert target.exists()


def test_list_ledger_entries_sorts_chronologically(tmp_path: Path) -> None:
    summary = _summary([_record()])
    ts1 = dt.datetime(2026, 1, 1, 10, 0, 0, tzinfo=dt.UTC)
    ts2 = dt.datetime(2026, 4, 1, 10, 0, 0, tzinfo=dt.UTC)
    ts3 = dt.datetime(2026, 2, 1, 10, 0, 0, tzinfo=dt.UTC)
    write_ledger_entry(summary, ledger_dir=tmp_path, run_timestamp=ts1)
    write_ledger_entry(summary, ledger_dir=tmp_path, run_timestamp=ts2)
    write_ledger_entry(summary, ledger_dir=tmp_path, run_timestamp=ts3)
    entries = list_ledger_entries(tmp_path)
    assert len(entries) == 3
    # Filename embeds the timestamp, so lexical sort = chronological sort.
    assert "20260101" in entries[0].name
    assert "20260201" in entries[1].name
    assert "20260401" in entries[2].name


def test_list_ledger_entries_empty_when_dir_missing(tmp_path: Path) -> None:
    assert list_ledger_entries(tmp_path / "no_dir") == []


def test_ledger_payload_is_pure_json(tmp_path: Path) -> None:
    """Ledger entries must serialize via plain json.load — no custom
    decoder required. This is what makes them durable across versions
    of the codebase.
    """
    summary = _summary([_record()])
    path = write_ledger_entry(summary, ledger_dir=tmp_path)
    raw = path.read_text(encoding="utf-8")
    # json.loads must succeed without exception.
    payload = json.loads(raw)
    assert payload["schema_version"] == 1
