"""Tests for the 3-Tier ticker registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.ticker_registry import (
    RegistryError,
    TickerEntry,
    TickerRegistry,
    load_registry,
)


def _write_registry(
    tmp_path: Path, body: str, vc_briefs: list[str] | None = None
) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    reg_path = config_dir / "tickers.yaml"
    reg_path.write_text(body, encoding="utf-8")
    # Create dummy value-chain briefs so Tier 1 validation passes.
    if vc_briefs:
        vc_dir = tmp_path / "docs" / "value_chains"
        vc_dir.mkdir(parents=True)
        for sym in vc_briefs:
            (vc_dir / f"{sym}.md").write_text("# dummy", encoding="utf-8")
    return reg_path


def test_load_registry_basic(tmp_path: Path) -> None:
    body = """
tier_1:
  - symbol: NVDA
    notes: core
tier_2:
  - AAPL
  - symbol: MSFT
tier_3: []
"""
    reg_path = _write_registry(tmp_path, body, vc_briefs=["NVDA"])
    vc_dir = tmp_path / "docs" / "value_chains"
    reg = load_registry(reg_path, value_chains_dir=vc_dir)
    assert isinstance(reg, TickerRegistry)
    assert reg.symbols_by_tier("tier_1") == ["NVDA"]
    assert reg.symbols_by_tier("tier_2") == ["AAPL", "MSFT"]
    assert reg.symbols_by_tier("tier_3") == []

    nvda = reg.find("nvda")  # case-insensitive lookup
    assert isinstance(nvda, TickerEntry)
    assert nvda.tier == "tier_1"
    assert nvda.notes == "core"


def test_load_registry_rejects_duplicate_across_tiers(tmp_path: Path) -> None:
    body = """
tier_1:
  - symbol: NVDA
tier_2:
  - NVDA
"""
    reg_path = _write_registry(tmp_path, body, vc_briefs=["NVDA"])
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(reg_path, value_chains_dir=tmp_path / "docs" / "value_chains")


def test_load_registry_rejects_missing_value_chain(tmp_path: Path) -> None:
    body = """
tier_1:
  - symbol: NEWCO
tier_2: []
tier_3: []
"""
    reg_path = _write_registry(tmp_path, body, vc_briefs=[])  # no value chain
    with pytest.raises(RegistryError, match="value chain brief"):
        load_registry(reg_path, value_chains_dir=tmp_path / "docs" / "value_chains")


def test_load_registry_strict_false_allows_missing_vc(tmp_path: Path) -> None:
    body = """
tier_1:
  - symbol: NEWCO
"""
    reg_path = _write_registry(tmp_path, body, vc_briefs=[])
    reg = load_registry(
        reg_path,
        value_chains_dir=tmp_path / "docs" / "value_chains",
        strict=False,
    )
    assert reg.symbols_by_tier("tier_1") == ["NEWCO"]


def test_load_registry_rejects_non_mapping_entry(tmp_path: Path) -> None:
    body = """
tier_2:
  - 123
"""
    reg_path = _write_registry(tmp_path, body)
    with pytest.raises(RegistryError):
        load_registry(
            reg_path,
            value_chains_dir=tmp_path / "docs" / "value_chains",
            strict=False,
        )


def test_load_registry_empty_tiers_are_optional(tmp_path: Path) -> None:
    body = "tier_1: []\n"
    reg_path = _write_registry(tmp_path, body)
    reg = load_registry(
        reg_path,
        value_chains_dir=tmp_path / "docs" / "value_chains",
        strict=False,
    )
    assert reg.entries == []


def test_load_registry_file_not_found() -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_registry(Path("/tmp/definitely-not-there.yaml"), strict=False)


def test_load_registry_rejects_non_mapping_root(tmp_path: Path) -> None:
    body = "- just a list\n"
    reg_path = _write_registry(tmp_path, body)
    with pytest.raises(RegistryError, match="top-level mapping"):
        load_registry(reg_path, strict=False)


def test_real_project_registry_loads() -> None:
    """The real config/tickers.yaml shipped in the repo must always load."""
    reg = load_registry()  # default path
    symbols = [e.symbol for e in reg.entries]
    # Tier 1 has both NVDA and GEV after the Phase 2-B commit.
    assert "NVDA" in reg.symbols_by_tier("tier_1")
    assert "GEV" in reg.symbols_by_tier("tier_1")
    # Tier counts are in the configured ranges.
    assert 1 <= len(reg.by_tier("tier_1")) <= 5
    assert 0 <= len(reg.by_tier("tier_2")) <= 20
    # Every symbol unique.
    assert len(set(symbols)) == len(symbols)
