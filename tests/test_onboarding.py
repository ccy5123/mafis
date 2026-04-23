"""Tests for the Phase 2 ticker-onboarding pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.onboarding.brief_generator import (
    DRAFT_BANNER,
    RawMaterial,
    build_brief_prompt,
    generate_value_chain_draft,
)
from wise_investor.onboarding.tickers_yaml import (
    RegistryError,
    add_ticker_to_registry,
    load_registry_yaml,
    ticker_in_registry,
)


# ---------------------------------------------------------------------------
# RawMaterial + prompt rendering
# ---------------------------------------------------------------------------


def test_raw_material_renders_empty_inputs_gracefully() -> None:
    raw = RawMaterial(symbol="ZZZZ", company_name=None, industry=None)
    ctx = raw.as_prompt_context()
    assert "ZZZZ" in ctx
    assert "(unknown)" in ctx  # company/industry placeholder
    assert "(none returned)" in ctx  # peers placeholder


def test_raw_material_includes_edgar_blocks_when_present() -> None:
    raw = RawMaterial(
        symbol="NVDA",
        company_name="NVIDIA Corp",
        industry="Semiconductors",
        peers=["AMD", "INTC", "AVGO"],
        edgar_business_excerpt="We design GPUs.",
        edgar_risk_factors_excerpt="Supply chain concentrated.",
        edgar_moat_excerpt="CUDA ecosystem.",
        edgar_filing_date="2026-02-25",
    )
    ctx = raw.as_prompt_context()
    assert "NVIDIA Corp" in ctx
    assert "Semiconductors" in ctx
    assert "AMD, INTC, AVGO" in ctx
    assert "2026-02-25" in ctx
    assert "## 10-K Business excerpt" in ctx
    assert "CUDA ecosystem" in ctx


def test_build_brief_prompt_returns_system_and_user() -> None:
    raw = RawMaterial(symbol="NVDA", company_name="NVIDIA", industry=None)
    system, user = build_brief_prompt(raw)
    assert "research associate" in system
    assert "NVDA" in user
    # Template's eight mandated headings are in the prompt.
    for heading in [
        "## Peer Override",
        "## Upstream — Suppliers",
        "## Peers — Direct competition",
        "## Downstream — Customers",
        "## Infrastructure dependencies",
        "## Geopolitical / regulatory pressure points",
        "## Vulnerable links (Skeptic's attack surface)",
        "## Known unknowns (do not pretend to know)",
    ]:
        assert heading in user, f"missing mandated heading: {heading}"
    # Template bans inventing numbers.
    assert "No numbers that do not appear verbatim" in user
    # Template requires uncertainty flag.
    assert "[?UNCERTAIN]" in user


# ---------------------------------------------------------------------------
# generate_value_chain_draft (with stub LLM)
# ---------------------------------------------------------------------------


def test_generate_draft_wraps_llm_output_with_banner() -> None:
    raw = RawMaterial(
        symbol="NVDA",
        company_name="NVIDIA Corp",
        industry="Semiconductors",
        edgar_filing_date="2026-02-25",
    )

    def _stub_llm(system: str, user: str) -> str:
        return "## Peer Override\n- (none)\n\n## Upstream — Suppliers\n- TSMC"

    out = generate_value_chain_draft("NVDA", raw=raw, llm_call=_stub_llm)
    assert out.startswith("# NVDA — Value Chain Brief (auto-drafted)")
    assert DRAFT_BANNER in out
    # LLM body is appended after the header/banner block.
    assert "## Peer Override" in out
    assert "TSMC" in out
    # Source line references the 10-K filing date and the data sources.
    assert "2026-02-25" in out


def test_generate_draft_passes_raw_material_to_llm() -> None:
    raw = RawMaterial(
        symbol="AMD",
        company_name="Advanced Micro Devices",
        industry="Semiconductors",
        peers=["NVDA", "INTC"],
        edgar_filing_date="2026-02-10",
    )
    captured: dict[str, str] = {}

    def _stub_llm(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return "stub body"

    generate_value_chain_draft("AMD", raw=raw, llm_call=_stub_llm)
    assert "AMD" in captured["user"]
    assert "Advanced Micro Devices" in captured["user"]
    assert "NVDA, INTC" in captured["user"]


# ---------------------------------------------------------------------------
# tickers.yaml CRUD
# ---------------------------------------------------------------------------


def test_load_registry_returns_empty_tiers_when_missing(tmp_path: Path) -> None:
    data = load_registry_yaml(tmp_path / "does_not_exist.yaml")
    assert data == {"tier_1": [], "tier_2": [], "tier_3": []}


def test_add_ticker_creates_fresh_file(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "NVDA", tier=1, notes="primary target")
    data = load_registry_yaml(path)
    assert data["tier_1"] == [{"symbol": "NVDA", "notes": "primary target"}]
    assert data["tier_2"] == []
    assert data["tier_3"] == []


def test_add_ticker_without_notes(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "AMD", tier=2)
    data = load_registry_yaml(path)
    assert data["tier_2"][0] == {"symbol": "AMD"}


def test_add_ticker_appends_to_existing(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "NVDA", tier=1)
    add_ticker_to_registry(path, "GEV", tier=1)
    data = load_registry_yaml(path)
    symbols = [e["symbol"] for e in data["tier_1"]]
    assert symbols == ["NVDA", "GEV"]


def test_add_ticker_rejects_duplicate_without_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "NVDA", tier=1)
    with pytest.raises(RegistryError, match="tier_1"):
        add_ticker_to_registry(path, "NVDA", tier=2)


def test_add_ticker_overwrite_moves_tier(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "NVDA", tier=2)
    add_ticker_to_registry(path, "NVDA", tier=1, overwrite=True)
    data = load_registry_yaml(path)
    assert [e["symbol"] for e in data["tier_1"]] == ["NVDA"]
    assert data["tier_2"] == []


def test_add_ticker_rejects_invalid_tier(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    with pytest.raises(ValueError, match="tier must be one of"):
        add_ticker_to_registry(path, "NVDA", tier=4)


def test_ticker_in_registry_finds_by_symbol(tmp_path: Path) -> None:
    path = tmp_path / "tickers.yaml"
    add_ticker_to_registry(path, "NVDA", tier=1)
    add_ticker_to_registry(path, "AMD", tier=2)
    data = load_registry_yaml(path)
    assert ticker_in_registry(data, "NVDA") == 1
    assert ticker_in_registry(data, "AMD") == 2
    assert ticker_in_registry(data, "nvda") == 1  # case-insensitive
    assert ticker_in_registry(data, "ZZZZ") is None


def test_add_ticker_preserves_leading_comments(tmp_path: Path) -> None:
    """If the existing tickers.yaml starts with a `# ...` comment block,
    the save operation must preserve it.
    """
    path = tmp_path / "tickers.yaml"
    path.write_text(
        "# Tier registry banner\n# line 2 of comments\n\ntier_1: []\n"
        "tier_2: []\ntier_3: []\n",
        encoding="utf-8",
    )
    add_ticker_to_registry(path, "NVDA", tier=1)
    saved = path.read_text(encoding="utf-8")
    assert saved.startswith("# Tier registry banner\n# line 2 of comments")
    # Ticker still landed in the file.
    assert "NVDA" in saved
