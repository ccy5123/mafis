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


def test_gather_raw_material_dispatches_korean_to_dart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Korean tickers MUST NOT hit the Finnhub / SEC EDGAR code paths
    (which have no data on KRX listings). Verified by stubbing both
    Finnhub and EDGAR to raise if called, and the DART facts adapter
    to return a sentinel dict.
    """
    import wise_investor.onboarding.brief_generator as bg_mod
    import wise_investor.data.dart_facts as dart_facts_mod

    # Finnhub / EDGAR must NOT be invoked for Korean symbols.
    class _UnreachableFinnhub:
        def __enter__(self):
            raise AssertionError("Finnhub path fired for Korean ticker")

        def __exit__(self, *a):
            pass

    import wise_investor.data.finnhub as finnhub_mod

    monkeypatch.setattr(finnhub_mod, "FinnhubClient", _UnreachableFinnhub)

    import wise_investor.rag.integration as rag_mod

    def _boom_rag(symbol):
        raise AssertionError("EDGAR path fired for Korean ticker")

    monkeypatch.setattr(rag_mod, "ensure_10k_indexed", _boom_rag)

    # Stub DART client + adapter.
    class _StubDart:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def corp_code_from_stock_code(self, code):
            return "00126380"

        def load_corp_mapping(self):
            from wise_investor.data.dart import CorpMapping

            return [
                CorpMapping(
                    corp_code="00126380",
                    corp_name="삼성전자",
                    stock_code="005930",
                    modify_date="20260401",
                )
            ]

        def close(self):
            pass

    import wise_investor.data.dart as dart_mod

    monkeypatch.setattr(dart_mod, "DartClient", lambda *a, **k: _StubDart())

    monkeypatch.setattr(
        dart_facts_mod,
        "pre_gather_dart_facts",
        lambda code: {
            "dart.metadata": "Symbol: 005930 / Name: 삼성전자",
            "dart.revenue": "revenue: 300T KRW",
        },
    )

    # Skip geopolitics (noisy; not under test here).
    monkeypatch.setattr(bg_mod, "_attach_geopolitics", lambda raw, sym: None)

    raw = bg_mod.gather_raw_material("005930")
    assert raw.symbol == "005930"
    assert raw.company_name == "삼성전자"
    assert "Korean listing" in (raw.industry or "")
    # DART financial lines end up in the business excerpt.
    assert "revenue: 300T KRW" in raw.edgar_business_excerpt
    assert raw.edgar_filing_date == "(DART annual filing)"


def test_build_brief_prompt_adds_korean_caveat_for_dart_source() -> None:
    """When the input signals DART (filing_date placeholder), the
    prompt MUST include the Korean-listing caveat so the LLM doesn't
    hallucinate US-based suppliers from pretraining memory.
    """
    raw = RawMaterial(
        symbol="005930",
        company_name="삼성전자",
        industry="(Korean listing; DART does not classify)",
        edgar_filing_date="(DART annual filing)",
        edgar_business_excerpt="[DART-sourced] revenue: 300T KRW",
    )
    _, user = build_brief_prompt(raw)
    assert "Korean-listing caveat" in user
    assert "DO NOT list suppliers" in user
    assert "DART" in user


def test_build_brief_prompt_no_korean_caveat_for_us_ticker() -> None:
    raw = RawMaterial(
        symbol="NVDA",
        company_name="NVIDIA",
        industry="Semiconductors",
        edgar_filing_date="2026-02-25",  # real ISO date
    )
    _, user = build_brief_prompt(raw)
    assert "Korean-listing caveat" not in user


def test_generate_draft_source_line_korean() -> None:
    """Korean-path header should say 'DART ...' not 'SEC 10-K ((...))'."""
    raw = RawMaterial(
        symbol="005930",
        company_name="삼성전자",
        industry="(Korean listing; DART does not classify)",
        edgar_filing_date="(DART annual filing)",
    )

    def _stub(system, user):
        return "## Peer Override\n- (none)\n"

    out = generate_value_chain_draft("005930", raw=raw, llm_call=_stub)
    assert "DART" in out
    # No double-paren regression.
    assert "((DART" not in out
    assert "SEC 10-K (" not in out


def test_generate_draft_source_line_us_ticker() -> None:
    raw = RawMaterial(
        symbol="NVDA",
        company_name="NVIDIA",
        industry="Semiconductors",
        edgar_filing_date="2026-02-25",
    )

    def _stub(system, user):
        return "## Peer Override\n- (none)\n"

    out = generate_value_chain_draft("NVDA", raw=raw, llm_call=_stub)
    assert "SEC 10-K filed 2026-02-25" in out


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
