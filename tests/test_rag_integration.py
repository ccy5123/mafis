"""Tests for the RAG integration layer (rag/integration.py + runner hook).

These tests monkey-patch the three heavy boundaries — EDGAR HTTP,
ChromaDB search, and the facts-cache path — so the crew's pre_gather
hook is exercised fully offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.rag import integration
from wise_investor.rag.edgar import EdgarError, FilingRef
from wise_investor.rag.index import PassageHit


def _fake_filing_ref() -> FilingRef:
    return FilingRef(
        cik="0001045810",
        symbol="NVDA",
        form="10-K",
        filing_date="2025-02-26",
        accession_number="0001045810-25-000023",
        primary_document="nvda-20250126.htm",
    )


def _fake_hit(section: str, text: str, chunk_id: int = 0) -> PassageHit:
    return PassageHit(
        symbol="NVDA",
        section=section,
        filing_date="2025-02-26",
        chunk_id=chunk_id,
        text=text,
        distance=0.33,
    )


# ---------------------------------------------------------------------------
# ensure_10k_indexed
# ---------------------------------------------------------------------------


def test_ensure_10k_indexed_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = _fake_filing_ref()

    monkeypatch.setattr(
        integration,
        "download_10k",
        lambda symbol, use_cache=True: (ref, "<html><body>stub</body></html>"),
    )

    captured: dict = {}

    class _StubParsed:
        def as_dict(self) -> dict[str, str]:
            return {"business": "We make GPUs.", "risk_factors": "Supply risk."}

    monkeypatch.setattr(integration, "extract_sections", lambda html: _StubParsed())

    def _fake_upsert(symbol: str, filing_date: str, sections: dict[str, str]) -> int:
        captured["symbol"] = symbol
        captured["filing_date"] = filing_date
        captured["sections"] = sections
        return len(sections)

    monkeypatch.setattr(integration, "upsert_10k_sections", _fake_upsert)

    result = integration.ensure_10k_indexed("NVDA")
    assert result is ref
    assert captured["symbol"] == "NVDA"
    assert captured["filing_date"] == "2025-02-26"
    assert set(captured["sections"]) == {"business", "risk_factors"}


def test_ensure_10k_indexed_returns_none_on_edgar_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(symbol: str, use_cache: bool = True):
        raise EdgarError(f"no CIK for {symbol}")

    monkeypatch.setattr(integration, "download_10k", _raise)
    # Korean/non-SEC ticker scenario.
    assert integration.ensure_10k_indexed("005930.KS") is None


def test_ensure_10k_indexed_tolerates_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _fake_filing_ref()
    monkeypatch.setattr(
        integration, "download_10k", lambda symbol, use_cache=True: (ref, "<html/>")
    )

    def _boom(html: str):
        raise RuntimeError("lxml exploded")

    monkeypatch.setattr(integration, "extract_sections", _boom)
    # Parse failure still returns the FilingRef — the filing exists, we just
    # couldn't index it.
    assert integration.ensure_10k_indexed("NVDA") is ref


# ---------------------------------------------------------------------------
# gather_section_passages
# ---------------------------------------------------------------------------


def test_gather_section_passages_uses_default_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integration, "stats", lambda: {"total_chunks": 10})

    def _fake_search(query: str, symbol: str, k: int = 3):
        # Return a hit whose text echoes the query so we can distinguish labels.
        return [_fake_hit("risk_factors", f"hit for: {query}", chunk_id=0)]

    monkeypatch.setattr(integration, "search", _fake_search)

    sections = integration.gather_section_passages("NVDA")
    # Every default label is present.
    assert set(sections) == set(integration.DEFAULT_QUERIES)
    for label, section in sections.items():
        assert section.label == label
        assert len(section.passages) == 1
        assert section.filing_date == "2025-02-26"


def test_gather_section_passages_empty_index_returns_empty_passages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integration, "stats", lambda: {"total_chunks": 0})
    # search() must NOT be called when the index is empty.
    monkeypatch.setattr(
        integration,
        "search",
        lambda *a, **k: pytest.fail("search should not be called on empty index"),
    )

    sections = integration.gather_section_passages("NVDA")
    for section in sections.values():
        assert section.passages == []
        assert section.filing_date is None


def test_gather_section_passages_swallows_search_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integration, "stats", lambda: {"total_chunks": 10})

    def _boom(query: str, symbol: str, k: int = 3):
        raise RuntimeError("chroma exploded")

    monkeypatch.setattr(integration, "search", _boom)

    sections = integration.gather_section_passages("NVDA")
    assert set(sections) == set(integration.DEFAULT_QUERIES)
    for section in sections.values():
        assert section.passages == []


# ---------------------------------------------------------------------------
# format_passages_as_tool_output
# ---------------------------------------------------------------------------


def test_format_passages_includes_citation_hints() -> None:
    section = integration.SectionPassages(
        label="risk_factors",
        filing_date="2025-02-26",
        passages=[
            _fake_hit(
                "risk_factors",
                "Our business is dependent on a limited number of foundry partners.",
            ),
            _fake_hit(
                "risk_factors",
                "Export controls could restrict our ability to sell in certain regions.",
                chunk_id=1,
            ),
        ],
    )
    body = integration.format_passages_as_tool_output("NVDA", section)
    assert "NVDA" in body
    assert "filed 2025-02-26" in body
    # Each passage has a copy-paste-ready citation hint.
    assert body.count("[Cite as: 10-K risk_factors, filed 2025-02-26]") == 2
    assert "foundry partners" in body


def test_format_passages_truncates_long_passages() -> None:
    long_text = "A" * 5000
    section = integration.SectionPassages(
        label="business_segments",
        filing_date="2025-02-26",
        passages=[_fake_hit("business", long_text)],
    )
    body = integration.format_passages_as_tool_output(
        "NVDA", section, max_chars_per_passage=200
    )
    assert "..." in body
    # Body should contain at most a modest multiple of the passage cap.
    assert len(body) < 1500


def test_format_passages_handles_empty_section() -> None:
    section = integration.SectionPassages(
        label="moat_signals", filing_date=None, passages=[]
    )
    body = integration.format_passages_as_tool_output("NVDA", section)
    assert "No passages" in body


# ---------------------------------------------------------------------------
# gather_and_format_for_pre_gather (end-to-end)
# ---------------------------------------------------------------------------


def test_gather_and_format_returns_error_entries_on_missing_filing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(integration, "ensure_10k_indexed", lambda sym: None)

    out = integration.gather_and_format_for_pre_gather("005930.KS")
    for label in integration.DEFAULT_QUERIES:
        key = f"edgar.{label}"
        assert key in out
        assert out[key].startswith("ERROR:")


def test_gather_and_format_populates_all_four_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ref = _fake_filing_ref()
    monkeypatch.setattr(integration, "ensure_10k_indexed", lambda sym: ref)
    monkeypatch.setattr(integration, "stats", lambda: {"total_chunks": 10})
    monkeypatch.setattr(
        integration,
        "search",
        lambda query, symbol, k=3: [_fake_hit("risk_factors", "We depend on TSMC.")],
    )

    out = integration.gather_and_format_for_pre_gather("NVDA")
    assert set(out) == {f"edgar.{label}" for label in integration.DEFAULT_QUERIES}
    for body in out.values():
        assert "NVDA" in body
        assert "[Cite as: 10-K" in body


# ---------------------------------------------------------------------------
# Runner integration: pre_gather_facts surfaces edgar.* entries
# ---------------------------------------------------------------------------


def test_pre_gather_facts_includes_edgar_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from wise_investor.agents import runner

    # Redirect the facts cache to a tmp dir.
    monkeypatch.setattr(runner, "FACTS_CACHE_DIR", tmp_path)
    monkeypatch.setattr(runner, "_facts_cache_path", lambda sym: tmp_path / f"{sym}.json")

    # Stub every Phase 1A tool to return a sentinel string so we don't hit
    # the network or require API keys.
    monkeypatch.setattr(runner, "_exec_cross_validate_quote", lambda args: "CVQ")
    monkeypatch.setattr(runner, "_exec_calculate_per", lambda args: "PER")
    monkeypatch.setattr(runner, "_exec_calculate_ev_ebitda", lambda args: "EVE")
    monkeypatch.setattr(runner, "_exec_get_peer_multiples", lambda args: "PEERS")
    monkeypatch.setattr(runner, "_exec_reverse_dcf", lambda args: "RDCF")
    monkeypatch.setattr(runner, "_exec_fetch_field", lambda args: f"FIELD:{args['field']}")
    monkeypatch.setattr(runner, "get_macro_snapshot", lambda: object())
    monkeypatch.setattr(runner, "format_macro_snapshot", lambda snap: "MACRO_OK")
    monkeypatch.setattr(runner, "_load_peer_overrides_for", lambda sym: [])

    # Force the RAG integration to return predictable edgar.* bodies.
    import wise_investor.rag.integration as integration_mod

    def _stub_rag(symbol: str, queries=None, k: int = 3) -> dict[str, str]:
        return {
            "edgar.business_segments": f"BIZ-BLOCK-{symbol}",
            "edgar.moat_signals": f"MOAT-BLOCK-{symbol}",
            "edgar.risk_factors": f"RISK-BLOCK-{symbol}",
            "edgar.mdna_highlights": f"MDNA-BLOCK-{symbol}",
        }

    monkeypatch.setattr(
        integration_mod, "gather_and_format_for_pre_gather", _stub_rag
    )

    facts = runner.pre_gather_facts("NVDA", use_cache=False)
    for label in integration_mod.DEFAULT_QUERIES:
        key = f"edgar.{label}"
        assert key in facts
        assert facts[key].endswith("NVDA")
    # Existing keys still there.
    assert "calculate_per" in facts
    assert facts["fred.macro_snapshot"] == "MACRO_OK"


def test_wrap_user_prompt_includes_10k_citation_rule() -> None:
    from wise_investor.agents.runner import _wrap_user_prompt_with_facts

    wrapped = _wrap_user_prompt_with_facts(
        "Task body.", {"edgar.risk_factors": "10-K risk block"}
    )
    # Universal citation rule now teaches the model to cite 10-K excerpts.
    assert "10-K EXCERPT CITATIONS" in wrapped
    assert "[Cite as:" in wrapped
    # The facts block is still rendered as <tool_output> XML.
    assert '<tool_output name="edgar.risk_factors">' in wrapped
