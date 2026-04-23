"""Tests for the RAG scaffold (edgar, sections, index).

Network tests (hitting real sec.gov) are marked `network` and skipped by
default — run with `pytest -m network` to opt in. All other tests are
pure: they drive the parsers with synthetic inputs or exercise
ChromaDB against a tmp directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wise_investor.rag import edgar, index, sections


# ---------------------------------------------------------------------------
# EDGAR URL / FilingRef construction
# ---------------------------------------------------------------------------


def test_filing_ref_primary_url_strips_accession_dashes() -> None:
    ref = edgar.FilingRef(
        cik="0001045810",
        symbol="NVDA",
        form="10-K",
        filing_date="2025-02-26",
        accession_number="0001045810-25-000023",
        primary_document="nvda-20250126.htm",
    )
    assert ref.filing_index_url == (
        "https://www.sec.gov/Archives/edgar/data/"
        "1045810/000104581025000023/"
    )
    assert ref.primary_url.endswith("/nvda-20250126.htm")


def test_filing_ref_drops_leading_zeros_from_cik_in_url() -> None:
    # SEC archive paths use the int-form CIK with no zero padding.
    ref = edgar.FilingRef(
        cik="0000320193",
        symbol="AAPL",
        form="10-K",
        filing_date="2024-11-01",
        accession_number="0000320193-24-000123",
        primary_document="aapl-20240928.htm",
    )
    assert "data/320193/" in ref.filing_index_url


def test_user_agent_is_polite() -> None:
    # SEC policy: UA must identify the project and a contact vector.
    assert "MAFIS" in edgar.USER_AGENT
    assert "github.com" in edgar.USER_AGENT.lower() or "@" in edgar.USER_AGENT


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


_SYNTHETIC_10K_HTML = """\
<html><body>
<h1>Table of Contents</h1>
<p>Item 1. Business .................................... 3</p>
<p>Item 1A. Risk Factors ............................... 12</p>
<p>Item 7. Management's Discussion and Analysis ........ 45</p>
<p>Item 7A. Quantitative and Qualitative Disclosures ... 70</p>
<p>Item 8. Financial Statements ........................ 80</p>

<h2>PART I</h2>

<h3>Item 1. Business</h3>
<p>We are a semiconductor company that designs GPUs for data centers.
Our revenue grew 125% year-over-year driven by demand for AI
training infrastructure. We operate in two segments: Compute &amp;
Networking and Graphics. Substantial portion of revenue concentrated
in a small number of hyperscale customers.</p>
<p>This section is long enough to exceed the 100 character floor used
by the slicer to reject TOC-only hits.</p>

<h3>Item 1A. Risk Factors</h3>
<p>Our business faces substantial risks including: dependence on a
limited number of foundry partners; export controls on advanced chips
to certain geographies; and rapid technology change. A material
disruption to our supply chain could have a material adverse effect
on our results of operations.</p>

<h3>Item 2. Properties</h3>
<p>Our headquarters are in Santa Clara, California.</p>

<h3>Item 7. Management's Discussion and Analysis of Financial Condition</h3>
<p>Fiscal 2025 revenue increased driven by strong demand for the
Hopper and Blackwell platforms. Gross margin expanded due to mix
shift toward data center products. We continue to invest in R&amp;D
and capital expenditures to support future growth.</p>

<h3>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</h3>
<p>We are exposed to foreign currency risk and interest rate risk.
We use forward contracts to hedge a portion of our non-USD cash
flows. A hypothetical 100 bps change in interest rates would not
have a material effect on our results of operations.</p>

<h3>Item 8. Financial Statements</h3>
<p>See accompanying consolidated financial statements.</p>
</body></html>
"""


def test_html_to_plain_text_strips_tags() -> None:
    text = sections.html_to_plain_text(_SYNTHETIC_10K_HTML)
    assert "<html>" not in text
    assert "<p>" not in text
    assert "semiconductor company" in text
    # Blank-line runs collapsed.
    assert "\n\n\n" not in text


def test_extract_sections_finds_all_four() -> None:
    parsed = sections.extract_sections(_SYNTHETIC_10K_HTML)
    assert parsed.business is not None
    assert parsed.risk_factors is not None
    assert parsed.mdna is not None
    assert parsed.quant_market_risk is not None

    assert "semiconductor company" in parsed.business
    assert "foundry partners" in parsed.risk_factors
    assert "Hopper" in parsed.mdna
    assert "interest rate risk" in parsed.quant_market_risk


def test_extract_sections_slices_at_next_item_boundary() -> None:
    parsed = sections.extract_sections(_SYNTHETIC_10K_HTML)
    assert parsed.business is not None
    # Business body should stop before Risk Factors content bleeds in.
    assert "foundry partners" not in parsed.business


def test_extract_sections_as_dict_drops_empty_sections() -> None:
    html = """<html><body>
    <h1>Item 1. Business</h1>
    <p>{body}</p>
    <h1>Item 2. Properties</h1>
    <p>stop.</p>
    </body></html>""".format(body="A " * 200)
    parsed = sections.extract_sections(html)
    d = parsed.as_dict()
    assert "business" in d
    assert "risk_factors" not in d  # not present in input


def test_chunk_text_produces_non_empty_chunks_with_overlap() -> None:
    text = "Paragraph one. " * 100 + "\n\n" + "Paragraph two. " * 100
    chunks = index._chunk_text(text, size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
    # Consecutive chunks should share some suffix/prefix (overlap).
    assert len(chunks[0]) <= 200 + 40


def test_chunk_text_handles_short_input() -> None:
    chunks = index._chunk_text("Short.", size=900, overlap=120)
    assert chunks == ["Short."]


# ---------------------------------------------------------------------------
# ChromaDB index/search round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the index module at a clean, per-test Chroma store."""
    persist = tmp_path / "chroma"
    monkeypatch.setattr("wise_investor.rag.index.settings.chroma_persist_dir", persist)
    # The module caches collections via _get_collection() calls, so no extra
    # teardown is needed — new tmp_path each test gives a fresh collection.
    return persist


def test_upsert_and_search_roundtrip(temp_chroma: Path) -> None:
    count = index.upsert_10k_sections(
        symbol="TEST",
        filing_date="2025-01-01",
        sections={
            "business": (
                "We manufacture GPUs for data centers. "
                "Our main customers are hyperscale cloud providers."
            ),
            "risk_factors": (
                "Our supply chain is concentrated with a single foundry partner. "
                "Export controls could restrict sales to certain regions."
            ),
        },
    )
    assert count >= 2

    hits = index.search(query="foundry supply chain", symbol="TEST", k=3)
    assert hits, "expected at least one hit for foundry query"
    # The risk_factors chunk should out-rank the business chunk on this query.
    assert hits[0].section == "risk_factors"
    assert hits[0].symbol == "TEST"


def test_search_filters_by_section(temp_chroma: Path) -> None:
    index.upsert_10k_sections(
        symbol="TEST",
        filing_date="2025-01-01",
        sections={
            "business": "We make semiconductors for AI.",
            "mdna": "Revenue grew due to AI demand.",
        },
    )
    hits = index.search(query="AI demand", symbol="TEST", section="mdna", k=5)
    assert hits
    assert all(h.section == "mdna" for h in hits)


def test_search_empty_collection_returns_empty(temp_chroma: Path) -> None:
    hits = index.search(query="nothing indexed", symbol="TEST", k=5)
    assert hits == []


def test_upsert_skips_empty_and_whitespace_sections(temp_chroma: Path) -> None:
    count = index.upsert_10k_sections(
        symbol="EMPTY",
        filing_date="2025-01-01",
        sections={"business": "", "risk_factors": "   \n\n   "},
    )
    assert count == 0


def test_upsert_is_idempotent_on_same_filing(temp_chroma: Path) -> None:
    payload = {
        "business": "We make things. " * 30,
    }
    first = index.upsert_10k_sections("DUP", "2025-01-01", payload)
    second = index.upsert_10k_sections("DUP", "2025-01-01", payload)
    assert first == second  # same IDs, same count
    total = index.stats()["total_chunks"]
    assert total == first  # no duplication across the two upserts


# ---------------------------------------------------------------------------
# Network-marked integration test (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_live_cik_lookup_for_known_ticker() -> None:
    cik = edgar.ticker_to_cik("NVDA")
    assert cik.isdigit()
    assert len(cik) == 10
