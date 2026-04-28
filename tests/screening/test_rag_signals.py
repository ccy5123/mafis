"""Tests for screening.rag_signals — top-5 customer + diversification.

Both ChromaDB search and the LLM call are stubbed: tests pass
duck-typed callables that return canned data. No network, no embedding
download.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from wise_investor.screening.live_adapter import (
    fetch_live_fundamentals_us,
)
from wise_investor.screening.rag_signals import (
    DIVERSIFICATION_CAP,
    RagSignals,
    _parse_top5_response,
    extract_diversification_signals,
    extract_rag_signals,
    extract_top5_customer_share,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeHit:
    """Mirrors the public attributes of `rag.index.PassageHit`."""

    symbol: str
    section: str
    filing_date: str
    chunk_id: int
    text: str
    distance: float = 0.5


def _make_search(passages_by_section: dict[str, list[_FakeHit]]):
    """Returns a search_fn that returns canned hits filtered by section."""
    calls: list[dict] = []

    def _search(query, symbol=None, section=None, k=5):
        calls.append({"query": query, "symbol": symbol, "section": section, "k": k})
        if section is None:
            return []
        return list(passages_by_section.get(section, []))[:k]

    _search.calls = calls  # type: ignore[attr-defined]
    return _search


def _empty_search():
    def _search(query, symbol=None, section=None, k=5):
        return []
    return _search


def _llm_returning(response: str):
    """Helper: build a stub LLM that always returns the given string."""
    def _llm(prompt: str) -> str:
        return response
    return _llm


# ---------------------------------------------------------------------------
# extract_top5_customer_share
# ---------------------------------------------------------------------------


def test_top5_returns_none_when_no_passages_indexed() -> None:
    out, evidence = extract_top5_customer_share(
        "NVDA", search_fn=_empty_search(), llm_call=lambda p: '{"top5_share": 0.5, "evidence": "x"}',
    )
    assert out is None
    assert evidence == ""


def test_top5_extracts_from_llm_json() -> None:
    hits = [
        _FakeHit(
            symbol="NVDA", section="risk_factors", filing_date="2024-02-21",
            chunk_id=0,
            text="Our largest customer accounted for approximately 18% of our revenue in fiscal 2024...",
        ),
    ]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('{"top5_share": 0.45, "evidence": "five largest customers represented 45%"}')
    share, evidence = extract_top5_customer_share(
        "NVDA", search_fn=search, llm_call=llm,
    )
    assert share == pytest.approx(0.45)
    assert "45%" in evidence


def test_top5_returns_none_when_llm_says_null() -> None:
    """LLM correctly declines extraction when text doesn't disclose."""
    hits = [
        _FakeHit(
            symbol="MSFT", section="risk_factors", filing_date="2024-07-30",
            chunk_id=0, text="We have a broad customer base across enterprise and consumer markets.",
        ),
    ]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('{"top5_share": null, "evidence": ""}')
    share, evidence = extract_top5_customer_share(
        "MSFT", search_fn=search, llm_call=llm,
    )
    assert share is None
    assert evidence == ""


def test_top5_handles_markdown_fenced_json() -> None:
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('```json\n{"top5_share": 0.27, "evidence": "27% disclosure"}\n```')
    share, _ = extract_top5_customer_share("NVDA", search_fn=search, llm_call=llm)
    assert share == pytest.approx(0.27)


def test_top5_handles_json_with_preamble() -> None:
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('Sure, here is the extraction:\n{"top5_share": 0.32, "evidence": "concentrated"}')
    share, _ = extract_top5_customer_share("NVDA", search_fn=search, llm_call=llm)
    assert share == pytest.approx(0.32)


def test_top5_returns_none_on_malformed_llm_output() -> None:
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning("the customer share is around 30 percent")  # not JSON
    share, evidence = extract_top5_customer_share("NVDA", search_fn=search, llm_call=llm)
    assert share is None
    assert evidence == ""


def test_top5_returns_none_when_share_out_of_range() -> None:
    """LLM hallucinated a share of 5.0 (500%) — we refuse to use it."""
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('{"top5_share": 5.0, "evidence": "x"}')
    share, _ = extract_top5_customer_share("NVDA", search_fn=search, llm_call=llm)
    assert share is None


def test_top5_returns_none_when_share_negative() -> None:
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    llm = _llm_returning('{"top5_share": -0.1, "evidence": "x"}')
    share, _ = extract_top5_customer_share("NVDA", search_fn=search, llm_call=llm)
    assert share is None


def test_top5_returns_none_on_llm_exception() -> None:
    hits = [_FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text="...")]
    search = _make_search({"risk_factors": hits})
    def _err(prompt):
        raise RuntimeError("LLM offline")
    share, _ = extract_top5_customer_share("NVDA", search_fn=search, llm_call=_err)
    assert share is None


def test_top5_dedupes_overlapping_passages() -> None:
    """Same passage returned by multiple queries shouldn't bloat the prompt."""
    duplicate_text = "Our largest customer accounted for 25% of revenue in fiscal 2024."
    hits = [
        _FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text=duplicate_text),
        _FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0, text=duplicate_text),
        _FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=1, text="Different chunk text here."),
    ]
    search = _make_search({"risk_factors": hits})
    captured_prompt: list[str] = []
    def _capture(prompt):
        captured_prompt.append(prompt)
        return '{"top5_share": 0.25, "evidence": "x"}'
    extract_top5_customer_share("NVDA", search_fn=search, llm_call=_capture)
    # The prompt should contain duplicate_text exactly once
    assert captured_prompt[0].count(duplicate_text) == 1


def test_top5_search_failure_silently_continues() -> None:
    """One query failing shouldn't prevent the others from contributing."""
    counter = {"n": 0}
    def _flaky_search(query, symbol=None, section=None, k=5):
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("Chroma query timeout")
        return [
            _FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0,
                     text="Our top 5 customers accounted for 30%."),
        ]
    llm = _llm_returning('{"top5_share": 0.30, "evidence": "30%"}')
    share, _ = extract_top5_customer_share(
        "NVDA", search_fn=_flaky_search, llm_call=llm,
    )
    assert share == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# extract_diversification_signals
# ---------------------------------------------------------------------------


def test_diversif_returns_zero_when_no_passages() -> None:
    assert extract_diversification_signals("NVDA", search_fn=_empty_search()) == 0


def test_diversif_counts_keyword_matches() -> None:
    hits = [
        _FakeHit(
            symbol="NVDA", section="business", filing_date="2024-02-21", chunk_id=0,
            text=(
                "We entered the new market for autonomous vehicles. "
                "We launched a new product line for data center inference. "
                "We acquired a new company in the gaming software space."
            ),
        ),
    ]
    out = extract_diversification_signals("NVDA", search_fn=_make_search({"business": hits}))
    assert out >= 3  # entered, launched, acquired


def test_diversif_capped_at_max() -> None:
    """Lots of keyword matches → cap at DIVERSIFICATION_CAP, not unbounded."""
    text = (
        "We entered the new market. We launched a new product. We acquired a new business. "
        "We expanded into a new market. We have a new business segment. "
        "We diversify into adjacent markets. "
        "We entered the new business. We launched a new segment."
    )
    hits = [_FakeHit(symbol="X", section="business", filing_date="2024", chunk_id=0, text=text)]
    out = extract_diversification_signals("X", search_fn=_make_search({"business": hits}))
    assert out <= DIVERSIFICATION_CAP


def test_diversif_zero_when_no_keywords() -> None:
    hits = [
        _FakeHit(
            symbol="NVDA", section="business", filing_date="2024-02-21", chunk_id=0,
            text="We continue to focus on our core graphics processing unit business.",
        ),
    ]
    out = extract_diversification_signals("NVDA", search_fn=_make_search({"business": hits}))
    assert out == 0


def test_diversif_dedupes_passages_before_counting() -> None:
    """Same passage from multiple queries shouldn't double-count keywords."""
    text = "We entered the new market for cloud computing services."
    hits = [
        _FakeHit(symbol="X", section="business", filing_date="2024", chunk_id=0, text=text),
        _FakeHit(symbol="X", section="business", filing_date="2024", chunk_id=0, text=text),
    ]
    out = extract_diversification_signals("X", search_fn=_make_search({"business": hits}))
    assert out == 1  # one keyword pattern matched, exactly once


# ---------------------------------------------------------------------------
# extract_rag_signals (combined)
# ---------------------------------------------------------------------------


def test_combined_extract_returns_rag_signals_dataclass() -> None:
    rf_hits = [
        _FakeHit(symbol="NVDA", section="risk_factors", filing_date="2024-02-21", chunk_id=0,
                 text="Our 5 largest customers accounted for 40% of revenue."),
    ]
    bs_hits = [
        _FakeHit(symbol="NVDA", section="business", filing_date="2024-02-21", chunk_id=0,
                 text="We entered the new market for AI accelerators in 2024."),
    ]
    search = _make_search({"risk_factors": rf_hits, "business": bs_hits})
    llm = _llm_returning('{"top5_share": 0.40, "evidence": "40% disclosure"}')
    out = extract_rag_signals("NVDA", search_fn=search, llm_call=llm)
    assert isinstance(out, RagSignals)
    assert out.top5_customer_share == pytest.approx(0.40)
    assert out.diversification_attempt_signals >= 1
    assert "40%" in out.top5_evidence


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def test_parse_top5_response_handles_plain_json() -> None:
    out = _parse_top5_response('{"top5_share": 0.25, "evidence": "x"}')
    assert out == {"top5_share": 0.25, "evidence": "x"}


def test_parse_top5_response_handles_fenced_json() -> None:
    raw = '```json\n{"top5_share": 0.30}\n```'
    out = _parse_top5_response(raw)
    assert out == {"top5_share": 0.30}


def test_parse_top5_response_handles_unfenced_with_preamble() -> None:
    raw = 'I extracted: {"top5_share": 0.50, "evidence": ""} that is the answer.'
    out = _parse_top5_response(raw)
    assert out == {"top5_share": 0.50, "evidence": ""}


def test_parse_top5_response_returns_none_on_garbage() -> None:
    assert _parse_top5_response("the answer is 30%") is None
    assert _parse_top5_response("") is None
    assert _parse_top5_response("{ this is not json }") is None


# ---------------------------------------------------------------------------
# Integration: live adapter accepts and applies RagSignals
# ---------------------------------------------------------------------------


def test_live_us_adapter_consumes_rag_signals() -> None:
    """When rag_signals is supplied, the adapter populates the two
    bottleneck axis fields from it instead of the legacy None/0 defaults.
    """
    # Reuse the Finnhub stub style from test_live_adapter.py (duck-typed).
    class _Item:
        def __init__(self, concept, value):
            self.concept = concept
            self.value = value

    class _Report:
        def __init__(self, ic=(), bs=()):
            self.ic = list(ic)
            self.bs = list(bs)
            self.cf = []

    class _Entry:
        def __init__(self, year):
            self.year = year
            self.quarter = None
            self.form = "10-K"
            self.report = _Report(
                ic=[_Item("us-gaap_OperatingIncomeLoss", 200)],
                bs=[_Item("us-gaap_StockholdersEquity", 500)],
            )

    class _Resp:
        def __init__(self, data): self.data = data

    class _Profile:
        finnhub_industry = "Test"

    class _Client:
        def financials(self, sym, freq="annual"):
            return _Resp([_Entry(2024)] if freq == "annual" else [])
        def profile(self, sym):
            return _Profile()

    rag = RagSignals(
        top5_customer_share=0.42,
        diversification_attempt_signals=2,
        top5_evidence="42% concentration in top 5",
    )
    funds = fetch_live_fundamentals_us(
        "TEST", client=_Client(), rag_signals=rag,
    )
    assert funds.top5_customer_share == pytest.approx(0.42)
    assert funds.diversification_attempt_signals == 2


def test_live_us_adapter_defaults_when_rag_signals_none() -> None:
    """No rag_signals → legacy None/0 defaults preserved."""
    class _Item:
        def __init__(self, concept, value):
            self.concept = concept
            self.value = value

    class _Report:
        def __init__(self, ic=(), bs=()):
            self.ic = list(ic)
            self.bs = list(bs)
            self.cf = []

    class _Entry:
        def __init__(self, year):
            self.year = year
            self.quarter = None
            self.form = "10-K"
            self.report = _Report(
                ic=[_Item("us-gaap_OperatingIncomeLoss", 200)],
                bs=[_Item("us-gaap_StockholdersEquity", 500)],
            )

    class _Resp:
        def __init__(self, data): self.data = data

    class _Profile:
        finnhub_industry = "Test"

    class _Client:
        def financials(self, sym, freq="annual"):
            return _Resp([_Entry(2024)] if freq == "annual" else [])
        def profile(self, sym):
            return _Profile()

    funds = fetch_live_fundamentals_us("TEST", client=_Client())
    assert funds.top5_customer_share is None
    assert funds.diversification_attempt_signals == 0
