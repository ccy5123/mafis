"""Auto-generate a value chain brief draft from Finnhub + 10-K + geo data.

Pipeline:

    ticker
     │
     ├── FinnhubClient.profile       → company name, industry
     ├── FinnhubClient.peers         → seed peer list
     ├── rag.integration.ensure_10k_indexed + gather_section_passages
     │                               → Business + Risk Factors excerpts
     └── geopolitics.get_geopolitics_snapshot (best-effort)
                                     → recent news context

    └──→ render_prompt(raw)          → structured Ollama prompt
    └──→ ollama.chat(qwen 2.5 7B)    → markdown draft
    └──→ post-process + banner       → <SYMBOL>.draft.md body

The draft preserves the canonical 8-heading structure from
docs/value_chains/README.md (Peer Override, Upstream — Suppliers, Peers,
Downstream — Customers, Infrastructure dependencies, Geopolitical /
regulatory pressure points, Vulnerable links, Known unknowns). Human
review is expected particularly on Vulnerable Links — we prompt the
LLM to mark uncertain entries with a `[?UNCERTAIN]` flag.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from typing import Any, Callable


logger = logging.getLogger(__name__)


DRAFT_BANNER = (
    "> ⚠ AUTO-GENERATED DRAFT — reviewed by the onboarding pipeline, "
    "NOT by a human yet. Before renaming to <SYMBOL>.md, verify the "
    "**Vulnerable links** section in particular: auto-drafted attack "
    "vectors are the highest-risk hallucination surface. Entries "
    "prefixed with `[?UNCERTAIN]` are the LLM flagging its own low "
    "confidence — resolve or delete them."
)


@dataclass
class RawMaterial:
    """Everything the LLM sees when drafting a value chain brief."""

    symbol: str
    company_name: str | None
    industry: str | None
    peers: list[str] = field(default_factory=list)
    edgar_business_excerpt: str = ""
    edgar_risk_factors_excerpt: str = ""
    edgar_moat_excerpt: str = ""
    edgar_filing_date: str | None = None
    geo_snapshot: str = ""

    def as_prompt_context(self) -> str:
        """Render raw inputs as one structured prompt body."""
        lines = [
            f"# Symbol: {self.symbol}",
            f"- Company name: {self.company_name or '(unknown)'}",
            f"- Industry: {self.industry or '(unknown)'}",
            f"- Finnhub peer seed: {', '.join(self.peers) if self.peers else '(none returned)'}",
            f"- 10-K filing date: {self.edgar_filing_date or '(no 10-K indexed)'}",
        ]
        if self.edgar_business_excerpt:
            lines.append("\n## 10-K Business excerpt (verbatim)")
            lines.append(self.edgar_business_excerpt)
        if self.edgar_risk_factors_excerpt:
            lines.append("\n## 10-K Risk Factors excerpt (verbatim)")
            lines.append(self.edgar_risk_factors_excerpt)
        if self.edgar_moat_excerpt:
            lines.append("\n## 10-K Moat signals excerpt (verbatim)")
            lines.append(self.edgar_moat_excerpt)
        if self.geo_snapshot:
            lines.append("\n## Recent geopolitical / news snapshot")
            lines.append(self.geo_snapshot)
        return "\n".join(lines)


BRIEF_TEMPLATE = """\
You are generating a value chain brief for {symbol}. The output goes
directly into the agent crew's prompt context, so be dense, factual,
and biased toward "dependencies and vulnerabilities" rather than
marketing narrative.

The brief MUST follow this exact 8-heading structure, in this order,
with these exact H2 headings:

## Peer Override
One bullet list. Use `- (none)` if the Finnhub peer seed above looks
credible for valuation comparison. Otherwise list 3-5 tickers that
the human should manually add. If you add any, brief rationale per
entry.

## Upstream — Suppliers
Sub-sections with H3 headings (`### Chip fabrication`,
`### Memory`, `### Design tooling`, etc.) when the inputs support
distinct layers. Each supplier entry: bold name + concise role + one
quantified concentration/ fragility note when the inputs give one.
Cite the 10-K when you quote directly (e.g.
"per 10-K Business, filed {filing_date}").

## Peers — Direct competition
Markdown table with columns: Peer | Ticker | Product | Threat level.
Use the Finnhub peer seed + any explicit competitors named in the
10-K Business excerpt. If the threat level can't be inferred, use
`Medium` + a one-phrase justification.

## Downstream — Customers
Revenue concentration if the 10-K mentions it, followed by
customer categories (one H3 per category). Prefer direct 10-K
phrasing over invented breakdowns.

## Infrastructure dependencies
Anything the target company relies on that is not a direct supplier
or customer: regulatory regimes, utilities, standard bodies, export
control frameworks. Use the Risk Factors excerpt.

## Geopolitical / regulatory pressure points
Sovereign-level risks drawn from the Risk Factors excerpt AND the
geopolitical news snapshot. Each bullet should name one specific
risk + the likely first-order impact on revenue or operations.

## Vulnerable links (Skeptic's attack surface)
This section is the Skeptic agent's primary fuel. Exactly 5-7
numbered entries. Each entry is one sentence describing a concrete
failure mode AND the magnitude of its first-order impact (e.g.
"one quarter of revenue lost" or "margin compression of several
hundred bps"). If the inputs don't support a confident entry,
prefix the entry with `[?UNCERTAIN]` so the human reviewer knows
to verify. Do NOT invent attack vectors; if the inputs are thin,
say so and produce fewer than 7.

## Known unknowns (do not pretend to know)
Bullet list of questions the Analyst agent must NOT try to answer.
Each bullet is one question the inputs do not cover (e.g.
"Precise revenue share by customer").

=== Hard rules ===

- No numbers that do not appear verbatim in the inputs. If you want
  to state "40% of revenue from top-4 customers" but the excerpts
  don't mention that figure, either find a substitute quote or
  rephrase as "revenue concentration in a small number of hyperscale
  customers (per 10-K Business, {filing_date})".
- Cite the 10-K when you quote: `(per 10-K Business, filed {filing_date})`
  or `(per 10-K Risk Factors, filed {filing_date})`.
- Never quote the Finnhub peer list as "top competitors" — only that
  they are the "Finnhub peer cohort".
- Output only the markdown brief. No preamble, no closing, no
  apology for uncertainty. The `[?UNCERTAIN]` prefix is how you flag
  uncertainty in Vulnerable links.
"""


def build_brief_prompt(raw: RawMaterial) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the brief-generator LLM call."""
    system = (
        "You are a research associate drafting a value chain brief that will "
        "feed a multi-agent investment analysis crew. Be dense, precise, and "
        "biased toward dependencies and vulnerabilities. Output only the "
        "brief markdown — no preamble, no commentary."
    )

    korean_note = ""
    if (raw.edgar_filing_date or "").startswith("(DART"):
        korean_note = (
            "\n\n=== Korean-listing caveat ===\n\n"
            "This ticker is a KRX-listed company. The inputs above are "
            "financial aggregates from DART's fnlttSinglAcntAll endpoint — "
            "NOT a Business / Risk-Factors narrative like a US 10-K "
            "provides. Implications for YOUR draft:\n"
            "  - Upstream suppliers: DO NOT list suppliers unless they "
            "appear by NAME in the inputs. If none are named, use a "
            "single bullet '(not derivable from DART financials — human "
            "reviewer must fill)'.\n"
            "  - Peers: DART does NOT provide a peer list. Treat any "
            "peer name you consider emitting as requiring a "
            "`[?UNCERTAIN]` prefix — the human will validate the peer "
            "cohort manually.\n"
            "  - Do NOT claim the target depends on TSMC / Intel / other "
            "US-scene companies purely from pretraining memory. If the "
            "relationship is not in the inputs, omit it.\n"
            "  - Known unknowns section SHOULD include 'Precise supplier "
            "list (DART financial statements do not enumerate upstream "
            "vendors)' as an explicit bullet."
        )

    user = (
        raw.as_prompt_context()
        + korean_note
        + "\n\n=== Brief template + rules ===\n\n"
        + BRIEF_TEMPLATE.format(
            symbol=raw.symbol,
            filing_date=raw.edgar_filing_date or "(unknown)",
        )
    )
    return (system, user)


def gather_raw_material(symbol: str) -> RawMaterial:
    """Collect every data input the brief generator needs.

    Fails soft per source — each empty / error source is recorded as an
    empty field on RawMaterial so the LLM can still draft a reduced
    brief and the human reviewer can tell what was missing.

    Dispatches on symbol shape: Korean tickers (6-digit KRX codes)
    pull from OpenDART; everything else from Finnhub + SEC EDGAR.
    """
    symbol = symbol.upper()
    raw = RawMaterial(symbol=symbol, company_name=None, industry=None)

    # Korean-ticker fast path: DART for company profile + a compressed
    # view of the annual financials as a Business "excerpt" substitute.
    # SEC EDGAR obviously has nothing for KRX companies.
    from wise_investor.data.dart_facts import is_korean_ticker
    if is_korean_ticker(symbol):
        try:
            from wise_investor.data.dart import DartClient
            from wise_investor.data.dart_facts import (
                normalize_korean_symbol,
                pre_gather_dart_facts,
            )

            with DartClient() as client:
                stock_code = normalize_korean_symbol(symbol)
                corp_code = client.corp_code_from_stock_code(stock_code)
                if corp_code:
                    mappings = client.load_corp_mapping()
                    entry = next(
                        (m for m in mappings if m.stock_code == stock_code),
                        None,
                    )
                    raw.company_name = entry.corp_name if entry else None
                    raw.industry = "(Korean listing; DART does not classify)"

            # Use the same DART facts adapter the crew pre-gather
            # uses — render each account value as a line. This gives
            # the LLM enough numerical context to draft the brief.
            dart_facts = pre_gather_dart_facts(stock_code)
            financial_lines = "\n".join(
                f"- {key}: {value}"
                for key, value in dart_facts.items()
                if key.startswith("dart.") and not value.startswith("ERROR")
            )
            if financial_lines:
                raw.edgar_filing_date = "(DART annual filing)"
                raw.edgar_business_excerpt = (
                    "[DART-sourced; SEC 10-K unavailable for Korean listings]\n"
                    + financial_lines
                )
        except Exception as e:
            logger.warning("DART gather failed for %s: %s", symbol, e)

        # Peers: DART doesn't expose a peer list. Leave empty; the
        # LLM will be instructed to mark them [?UNCERTAIN] and the
        # human reviewer adds a Peer Override list.
        _attach_geopolitics(raw, symbol)
        return raw

    # US path — original Finnhub + SEC EDGAR flow.
    try:
        from wise_investor.data.finnhub import FinnhubClient

        with FinnhubClient() as client:
            profile = client.profile(symbol)
            raw.company_name = profile.name
            raw.industry = profile.finnhub_industry
            peers = client.peers(symbol)
            # strip self-reference
            raw.peers = [p for p in peers if p.upper() != symbol][:8]
    except Exception as e:
        logger.warning("Finnhub profile/peers failed for %s: %s", symbol, e)

    # 10-K RAG
    try:
        from wise_investor.rag.integration import (
            ensure_10k_indexed,
            gather_section_passages,
        )

        ref = ensure_10k_indexed(symbol)
        if ref is not None:
            raw.edgar_filing_date = ref.filing_date
            sections = gather_section_passages(
                symbol,
                queries={
                    "business": (
                        "business segments products customers suppliers "
                        "manufacturing competition"
                    ),
                    "risk_factors": (
                        "supply chain concentration regulatory export "
                        "controls geopolitical critical dependence"
                    ),
                    "moat": (
                        "competitive advantages intellectual property "
                        "switching costs ecosystem market share"
                    ),
                },
                k=3,
            )
            raw.edgar_business_excerpt = _passages_to_block(sections.get("business"))
            raw.edgar_risk_factors_excerpt = _passages_to_block(
                sections.get("risk_factors")
            )
            raw.edgar_moat_excerpt = _passages_to_block(sections.get("moat"))
    except Exception as e:
        logger.warning("10-K RAG gather failed for %s: %s", symbol, e)

    _attach_geopolitics(raw, symbol)
    return raw


def _attach_geopolitics(raw: RawMaterial, symbol: str) -> None:
    """Best-effort geopolitical snapshot. Failure (e.g., GDELT timeout)
    leaves raw.geo_snapshot empty rather than aborting onboarding.
    """
    try:
        from wise_investor.geopolitics.snapshot import (
            format_geopolitics_snapshot,
            get_geopolitics_snapshot,
        )

        snap = get_geopolitics_snapshot(symbol)
        raw.geo_snapshot = format_geopolitics_snapshot(snap)
    except Exception as e:
        logger.warning("geopolitics snapshot failed for %s: %s", symbol, e)


def _passages_to_block(section: Any) -> str:
    """Compact a SectionPassages object into a short, quotable block."""
    if section is None:
        return ""
    passages = getattr(section, "passages", [])
    if not passages:
        return ""
    lines: list[str] = []
    for i, p in enumerate(passages, start=1):
        text = getattr(p, "text", "").strip()
        if len(text) > 800:
            text = text[:800].rstrip() + " ..."
        lines.append(f"Passage {i} (distance={getattr(p, 'distance', 0):.3f}):\n{text}")
    return "\n\n".join(lines)


def generate_value_chain_draft(
    symbol: str,
    raw: RawMaterial | None = None,
    llm_call: Callable[[str, str], str] | None = None,
) -> str:
    """End-to-end: gather raw → render prompt → call LLM → wrap with banner.

    `llm_call` is injectable so tests can skip the Ollama call. Signature:
    `llm_call(system_prompt, user_prompt) -> markdown body`.
    """
    if raw is None:
        raw = gather_raw_material(symbol)
    system, user = build_brief_prompt(raw)

    if llm_call is None:
        llm_call = _default_llm_call

    body = llm_call(system, user).strip()

    # Source line adapts to the data path actually used. Korean tickers
    # don't have a SEC 10-K; the filing_date field stores a placeholder
    # like "(DART annual filing)" which should be rendered verbatim.
    source_bits: list[str] = []
    if raw.edgar_filing_date and raw.edgar_filing_date.startswith("(DART"):
        source_bits.append(f"DART {raw.edgar_filing_date}")
    elif raw.edgar_filing_date:
        source_bits.append(f"SEC 10-K filed {raw.edgar_filing_date}")
    if raw.company_name is not None:
        source_bits.append("Finnhub/DART profile")
    source_bits.append("GDELT + Google News")
    source_line = "Source: " + " + ".join(source_bits)

    header = [
        f"# {raw.symbol} — Value Chain Brief (auto-drafted)",
        "",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        source_line,
        "",
        DRAFT_BANNER,
        "",
        "---",
        "",
    ]
    return "\n".join(header) + body + "\n"


def _default_llm_call(system: str, user: str) -> str:
    """Production default: routed through the active LLMBackend.

    Resolves to `agents.brief_generator` from agent_models.yaml; if
    that's unspecified the loader's legacy fallback gives us the
    Analyst model — same model historically used by this generator.
    """
    from wise_investor.llm import get_agent_config, get_backend

    backend = get_backend()
    cfg = get_agent_config("brief_generator", backend=backend.name)
    response = backend.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=cfg.model,
        sampling=cfg.sampling,
    )
    return response.content


__all__ = [
    "DRAFT_BANNER",
    "RawMaterial",
    "build_brief_prompt",
    "gather_raw_material",
    "generate_value_chain_draft",
]
