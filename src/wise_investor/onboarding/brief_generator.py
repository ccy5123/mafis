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

from wise_investor.config import settings


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
    user = (
        raw.as_prompt_context()
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
    """
    symbol = symbol.upper()
    raw = RawMaterial(symbol=symbol, company_name=None, industry=None)

    # Finnhub profile + peers
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

    # Geopolitical snapshot — skip if slow or unreachable (user's
    # onboarding experience shouldn't wait on GDELT timeouts).
    try:
        from wise_investor.geopolitics.snapshot import (
            format_geopolitics_snapshot,
            get_geopolitics_snapshot,
        )

        snap = get_geopolitics_snapshot(symbol)
        raw.geo_snapshot = format_geopolitics_snapshot(snap)
    except Exception as e:
        logger.warning("geopolitics snapshot failed for %s: %s", symbol, e)

    return raw


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

    header = [
        f"# {raw.symbol} — Value Chain Brief (auto-drafted)",
        "",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source: Finnhub + SEC 10-K ({raw.edgar_filing_date or 'n/a'}) + GDELT/Google News",
        "",
        DRAFT_BANNER,
        "",
        "---",
        "",
    ]
    return "\n".join(header) + body + "\n"


def _default_llm_call(system: str, user: str) -> str:
    """Production default: Ollama Qwen 2.5 7B at temp 0, seed 42."""
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "ollama package not available — install with `pip install ollama`"
        ) from e

    resp = ollama.chat(
        model=settings.analyst_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={
            "temperature": settings.llm_temperature,
            "seed": settings.llm_seed,
        },
    )
    return resp["message"]["content"]


__all__ = [
    "DRAFT_BANNER",
    "RawMaterial",
    "build_brief_prompt",
    "gather_raw_material",
    "generate_value_chain_draft",
]
