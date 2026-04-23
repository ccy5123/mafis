"""CrewAI Task factories for Phase 1B.

Phase 1B ships a single Task: `make_analyst_task`. It embeds the manual value
chain document into the prompt context (prompt injection, not RAG — see
design-v2.2 re-review High #5) and imposes a rigid seven-section output
structure so reports are comparable across iterations and across tickers.
"""

from __future__ import annotations

from pathlib import Path

from crewai import Agent, Task


REPO_ROOT = Path(__file__).resolve().parents[3]
VALUE_CHAINS_DIR = REPO_ROOT / "docs" / "value_chains"


# ---------------------------------------------------------------------------
# Analyst task template (Phase 1B)
# ---------------------------------------------------------------------------


REPORT_TEMPLATE = """\
You are producing an equity research note on {symbol}. The note MUST have these
seven sections, in this order, with these exact H2 headings.

## 1. Business Summary
Two to three short paragraphs: what the company sells, to whom, how it makes
money, recent strategic direction. Facts only; no valuation comments.

**10-K GROUNDING (MANDATORY):** At least ONE sentence in this section MUST
quote or paraphrase a claim from the `edgar.business_segments` tool_output
and end with its `[Cite as: ...]` hint copied verbatim. This forces the
Business Summary to reflect what the company actually says about itself
in its latest 10-K, not what the model remembers from pretraining.

## 2. Value Chain Context
Summarize the upstream, peer, and downstream relationships that matter for
durability. Cite the value chain brief when you use its claims, using the
phrase "per the value chain brief" so the reader can trace it.

**10-K GROUNDING (MANDATORY):** When you name a specific supplier, customer,
or competitor that also appears in the `edgar.*` excerpts, add the 10-K
citation alongside the value-chain citation, e.g. "[per the value chain
brief; Source: 10-K business, filed 2026-02-25]". This lets the Skeptic
distinguish claims grounded in the filing from claims we curated manually.

## 3. Financial Health
Every numeric line in this section MUST follow this exact format:
- **<Metric name>**: $<value> — [Source: <tool_output name>]

Draw values only from <tool_output> blocks in <pre_gathered_tool_outputs>.
Include at minimum: revenue, net_income, operating_income, gross_profit,
ebitda, free_cash_flow, total_debt. Also state current PER and EV/EBITDA
from the calculate_per and calculate_ev_ebitda outputs.

## 4. Competitive Position / Moat
Five-to-ten-year durability analysis. Structural advantages (scale,
ecosystem, switching cost, IP, regulatory) vs erosion forces.

**10-K GROUNDING (MANDATORY):** The structural-advantage paragraph MUST
cite at least one claim from `edgar.moat_signals` with its verbatim
`[Cite as: ...]` hint. If the edgar.moat_signals block is an ERROR or
empty, write "no 10-K passage available" in place of the citation and
omit the claim from the moat narrative.

Then quote the peer multiples table verbatim inside a fenced code block,
and below it write two short paragraphs (OUTSIDE the code block):
1. Interpreting the target's relative positioning versus peers. EVERY
   line in these paragraphs that mentions a specific peer PER or
   EV/EBITDA number MUST end with [Source: get_peer_multiples].
2. Naming the peer(s) that pose the most credible 5-year threat and
   why. If you cite a number here, end the line with [Source: ...].

## 5. Valuation Context (brief)
Two bullet lines only. EACH LINE MUST end with a [Source: ...] citation
listing every tool whose number is used on that line:
- **Current multiples vs peers**: state the target's PER and EV/EBITDA,
  then compare to the numerically highest and lowest peer from the peer
  multiples table (name both peers and their numbers). One sentence
  interpretation. End the line with
  [Source: calculate_per, calculate_ev_ebitda, get_peer_multiples].
- **Market-implied growth assumption**: quote the implied FCF growth
  rate from reverse_dcf, then one sentence comparing it to a plausible
  historical benchmark. End the line with [Source: reverse_dcf].

Do NOT issue a buy/sell/hold recommendation — that is the Steward agent's
responsibility in a later phase.

## 6. Data Gaps and Warnings
This section has two subsections.

**Tool warnings (verbatim):** Iterate through every <tool_output> in
<pre_gathered_tool_outputs>. If its body contains a line starting with
"Warnings:" followed by anything other than "none", write a bullet in
this exact format:
- `<tool name>`: <the verbatim text that follows "Warnings:" including any
  indented "- ..." sub-bullets>

If a tool_output says "Warnings: none" you may omit it from the bullet
list. Do NOT paraphrase warnings; copy them character-for-character.
If a warning contains a number (e.g. an implied growth rate), end that
bullet with [Source: <tool_name>].

**Known unknowns (from value chain brief):** Copy the bullet list under
the "Known unknowns (do not pretend to know)" heading of the value chain
brief verbatim into this subsection. These are qualitative — no numbers,
no citations required.

## 7. Questions for Skeptic
Exactly 5 questions, numbered 1 through 5. Each question MUST have these
three labeled fields in this order, each field one sentence:

1. **Claim**: <the specific claim elsewhere in this report that is under attack>
   - **Assumption**: <the assumption the claim rests on>
   - **Evidence that would falsify it**: <concrete evidence or event that would refute it>

If the **Claim** line contains any number, end that line with a
[Source: <tool_name>] citation — the Universal Citation Rule applies
here too.

Preferred sources for claims to attack: the reverse-DCF implied growth
rate, the premium multiple vs peers, and at least one vulnerable link
named in the value chain brief.

Output the full markdown report. Do not wrap it in a code fence. Do not
include a preamble, table of contents, or closing remark outside the seven
sections.
"""


CONTEXT_INSTRUCTIONS = """\
You are writing an equity research note on {symbol}. All six Phase 1A
calculation tools have already been executed by Python and their outputs are
attached above in <pre_gathered_tool_outputs>. Your job is to compose the
seven-section markdown report.

The authoritative qualitative context — upstream/peer/downstream map, known
vulnerable links, and explicit "known unknowns" — is in the value chain brief
below. Treat it as source material. When you use a claim from it, cite with
the phrase "per the value chain brief".

<value_chain_brief>
{value_chain}
</value_chain_brief>
"""


def _load_value_chain(symbol: str) -> str:
    path = VALUE_CHAINS_DIR / f"{symbol.upper()}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"No value chain document for {symbol}. Create {path} before running "
            f"the Analyst (design-v2.2 §5.1 Phase 1 — manual value chain required)."
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Valuer task template (Phase 1C) — consumes facts + Analyst output
# ---------------------------------------------------------------------------


VALUER_REPORT_TEMPLATE = """\
You are producing the valuation section of an equity research note on {symbol}.
This section MUST have these three H2 headings, in this order, with no other
sections:

## Valuation Snapshot
Three short bullets. EVERY bullet MUST end with a [Source: ...] citation
naming every tool whose number appears on that line:
- **Current PER**: <value> vs peers (one sentence naming highest and
  lowest peer with their numbers).
  [Source: calculate_per, get_peer_multiples]
- **Current EV/EBITDA**: <value> vs peers (same structure).
  [Source: calculate_ev_ebitda, get_peer_multiples]
- **Enterprise Value**: <value> — one sentence on what the Analyst's
  financial-health picture (revenue scale, FCF, debt) implies about this
  EV. End the line with [Source: calculate_ev_ebitda] plus any Analyst
  sources whose numbers you paraphrase.

## Peer Context
Quote the `get_peer_multiples` table VERBATIM inside a fenced code block
(no edits, no summarization). The fenced table itself does NOT need
per-row citations. Then write two short paragraphs OUTSIDE the code
block; every line in those paragraphs that contains a peer number MUST
end with [Source: get_peer_multiples]:
1. Where the target sits in the peer distribution (quartile position, by
   symbol name, no approximations).
2. Which specific peer's multiple is the most informative benchmark for
   this target, and why that peer matches best.

## Market-Implied Growth Assessment
Three short paragraphs. EVERY line containing a numeric value MUST end
with a [Source: ...] citation:
1. Quote the reverse-DCF implied annual FCF growth rate
   [Source: reverse_dcf], and quote the three assumption parameters
   (discount_rate, terminal_growth, high_growth_years) on the same line
   or as a sub-bullet ending with [Source: reverse_dcf].
2. Compare that implied rate to the target's historical growth trend.
   Use only numbers present in <pre_gathered_tool_outputs> or in the
   Analyst's text (cite the Analyst source accordingly). If neither is
   available, say "historical comparison unavailable from current tool
   outputs" — do not invent benchmarks.
3. Surface all tool-output Warnings VERBATIM here. Any line starting
   with "Warnings:" in a <tool_output> that is not "Warnings: none" must
   appear character-for-character in this subsection, prefixed by the
   tool name.

Do NOT produce a buy/sell/hold recommendation. Do NOT write an overall
valuation verdict like "overvalued" / "undervalued" / "fairly valued".
Your role is to render the numbers and their relationships; judgment
belongs to the reader and the future Steward agent.
"""


def make_valuer_user_prompt(
    symbol: str, value_chain_text: str, analyst_output: str
) -> str:
    """Build the Valuer's user-prompt content (facts block is injected separately
    by the runner, same pattern as Analyst).

    Embeds both the value chain brief (for qualitative peer context) and the
    Analyst's full output (for moat-grounded valuation judgment).
    """
    symbol = symbol.upper()
    return (
        f"You are writing the Valuer section of the research note on {symbol}.\n\n"
        "The Analyst has already produced the business and financial sections "
        "of this note. You will build on their work. Here is the Analyst's "
        "output verbatim:\n\n"
        "<analyst_section>\n"
        f"{analyst_output}\n"
        "</analyst_section>\n\n"
        "The authoritative qualitative context (upstream/peer/downstream map "
        "and known vulnerable links) is here:\n\n"
        "<value_chain_brief>\n"
        f"{value_chain_text}\n"
        "</value_chain_brief>\n\n"
        + VALUER_REPORT_TEMPLATE.format(symbol=symbol)
    )


# ---------------------------------------------------------------------------
# Skeptic task template (Phase 1C) — consumes facts + Analyst + Valuer
# ---------------------------------------------------------------------------


SKEPTIC_REPORT_TEMPLATE = """\
You are producing the Skeptic (red-team) section of the equity research
note on {symbol}. This section MUST have these three H2 headings, in this
order:

## Attack on the Bull Thesis
Exactly 5 rebuttals, numbered 1 through 5. Each targets a specific claim
from the Analyst or Valuer section above. Each must follow this exact
structure:

1. **Target claim (Analyst|Valuer)**: <quote or close paraphrase of the
   exact sentence you are attacking; name which agent stated it>. If
   this claim line contains a specific number (PER, growth rate, dollar
   amount), end the line with [Source: <tool_name>] — the Universal
   Citation Rule applies here too.
   - **Assumption under attack**: <the implicit assumption the claim rests on>.
   - **Counter-evidence / scenario**: <concrete, falsifiable event or
     measurement that would invalidate the claim; prefer items from the
     vulnerable-links section of the value chain brief, referenced by
     numbered position (e.g. "Vulnerable link #2")>.
   - **Downside quantification**: <ONE of the following two options — NEVER
     a made-up number>:
       (a) A specific dollar or percentage figure that appears VERBATIM in
           the `<pre_gathered_tool_outputs>` block, followed by
           [Source: <tool_name>]. Example: "Total debt of $7.47B
           [Source: fetch.total_debt] would become a larger burden if FCF
           compresses."
       (b) The exact refusal phrase: "Downside not quantifiable from
           current facts." — followed by one sentence naming which number
           would be needed (e.g. "a peer-median multiple benchmark would
           be needed; it is not in the facts block").
     Do NOT invent figures like "$20B stock compression" or "10% YoY
     revenue decline" unless that exact number is already in the facts
     block or value chain brief.

At least 3 of the 5 rebuttals MUST ground in a specific entry from the
value chain brief's "Vulnerable links" section. Name the entry by its
numbered position or bolded title.

**10-K GROUNDING (MANDATORY):** At least ONE of the 5 rebuttals MUST
quote a risk from the `edgar.risk_factors` tool_output block and end
the Counter-evidence/scenario line with the verbatim `[Cite as: ...]`
hint from that passage. This forces the Skeptic to attack on
risks the filing itself discloses, not just on value-chain hypotheses.
If `edgar.risk_factors` is an ERROR entry, note that in a line and
skip the 10-K-grounded rebuttal.

=== Check-then-write protocol ===

Before writing each Downside quantification field, STOP. Scan the
<pre_gathered_tool_outputs> block. Is there a number that directly
supports your claim? If YES, quote it with [Source: ...]. If NO, use the
refusal phrase verbatim. Do not write a plausible-sounding number "to
give the reader a sense of scale" — that is the failure mode this
protocol exists to prevent.

## Reverse-DCF Stress Test
Quote the reverse_dcf implied growth rate verbatim with [Source: reverse_dcf].
Then answer these two questions. The ONLY correct answers are either (i) a
figure cited from the facts block, or (ii) the exact refusal phrase given
below. Invented answers are treated as fabricated evidence.

- Q1: Has any company in this industry historically sustained that FCF
  growth rate over the window the DCF assumes? **Either** name a specific
  company AND its actual historical rate IF that exact number is present
  in <pre_gathered_tool_outputs> or <value_chain_brief>, **or** write
  verbatim: "Unknown from current facts — I cannot name one without
  inventing a benchmark."

- Q2: What % decline in the implied growth rate would compress the stock
  to a peer-median multiple? **Either** compute it IF all required
  numbers (peer-median multiple, target current multiple) appear in the
  facts block, citing each, **or** write verbatim: "Not computable from
  current facts — no peer-median multiple is provided."

Do NOT estimate. Do NOT produce a round-number guess. The refusal phrases
are the correct answer when inputs are absent.

## Steelman Concession
One paragraph, max 4 sentences. Acknowledge the single strongest Bull
argument that survives all five rebuttals above. Do not soften the
rebuttals — this section exists so the reader knows what part of the
thesis you could NOT attack.

Do NOT write a conclusion, a summary, a recommendation, or a "balanced
view" paragraph. Your job is attack + single concession. Nothing else.
"""


def make_skeptic_user_prompt(
    symbol: str,
    value_chain_text: str,
    analyst_output: str,
    valuer_output: str,
) -> str:
    """Build the Skeptic's user-prompt content.

    Gives Skeptic the full Bull thesis (Analyst + Valuer) plus the value chain
    brief's vulnerable-links section as structured attack material.
    """
    symbol = symbol.upper()
    return (
        f"You are writing the Skeptic section of the research note on {symbol}.\n\n"
        "The Analyst and Valuer have produced the Bull thesis below. Your "
        "job is to attack it. Read both sections carefully for specific "
        "claims to rebut.\n\n"
        "<analyst_section>\n"
        f"{analyst_output}\n"
        "</analyst_section>\n\n"
        "<valuer_section>\n"
        f"{valuer_output}\n"
        "</valuer_section>\n\n"
        "Your most useful attack material is the 'Vulnerable links' section "
        "of this value chain brief. It was curated specifically for your "
        "role.\n\n"
        "<value_chain_brief>\n"
        f"{value_chain_text}\n"
        "</value_chain_brief>\n\n"
        + SKEPTIC_REPORT_TEMPLATE.format(symbol=symbol)
    )


# ---------------------------------------------------------------------------
# Economist task template (Phase 2) — consumes FRED macro snapshot + vc brief
# ---------------------------------------------------------------------------


ECONOMIST_REPORT_TEMPLATE = """\
You are producing the Economist (macro backdrop) section of the equity
research note on {symbol}. This section MUST have these four H2 headings,
in this order, with no other sections:

## Rate Cycle
One short paragraph describing the current Fed funds rate from the
macro snapshot and classifying the cycle as EASING, HIKING, or
HOLDING. Cite [Source: fred.FEDFUNDS] on the line with the rate.
Add the 10-year Treasury yield and the 10-year breakeven inflation
on a second line ending with
[Source: fred.DGS10, fred.T10YIE].

## Inflation
One short paragraph: state the CPI YoY percent from the snapshot and
frame it relative to the Fed's 2% target. Cite
[Source: fred.CPIAUCSL] on the line with the YoY number.

## Real Economy
One short paragraph citing the latest Real GDP (level and YoY percent)
and the Unemployment Rate. Every line with a number ends with
[Source: fred.GDPC1] or [Source: fred.UNRATE] as appropriate.

## FX and Geopolitical Backdrop
Two subsections:
- **KRW / USD**: one sentence stating the current rate from the
  snapshot and framing whether a USD strengthening or weakening move
  amplifies or compresses gains for a Korean investor holding
  USD-denominated equities. End the sentence with
  [Source: fred.DEXKOUS].
- **Geopolitical risks relevant to the target**: two or three bullets
  drawn from (1) the value chain brief's "Geopolitical / regulatory"
  and "Vulnerable links" subsections, AND (2) the `geo.snapshot`
  tool_output block. Keep to risks with macro character (sovereign /
  trade / currency), not firm-specific platform defects. Each bullet
  either cites "per the value chain brief" (no tool citation needed)
  OR carries a news citation copied verbatim from the snapshot —
  "[Source: Google News, <outlet>, <YYYY-MM-DD>]" for headlines or
  "[Source: GDELT <theme>, <domain>, <YYYY-MM-DD>]" for GDELT
  articles. At least ONE bullet SHOULD carry a news citation when
  `geo.snapshot` contains relevant items, so the Economist's
  geopolitical read is grounded in actual recent events rather than
  pretraining memory.

Do NOT write a summary, a bottom line, or stock-specific implications.
Downstream agents read your output and draw their own conclusions.
"""


def make_economist_user_prompt(symbol: str, value_chain_text: str) -> str:
    """Build the Economist's user-prompt content.

    Only needs the value chain brief for qualitative geopolitical context.
    The FRED macro snapshot is injected via the pre-gathered facts block.
    """
    symbol = symbol.upper()
    return (
        f"You are writing the Economist section of the research note on {symbol}. "
        "Your numeric inputs are in <pre_gathered_tool_outputs> above. Your "
        "qualitative geopolitical context is the value chain brief below.\n\n"
        "<value_chain_brief>\n"
        f"{value_chain_text}\n"
        "</value_chain_brief>\n\n"
        + ECONOMIST_REPORT_TEMPLATE.format(symbol=symbol)
    )


# ---------------------------------------------------------------------------
# Steward task template (Phase 2) — consumes Analyst + Valuer + Skeptic
# ---------------------------------------------------------------------------


STEWARD_REPORT_TEMPLATE = """\
You are producing the Steward (final verdict) section of the equity
research note on {symbol}. This section MUST have these five H2 headings,
in this order, with no other sections:

## Verdict
One word only, on its own line, in all caps: **BUY**, **HOLD**, or **PASS**.
Nothing else in this section — no hedging prose, no "Conditional BUY",
no "BUY with caveats". The caveats go in Confidence Caveats below.

## Conviction Level
An integer 1-5, on its own line, as "**Conviction: N**". BUY may carry
2-5. HOLD and PASS carry only 1 or 2. Add one sentence explaining why
this specific conviction number rather than one higher or lower.

## Rationale
Two to four short paragraphs.

The FIRST paragraph must state the Bull thesis in one sentence and the
Skeptic's strongest rebuttal in one sentence.

The SECOND paragraph must, for each of the top-two Skeptic rebuttals,
explicitly label it as either NEUTRALIZED or SURVIVED:
  - NEUTRALIZED: cite a specific number or named fact from
    <pre_gathered_tool_outputs> or from the Analyst/Valuer sections
    that directly refutes the Skeptic's scenario. Speculative Bull
    language ("could", "may", "should", "expected to", "is likely
    to") is NOT a valid neutralization; mark any such rebuttal
    SURVIVED.
  - SURVIVED: restate the Skeptic's rebuttal in one sentence and
    note "no concrete Bull counter-evidence in report".

Additional paragraphs may provide context, but the Verdict MUST follow
directly from the SURVIVED/NEUTRALIZED labels:
  - Both top-two rebuttals NEUTRALIZED → BUY allowed (C3-C5).
  - One NEUTRALIZED, one SURVIVED → HOLD (C1-C2) or PASS (C1).
  - Both SURVIVED → PASS (C1) by default, HOLD only if the Bull
    thesis has independent evidence the Skeptic did not attack.

Every numeric value MUST end with [Source: <agent_or_tool>] — cite
Analyst, Valuer, Skeptic, or a specific tool name from
pre_gathered_tool_outputs. Do not introduce numbers that do not
already appear earlier in the report.

## Position Sizing Guidance
If Verdict is BUY: suggest a position-size band as a percent of portfolio
(e.g., "2-4% of equity allocation for this conviction level"). Map
conviction to size as a rough rule: C2→1-2%, C3→2-3%, C4→3-5%, C5→5-8%.
State the band on one line.

If Verdict is HOLD: state "No addition at current price; existing
position retained." — no percent band.

If Verdict is PASS: state "No position. Revisit when the rebuttal that
blocks BUY is materially neutralized (name it)."

## Confidence Caveats
Bulleted list of specific data gaps or events that would change the
verdict. Each bullet names:
- what is unknown
- why its resolution matters
- what direction (toward BUY or toward PASS) the verdict would move if
  the gap were filled positively for the Bull

At least one bullet must reference a specific Skeptic rebuttal that
COULD have been falsified if certain data were available but was not.

Do NOT issue SELL — short-selling is outside the mandate. Do NOT hedge
("somewhere between HOLD and PASS"). Commit to a single verdict from
{{BUY, HOLD, PASS}} with a single conviction integer.
"""


def make_steward_user_prompt(
    symbol: str,
    value_chain_text: str,
    analyst_output: str,
    valuer_output: str,
    skeptic_output: str,
) -> str:
    """Build the Steward's user-prompt content.

    Gives the Steward the full three-agent research note plus the value
    chain brief, so the final verdict can cite whichever section carries
    the relevant evidence.
    """
    symbol = symbol.upper()
    return (
        f"You are writing the Steward section of the research note on {symbol}.\n\n"
        "The Analyst, Valuer, and Skeptic have each completed their sections "
        "above. Your job is to synthesize all three into a final verdict. "
        "Read the Skeptic section especially carefully — your Rationale must "
        "name which rebuttals survive and which are neutralized.\n\n"
        "<analyst_section>\n"
        f"{analyst_output}\n"
        "</analyst_section>\n\n"
        "<valuer_section>\n"
        f"{valuer_output}\n"
        "</valuer_section>\n\n"
        "<skeptic_section>\n"
        f"{skeptic_output}\n"
        "</skeptic_section>\n\n"
        "For qualitative context (industry, customer concentration, "
        "geopolitical risks), refer to the value chain brief:\n\n"
        "<value_chain_brief>\n"
        f"{value_chain_text}\n"
        "</value_chain_brief>\n\n"
        + STEWARD_REPORT_TEMPLATE.format(symbol=symbol)
    )


# ---------------------------------------------------------------------------
# Analyst task factory (Phase 1B, retained)
# ---------------------------------------------------------------------------


def make_analyst_task(symbol: str, agent: Agent) -> Task:
    """Build the Analyst's analysis Task, injecting the value chain for `symbol`.

    The Task is self-contained: description covers both the operating
    instructions and the value chain context, and `expected_output` locks the
    seven-section structure.
    """
    symbol = symbol.upper()
    value_chain = _load_value_chain(symbol)
    description = (
        CONTEXT_INSTRUCTIONS.format(symbol=symbol, value_chain=value_chain)
        + "\n\n"
        + REPORT_TEMPLATE.format(symbol=symbol)
    )
    expected = (
        "A markdown report on " + symbol + " with exactly these seven H2 "
        "sections in order: Business Summary, Value Chain Context, Financial "
        "Health, Competitive Position / Moat, Valuation Context, Data Gaps "
        "and Warnings, Questions for Skeptic. Every numeric value cites the "
        "tool and source field it came from."
    )
    return Task(
        description=description,
        expected_output=expected,
        agent=agent,
    )
