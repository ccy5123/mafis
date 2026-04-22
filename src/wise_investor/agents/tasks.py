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


REPORT_TEMPLATE = """\
You are producing an equity research note on {symbol}. The note MUST have these
seven sections, in this order, with these exact H2 headings.

## 1. Business Summary
Two to three short paragraphs: what the company sells, to whom, how it makes
money, recent strategic direction. Facts only; no valuation comments.

## 2. Value Chain Context
Summarize the upstream, peer, and downstream relationships that matter for
durability. Cite the value chain brief when you use its claims, using the
phrase "per the value chain brief" so the reader can trace it.

## 3. Financial Health
Every numeric line in this section MUST follow this exact format:
- **<Metric name>**: $<value> — [Source: <tool_output name>]

Draw values only from <tool_output> blocks in <pre_gathered_tool_outputs>.
Include at minimum: revenue, net_income, operating_income, gross_profit,
ebitda, free_cash_flow, total_debt. Also state current PER and EV/EBITDA
from the calculate_per and calculate_ev_ebitda outputs.

## 4. Competitive Position / Moat
Five-to-ten-year durability analysis. Structural advantages (scale,
ecosystem, switching cost, IP, regulatory) vs erosion forces. Then quote
the peer multiples table verbatim inside a fenced code block, and below it
write two short paragraphs: first interpreting the target's relative
positioning versus peers, second naming the peer(s) that pose the most
credible 5-year threat and why.

## 5. Valuation Context (brief)
Two bullet lines only:
- **Current multiples vs peers**: state the target's PER and EV/EBITDA with
  [Source: ...] citations, then compare to the numerically highest and
  lowest peer from the peer multiples table (name both peers and their
  numbers). One sentence interpretation.
- **Market-implied growth assumption**: quote the implied FCF growth rate
  from reverse_dcf verbatim with [Source: reverse_dcf], then one sentence
  comparing that rate to a plausible historical benchmark.

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

**Known unknowns (from value chain brief):** Copy the bullet list under
the "Known unknowns (do not pretend to know)" heading of the value chain
brief verbatim into this subsection.

## 7. Questions for Skeptic
Exactly 5 questions, numbered 1 through 5. Each question MUST have these
three labeled fields in this order, each field one sentence:

1. **Claim**: <the specific claim elsewhere in this report that is under attack>
   - **Assumption**: <the assumption the claim rests on>
   - **Evidence that would falsify it**: <concrete evidence or event that would refute it>

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
