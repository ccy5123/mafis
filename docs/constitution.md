# MAFIS Redefinition + Constitution — Final Report v2

**Status:** Final handoff for next implementation session
**Supersedes:** Technical Report v1 (now obsolete) and Sections 5-6 of
the original handoff document
**Posture:** This document is the single source of truth for the
redefined system. The next session should read this in full, then
reference the original handoff only for sections 1-4 and 7-8 (which
describe the system as it exists today).

---

## Table of Contents

**Part I — System Identity** (Why we are rebuilding)
1. Why this document exists
2. The system's redefined goal
3. What was wrong with the original plan
4. Six commitments locked during the dialogue
5. Comparison with master investors

**Part II — Architecture** (How the system is shaped)
6. Six-stage pipeline overview
7. Telegram tip channel — redefined
8. Open critiques carried forward

**Part III — Constitution (A): Definitions of the three axes**
9. Hierarchy structure
10. Moat axis
11. New Frontier axis
12. Bottleneck axis
13. General rules across all axes

**Part IV — Constitution (B): Quantitative proxy specification**
14. Pre-processing
15. Quantitative proxies per axis
16. Stage 2 integration logic
17. Limitations of automation

**Part V — Constitution (C): LLM prompt templates**
18. Stage 3 — Light qualitative screening
19. Stage 4 — Skeptic prompt
20. Stage 4 — Defender prompt
21. Stage 4 — Steward prompt
22. Operational notes

**Part VI — Implementation guidance**
23. Work order
24. What this document does not decide
25. Closing note

---

# Part I — System Identity

## 1. Why this document exists

The original handoff arrived with a confident plan: build a value
chain graph database, expand it via focused crawling, compress it via
community detection, hand the result to HRP for portfolio
construction. Four pending decisions waited for the user's sign-off
(graph DB choice, expansion aggressiveness, visualization stack,
optimizer library).

A long clarifying dialogue revealed that this plan rested on an
assumption the user did not actually hold. The original document
modeled the user as someone who **brings tickers to the system and
asks for adversarial review of those tickers**. The user is not that.
The user is someone who wants the system to **find tickers he doesn't
know about, screen them by his own evaluation rubric, and assemble a
portfolio from the survivors**, and who has explicitly committed that
his existing ticker preferences should not influence what the system
considers.

This is not a small revision. It moves the system's center of gravity
from per-ticker analysis to universe-driven discovery. It changes
which components of the original plan survive, which die, and which
need to be added. It also changes the work order: the original
document proposed starting with the graph database, but a
universe-driven system needs the evaluation rubric formalized first,
because the rubric is what the screening pipeline and the adversarial
analysis must both be aligned to.

The dialogue also produced the rubric itself, in the form of a
constitution — definitions of three axes, a hierarchy that combines
them, quantitative proxies that automate the first stage of
evaluation, and prompt templates that align the LLM stages to the
constitution. Parts III through V of this document are that
constitution.

## 2. The system's redefined goal

The system MAFIS is being redirected toward is described in one
sentence:

> Discover stocks the user does not currently know about from an
> objective universe, screen them through the user's evaluation rubric
> (moat / bottleneck-resolution / new-frontier), submit survivors to
> the existing adversarial analysis engine — modified to test the
> rubric directly — and assemble a portfolio from those that pass,
> using HRP for weight allocation.

The system's purpose is not to make the user a master investor.
Master investors achieve outsized returns through information
asymmetry, time devotion, and sustained willpower that retail
investors cannot match. The realistic purpose of the system is to
help the user **make better decisions than the average retail
investor consistently**, by externalizing willpower into code and
data-validating the resulting decisions over time. This framing
should sit at the top of every interaction: the system is a retail
discipline tool, not a Buffett emulator.

## 3. What was wrong with the original plan

The original handoff was internally consistent and technically sound.
It proposed mapping each piece of the next-phase work to a
well-studied algorithm: focused crawling for graph expansion,
filtered correlation networks plus community detection for
compression, HRP for portfolio construction, Thompson sampling for
re-analysis priority. This mapping is good engineering hygiene —
don't invent algorithms when published ones exist.

But it answered the wrong question. The original plan answered "given
that the user analyzes one stock at a time, how do we extend that
into a portfolio view?" The user's actual question is "how do I stop
choosing the stocks at all and let the system choose them?" These two
questions look similar but require different architectures. The
original plan's value chain graph starts at a seed ticker the user
provides (NVDA was the running example) and expands outward. That
expansion strategy is fundamentally tied to the user-provides-seed
model. If the seed is supposed to come from the universe rather than
the user, ego-graph expansion is solving the wrong problem; what is
needed is a global graph laid down across an industry, on top of
which discovery operates.

A second flaw was in HRP's positioning. The original document called
HRP a "near-perfect fit" for the portfolio problem. HRP's input is a
hierarchical clustering derived from a returns-based distance matrix.
The original plan implied that HRP would consume the value chain
graph directly, which would have required either ignoring the graph
(making it decorative) or inventing a non-standard distance metric on
graph edges (defeating the "don't invent algorithms" principle). The
redefined architecture resolves this cleanly: HRP runs on the price-
return matrix of the rubric-passing pool, and the value chain graph
contributes only as a post-hoc constraint.

A third flaw was implicit but corrosive. The original plan treated
the user-authored value chain briefs as the seed nodes of the graph.
This made the entire graph — and therefore every portfolio
recommendation derived from it — a function of the user's manually
authored content. If the user's NVDA brief omits EDA toolchain
exposure, the graph never learns about EDA exposure, the portfolio
optimizer never sees that risk, and the system's promise to protect
the user from his blind spots fails at the very point of greatest
leverage. Replacing user-authored seeds with an objective universe
removes this single point of failure.

The original handoff document anticipated none of this because the
user's redefinition emerged only during the dialogue. It is not the
fault of that document's author; it is the natural consequence of
writing a plan before the user had been forced to articulate what he
actually wanted.

## 4. Six commitments locked during the dialogue

These six commitments are non-negotiable in the redefined system.
Every architectural and constitutional decision in this document
follows from them.

**Commitment 1 — User preferences cannot influence universe
membership.** When asked whether stocks the user already favors
should receive special consideration if they fail the rubric, the
user answered: there is no reason to look at a stock that does not
pass the rubric, and that is the core of the automation. This closes
off whitelists, favorite lists, and any channel through which the
user's attention pattern could re-enter discovery.

**Commitment 2 — The system has authority to PASS, the user does
not have authority to override.** Stocks the system PASSes are not
candidates the user can revive through any in-system mechanism. The
user accepts that the system will miss some good companies (Tesla in
2010-2015 was discussed as the canonical example) as the price of
self-deception prevention.

**Commitment 3 — Precision over recall.** When in doubt, fail the
candidate. The cost of admitting a bad company exceeds the cost of
missing a good one. This commitment shows up in every constitutional
decision: conservative thresholds, multiple required conditions,
explicit auto-PASS lists.

**Commitment 4 — No Dreamer module.** Late in the dialogue the user
proposed adding a "Dreamer" module that would observe companies the
main system PASSed. Through examination it became clear that this
module — even with caveats that it could not trigger BUYs — would
function as a self-deception channel through mere-exposure effects.
The user explicitly rejected this module after seeing the pattern.

**Commitment 5 — Hierarchy is fixed, not adjustable per-ticker.**
The user does not get to apply different rules to different
companies. The hierarchy stated in Section 9 applies uniformly. If
later evidence suggests the hierarchy itself needs revision, that
revision is a constitutional change requiring explicit acknowledgment
in the ledger, not a per-ticker exception.

**Commitment 6 — Output is binary.** BUY or PASS. No conviction
levels, no watch lists, no maybe categories. Position sizing is
delegated entirely to HRP within configured constraints. Conviction
gradations were considered and rejected as a self-deception channel
("conviction-2 BUY, but I really like this one, so I'll size it
larger") and as a violation of master investors' actual practice
(neither Buffett nor Lynch use a numeric conviction scale).

## 5. Comparison with master investors

The constitution defined in Part III aligns with patterns visible in
the practice of Warren Buffett and Peter Lynch, with two important
adaptations.

**Pattern preserved — Definition by exclusion is sharper than
definition by inclusion.** Buffett's circle of competence and Lynch's
category mismatch avoidance both work by *what they refuse* more than
by *what they select*. The constitution mirrors this: each axis has
detailed auto-PASS conditions, and the most important sentences in
each axis definition are about what is *not* a moat, *not* a new
frontier, *not* a bottleneck.

**Pattern preserved — Only post-validation evidence counts.**
Buffett invested in Apple in 2016, eight years after the App Store, by
which point the moat was structurally evident. Lynch required fast
growers to have demonstrated revenue in two or more cities or
countries before qualifying. The constitution's three-year time
thresholds (on moat ROIC persistence, new frontier paradigm age, and
bottleneck position duration) implement the same principle: the
system catches the validated, not the trying.

**Pattern preserved — Narrative is treated as evidence of weakness,
not strength.** Buffett's "if you cannot explain in one or two
sentences, skip it" and Lynch's warning that a busy Dunkin' Donuts is
just a tip to investigate, not a buy signal, both reflect distrust of
narrative as a substitute for analysis. The Skeptic prompt
specifically blocks narrative-style attacks ("hindsight reasoning,"
"vague macro hand-waving") and the constitution rejected the Dreamer
module precisely because it would have generated narrative.

**Adaptation 1 — System externalizes willpower.** Master investors
ran their checklists in their own heads, applying personal discipline
each time. The user does not claim Buffett-level willpower and is
explicit that the system is meant to enforce what the user cannot
reliably enforce alone. This is a legitimate retail adaptation. It
is also the reason every commitment in Section 4 is locked
architecturally rather than left to user judgment.

**Adaptation 2 — System uses an oligopoly-aware bottleneck
definition.** Master investors implicitly handle division-of-labor
oligopolies (Cadence and Synopsys both pass Buffett's quality
filter despite occupying the same market) through human judgment.
The constitution makes this explicit, encoding division-of-labor
oligopoly recognition into the bottleneck axis Pass condition 2.

The honest expectation is that this system, well executed, will help
the user beat the average retail investor consistently. It will not
make the user beat S&P 500 over a long horizon, because that is
something even master investors achieve only in 10-20% of cases over
multi-decade periods. The ledger will tell the truth over time.

---

# Part II — Architecture

## 6. Six-stage pipeline overview

The redefined system has six stages, executed in sequence on a
schedule (probably weekly, given the user's once or twice per week
analysis rhythm).

**Stage 1 — Universe ingestion.** The system maintains a list of
tickers it considers eligible for analysis. Membership is determined
entirely by objective, user-independent criteria: index membership
(S&P 500, KOSPI 200, optionally Nasdaq-100 or other broad indices),
market capitalization floors, liquidity floors, exclusions for
categories the rubric is not tuned for (financials and utilities,
likely). The list is refreshed daily.

For each ticker in the universe, the system fetches and caches a
standard panel of fundamentals: revenue history, gross margin, ROIC,
cash flow, R&D as a fraction of sales, customer concentration where
disclosed, recent capex trajectory, segment breakdown.

**Stage 2 — Quantitative pre-filter.** Universe is large (hundreds
to thousands of tickers). Direct LLM evaluation on all of them is
prohibitively expensive. The pre-filter is therefore quantitative
and cheap, using the proxies specified in Part IV. It also applies
the multi-segment 30%+ rule (Section 13) as a hard filter. Output:
a shortlist of perhaps fifty to one hundred tickers, each tagged with
which axes are candidates for passing.

**Stage 3 — Qualitative screening (lightweight LLM).** Each
shortlisted ticker is passed to a focused LLM call following the
prompt template in Section 18. The call returns a binary verdict
on each axis (PASS or FAIL), an indication of whether the hierarchy
gate is satisfied (advance to Stage 4 or reject), and brief
reasoning. The point is to reduce fifty-to-one-hundred candidates to
a working pool of ten to fifteen names that survive both quantitative
and qualitative checks.

**Stage 4 — Adversarial analysis (rubric-aware MAFIS).** This is
where the existing MAFIS engine is reused, modified to align with
the constitution. Skeptic attacks are structured along the rubric
axes the company was tagged for (Section 19). Defender labels each
attack DEFENDED or CONCEDED with cited evidence (Section 20). The
audit verifies citations and downgrades weak ones. The Steward
applies the simplified hierarchy logic (Section 21) to issue a final
binary verdict.

**Stage 5 — Value chain positioning (post-pass).** Tickers that
survive Stage 4 are placed onto a global value chain graph that the
system maintains separately. The graph represents the structure of
industries the universe spans, with nodes for companies, sectors,
shared infrastructure, and typed edges for supply, competition,
customer relationships, and shared dependencies. The positioning
step asks: where on this graph do today's survivors sit, and which
clusters are over-represented or under-represented in the current
pool?

**Stage 6 — Portfolio construction (HRP).** Hierarchical Risk
Parity is applied to the price-return matrix of the rubric-passing
pool. The hierarchy is built from price correlations following the
standard HRP recipe; the value chain graph contributes post-hoc
adjustments rather than serving as the primary hierarchy. After HRP
produces its initial weight allocation, the system checks for cases
where two HRP-favored names sit at the same value-chain node and
down-weights one. Single-position bounds: minimum 1%, maximum 30%.
The 30% upper bound matches Buffett's historical average for largest
holdings (1981-2024).

The user's existing portfolio enters at Stage 6 as a constraint,
producing incremental rebalancing recommendations rather than clean-
sheet portfolios.

## 7. Telegram tip channel — redefined

The original tip channel design — user forwards a message, system
classifies it, classified tip becomes user-provided context for
future analysis — predates the user's commitment in Section 4 to
keep preferences out of universe membership. Under the redefined
goal, that channel is a leak: any ticker the user mentions becomes a
ticker the system analyzes, which means the user's attention pattern
is covertly determining what gets analyzed.

The redefined design separates two functions the original channel
conflated. **Tip ingestion is decoupled from analysis triggering.**

When the user forwards a tip, the system:
- Logs the tip with full metadata (ticker, source platform, date,
  rough content classification)
- Does NOT trigger any analysis as a consequence of the tip
- If the mentioned ticker is in the universe and survives screening
  on its own merits, it will be analyzed in the normal flow, and its
  appearance in Stage 4 output will carry an annotation indicating
  the user mentioned it N days ago
- The annotation is metadata for the user to read, NOT context
  delivered to any LLM in any stage

This design preserves the architectural commitment to keeping user
preferences out of universe membership, while retaining the
analytical artifact (the tip log itself). After sufficient time, the
tip log enables a separate analysis: do the user's tip intuitions
have predictive power compared to system-discovered tickers? That is
a legitimate empirical question for later, not a reason to feed tips
into the LLM today.

A side benefit: the set of tickers the user mentioned but the system
did not surface is itself useful. It represents the gap between the
user's attention and the system's rubric. If that gap is
systematically biased, the user should see it. The redefined
channel naturally produces this comparison; the original did not.

## 8. Open critiques carried forward

The dialogue began with six critiques of MAFIS as it exists today.
The redefinition resolves some, mitigates others, and leaves the rest
standing. Status of each:

The discipline matrix's mathematical force — partially preserved.
The matrix is now structured along rubric axes, which constrains the
LLM more tightly than the original generic structure. But the
underlying issue remains: DEFENDED/CONCEDED labels are still LLM-
produced, and the matrix's surface still rests on that foundation.

The Skeptic five-attack mandate — partially fixed by axis constraint.
Attacks are now constrained to the axes the company was scored on,
making them less interchangeable and harder to dilute. Whether five
is the right number remains open.

The dual-language audit-trail issue — unresolved. Reports translated
to Korean while audit trail remains English creates two texts where
decision-relevant hedging may differ. The next session should either
move the user's primary reading surface to English or design loss-
controlled translation that preserves conditional language.

The six-month evaluation window — partially mitigated. The primary
measurable shifts from "did the verdict outperform" to "did the
rubric-screening correctly classify candidates." Rubric correctness
can be probed against fundamental events (did the alleged moat erode?
did the alleged bottleneck get resolved?) faster than against price
movement alone.

The HRP-graph integration problem — resolved. HRP consumes price
returns; value chain serves as a post-hoc constraint. The two
structures cooperate at different stages.

The seed-bias problem — resolved. Seeds come from the universe, not
from user-authored briefs. The universe-driven design eliminates
this entire failure mode.

---

# Part III — Constitution (A): Definitions

## 9. Hierarchy structure

The hierarchy is the most important architectural decision in the
constitution. Every axis definition serves it.

**Stated hierarchy:**

> Two axes or more must pass. The growth axis (New Frontier or
> Bottleneck) must be among them. Moat alone, New Frontier alone, and
> Bottleneck alone all PASS the company.

**Allowed BUY pairs:**
- Moat + New Frontier
- Moat + Bottleneck
- New Frontier + Bottleneck
- All three (subsumed by the above)

**Why this hierarchy:**

The user's investment philosophy is explicitly to capture long-term
shifts in the technology world. Coca-Cola, with a strong moat but no
growth axis, was rejected during the dialogue as not matching this
philosophy — durable but not aligned with the user's directional
view. The hierarchy enforces this rejection mechanically.

The two-axes-or-more requirement implements the precision-over-
recall commitment. Companies strong on a single axis often fail on
the others in ways that materially reduce expected returns. Tesla in
2010-2015, with new frontier alone, was rejected by the user as too
risky given that most new-frontier attempts fail. TSMC and ASML,
with moat + bottleneck, were accepted as canonical BUYs.

**Output is binary** (Commitment 6). BUY or PASS. No conviction
levels. Position sizing handled by HRP within bounds 1% to 30%.

## 10. Moat axis

**Definition:** A company holds a moat when its return on invested
capital sustainably exceeds its cost of capital because of a
structural reason that has held for at least 3 years and shows no
clear erosion threat going forward.

Three key terms: **structural** (per Pat Dorsey — inherent to the
business, not a result of a great product or service or a single
executive); **3 years or more** (time validation; shorter periods may
be cycle phase artifacts); **no erosion threat** (forward-looking;
PayPal-pattern past moats with current erosion fail).

**Pass conditions** (all required):

*Condition 1 — Quantitative persistence.* ROIC has exceeded the
industry median by at least 5 percentage points for 3+ years, and
the gap is not narrowing rapidly (less than 0.5pp per year reduction).

*Condition 2 — Structural reason identifiable.* The reason for the
ROIC advantage is classifiable as one of Morningstar's four buckets:
intangible assets (brand, patents, licenses, regulatory protection);
switching costs (integration, learning, data migration); network
effects (preferring interstitial networks per Dorsey's PayPal
lesson); cost advantages (scale, location, proprietary resources,
process technology).

*Condition 3 — No active erosion.* None of the following clearly in
progress: new entrants gaining material share; substitute technology
bypassing the moat's core value proposition; regulatory change
neutralizing the advantage; key customer churn pattern.

**Auto-PASS conditions:**

*Auto-PASS 1.* ROIC data spans less than 3 years (insufficient
history).

*Auto-PASS 2.* ROIC advantage exists but no structural reason fits
the four buckets — likely management excellence or temporary market
conditions, both subject to mean reversion.

*Auto-PASS 3.* Past moat with current erosion signs (the PayPal
pattern). This is the most important auto-PASS — it is the trap the
system is most likely to fall into without explicit guard.

*Auto-PASS 4.* The moat depends primarily on a single executive.
Steve Jobs's Apple, Elon Musk's Tesla — the executive-dependent
component is excluded; only institutionalized components (Apple's
ecosystem lock-in, Tesla's Supercharger network) are evaluated.

**Explicit non-inclusions** (these are not moats):

High margin alone (margins are a possible *result* of a moat, not
evidence). Market share rank one (rank without identified reason for
holding it). Brand recognition (recognition without pricing power
conversion). Fast growth (growth belongs to New Frontier axis).
Monopoly in a small or shrinking market (position without market).

**Sample applications:**

Pass — Coca-Cola (intangible + cost advantage; ROIC above industry
50+ years; minimal erosion). Note: passes moat axis but fails
hierarchy because no growth axis.

Pass — TSMC (cost advantage + switching costs; clear and measurable;
geopolitical risk exists but does not erode the technical advantage
itself).

Pass — ASML (intangible + switching costs; near-monopoly; clearly
durable).

Pass — Microsoft Office/Windows/Azure (switching costs + network
effects; cloud transition successful; erosion threats addressed).

Auto-PASS 3 — PayPal (past network effects, current erosion via NFC
bypass).

Auto-PASS 3 + 4 — Tesla (Supercharger institutionalized, but the
overall company is heavily Musk-dependent and EV competition
clearly eroding the broader moat).

Auto-PASS 3 — Intel current (past x86 dominance, current erosion via
ARM/RISC-V and lost process leadership).

Auto-PASS 1 — most fast-growing SaaS new entrants (insufficient
history).

## 11. New Frontier axis

**Definition:** A company has opened a new frontier when it has
changed how an industry operates, and that change has been *imitated*
by other companies in the industry, becoming an industry standard.
The imitation must be externally verifiable, not the company's own
claims, and the original change must be at least 3 years old.

Three key terms: **as a result** (not intent — "trying to change a
paradigm" does not count, "the paradigm has changed" does);
**imitated** (industry standardization, not single-company
differentiation); **externally verifiable** (analyst recognition,
competitor behavior, not company PR).

**Pass conditions** (all required):

*Condition 1 — Imitation evidence.* At least 2 same-industry
companies have adopted the same operating method, product structure,
or business model. Not competitive product launches, but
fundamental adoption of the operating approach.

*Condition 2 — Analyst recognition.* Multiple independent industry
analyst sources state explicitly that this company has changed the
industry, or that the industry has reorganized after this company's
arrival. The company's own statements do not count.

*Condition 3 — Time elapsed.* At least 3 years since the paradigm
introduction (typically dated by the first major new-segment revenue
or first paradigm-defining product launch). Less than 3 years
without imitation likely means a temporary differentiation, not a
paradigm.

**Auto-PASS conditions:**

*Auto-PASS 1.* The company's own announcements, IR materials, or CEO
interviews are the sole or primary source of the new-frontier claim.
Self-claim is not evidence.

*Auto-PASS 2.* The new paradigm depends on unproven future promises
(e.g., "FSD will be completed soon," "robotaxi service will start").
Future-tense dependence directly contradicts the post-validation
definition.

*Auto-PASS 3.* No same-industry company has imitated. Paradigm
change is by definition industry standardization, so absence of
imitation means isolated differentiation.

*Auto-PASS 4.* Paradigm introduction is less than 3 years old —
insufficient validation time. Re-evaluate after 3 years.

**Explicit non-inclusions** (these are not new frontiers):

Fast revenue growth alone (growth may follow paradigm change but is
not evidence of it). CEO vision (charismatic announcements are
unrelated to whether the paradigm has actually changed; evaluate
only after results validate). Market share surge (share may indicate
winning within the existing paradigm, not changing it). Technical
superiority (better chips, better algorithms — these are
improvements within an existing paradigm). New market entry
(geographic or segment expansion is not paradigm extension).

**Sample applications:**

Pass — Tesla 2026 (OTA + direct sales + battery-first design now
imitated by BMW, Mercedes, GM, Ford; 10+ years elapsed; analyst
consensus on industry reorganization).

Pass — Netflix (streaming + original content; Disney+, HBO Max,
Apple TV+ all converged on same model; 10+ years).

Pass — AWS (cloud-as-service; Azure and GCP same model; clear).

Pass — NVIDIA (GPU as general-purpose parallel accelerator; AMD and
Intel converged; CUDA became industry standard).

Auto-PASS 4 + 3 — Tesla 2010-2015 (paradigm introduction starting,
no imitation yet, less than 3 years validation).

Auto-PASS 1 — Theranos (paradigm claim entirely from company self-
statements; no external validation).

Auto-PASS 2 + 3 — Magic Leap 2015-2020 (AR headset paradigm claim,
future-tense dependence, no imitation).

Sample post-PASS revealed — WeWork (some imitation existed but
paradigm economics never validated; revealed clearly through time as
the paradigm itself failed).

**Note on this axis being primarily a classification tool:** Because
the post-validation requirement is strict, this axis primarily
catches companies that are already well-known. Genuine *discovery*
in the system happens primarily through the bottleneck and moat
axes. The user accepted this trade-off explicitly during the
dialogue: missing earlier-stage paradigm shifts is the price of the
precision commitment.

## 12. Bottleneck axis

**Definition:** A company occupies a bottleneck position when its
supply or production cannot be replaced by other companies in 1-2
years for technical, resource, or regulatory reasons, and downstream
companies materially depend on it.

Three key terms: **non-replaceable in 1-2 years** (short-term, not
permanent — long enough that disruption matters); **downstream
dependency** (others depend on this company, not just this company's
own importance); **structural reason** (technical, resource, or
regulatory — not luck or temporary market conditions).

**Pass conditions** (all required):

*Condition 1 — Downstream dependency proven.* Combined revenue of
companies that disclose this company as a critical supplier or
partner is at least 5 times this company's own revenue. Or, as a
weaker quantitative proxy, top customer concentration is 40% or
greater with downstream Risk Factors disclosure confirming
materiality.

*Condition 2 — Replacement difficulty proven.* At least one of:
**technical** (0-2 companies with equivalent technology AND market
position; OR for division-of-labor oligopolies like Cadence/Synopsys
or Samsung/SK Hynix in memory, each member passes if it cannot be
replaced in its own area within 1-2 years; simple competitive
oligopolies where any member can substitute for any other do not
pass); **resource** (control of or dominant access to critical raw
materials); **regulatory** (high entry barriers — defense, parts of
medical, infrastructure).

*Condition 3 — Position duration.* At least 3 years in the
bottleneck position with no clear erosion signal. Erosion signals
include: government policy push for self-sufficiency (China
semiconductors, US CHIPS Act); demonstrated alternative technology
reaching pilot deployment; major downstream customer announcing
diversification.

**Auto-PASS conditions:**

*Auto-PASS 1.* High market share but easily replaceable (Coca-Cola
pattern — Pepsi can substitute, so not a bottleneck).

*Auto-PASS 2.* Downstream market itself is shrinking. Bottleneck in
a sunset industry has limited value.

*Auto-PASS 3.* Position formed by temporary geopolitical or
situational reasons (some COVID-era medical supply bottlenecks
disappeared post-2022).

*Auto-PASS 4.* Major downstream companies actively reducing
dependency. Apple's Silicon to reduce dependency on third-party
chip suppliers, Apple/Samsung/Intel foundry diversification —
diversification *attempts in progress* are themselves erosion
signals. (This trigger may be sensitive; calibrate based on whether
the reduction is announced versus actually executed.)

*Auto-PASS 5.* Less than 3 years in position; insufficient duration.

**Explicit non-inclusions:**

Simple market share rank (rank is position, not bottleneck —
replaceability is the test). High vertical integration (Apple
Silicon is internal-use; no external dependents means no external
bottleneck). Innovative product holding (innovation belongs to New
Frontier or Moat axes). High prices (price may follow dependency
but is not evidence of it).

**Sample applications:**

Pass — TSMC (advanced node sole producer; NVIDIA, Apple, AMD,
Qualcomm all dependent; downstream revenue 10x+; Samsung gap clear).

Pass — ASML (EUV 100%; TSMC, Samsung, Intel all dependent; zero
substitute).

Pass — Shin-Etsu / SUMCO (300mm wafers 54-80%; all foundries
dependent; near-Japan-monopoly).

Pass — Cadence and Synopsys (division-of-labor oligopoly in EDA;
each non-replaceable in its specialty within 1-2 years; near-100%
customer retention with chip designers globally).

Pass — SK Hynix (HBM near-sole supplier 2024-2026 for NVIDIA AI
chips; division-of-labor oligopoly with Samsung and Micron in
memory).

Pass — Samsung Electronics memory division (division-of-labor
oligopoly partner; passes via the multi-segment 30%+ rule because
memory is more than 30% of total revenue).

Auto-PASS 1 — Coca-Cola (high share, but Pepsi substitutes; not a
bottleneck).

Excluded — Apple Silicon (vertical integration; no external
dependents).

Auto-PASS 4 — Tesla Supercharger (was the de facto charging
standard, but other manufacturers adopting NACS plus building own
networks signals dependency reduction).

Auto-PASS 4 — Intel x86 (past clear bottleneck; ARM/RISC-V
substitution in progress).

Auto-PASS 4 — Boeing 737 (was bottleneck for short-haul; Airbus
A320 substitution plus safety-driven erosion).

## 13. General rules across all axes

**Multi-segment company rule (30% threshold):** A company is
evaluated through its primary segment when one exists. The system
identifies segments accounting for 30% or more of total revenue. If
no segment crosses 30%, the company is too diversified for clean
axis evaluation and is excluded from the universe before Stage 3.
If one or more segments cross 30%, axis evaluation is performed for
those segments, and the company passes an axis if its primary
segment passes that axis.

This rule resolves the conglomerate problem (Samsung Electronics,
Microsoft, Apple, Amazon all have multiple substantial segments). It
also creates a natural exclusion of holding-company structures where
no single business dominates.

**Time uniformity:** All three axes use the same 3-year time
horizon for persistence checks. This was a deliberate alignment
during the dialogue; it makes cross-axis comparison clean and
consistent.

**Conservative tie-breaking:** When in doubt on any axis, FAIL the
axis. The constitution is designed for a precision-over-recall
system.

**Constitutional change is explicit:** Modifications to axis
definitions, hierarchy, or thresholds are version-controlled and
recorded in the ledger. Every Stage 4 record carries the constitution
version it was evaluated against. This enables clean attribution
when later analyzing system performance — if results change, the
constitution change is visible.

---

# Part IV — Constitution (B): Quantitative proxy specification

## 14. Pre-processing

For each ticker entering the system:

```
primary_segment_exists: boolean
primary_segment_name: string (null if not exists)
primary_segment_revenue_share: float (0.0-1.0)
```

Computed by parsing 10-K Item 8 segment reporting (US) or K-IFRS
business segment disclosures (Korea). Companies failing
`primary_segment_exists` are removed from universe.

## 15. Quantitative proxies per axis

**Moat axis proxies:**

```
roic_3y_avg:                ROIC averaged over 3 years
roic_industry_3y_median:    Industry median (GICS Level 3)
roic_advantage:             roic_3y_avg - roic_industry_3y_median
roic_advantage_trend:       Linear regression slope of advantage
gross_margin_3y_std:        Standard deviation of quarterly gross margin
gross_margin_industry_std:  Industry median of same metric
top10_customer_share_3y_change: Trend in customer concentration
```

PASS conditions:
- `roic_advantage >= 0.05` (5pp)
- `roic_advantage_trend >= -0.005` (less than 0.5pp/year erosion)
- `gross_margin_3y_std <= gross_margin_industry_std * 1.2`

Customer share trend feeds Stage 3 LLM as qualitative input rather
than auto-pass/fail.

**New Frontier axis proxies:**

```
years_since_first_segment_introduction: int
new_segments_added_5y:                  int
analyst_keyword_frequency:              float (optional, NLP)
```

PASS conditions:
- `years_since_first_segment_introduction >= 3` (auto check)
- Other proxies are inputs to Stage 3 LLM, which performs the
  imitation-evidence and analyst-recognition checks qualitatively

This axis is primarily LLM-driven by design. Stage 2 auto-rejects
only on the time threshold.

**Bottleneck axis proxies:**

```
top5_customer_share:               float (0.0-1.0)
hhi:                               int (Herfindahl, 0-10000)
top3_market_share_sum:             float
each_top3_member_share:            list of float
diversification_attempt_signals:   int (count of recent events)
```

PASS conditions:
- `top5_customer_share >= 0.4` OR explicit downstream dependency
  documented (LLM verification at Stage 3)
- `hhi >= 2500` (concentrated market)
- For division-of-labor candidates: `top3_market_share_sum >= 0.7
  AND each_top3_member_share >= 0.15` (LLM verifies whether the
  oligopoly is division-of-labor or simple-competition)

Diversification signal proxy:
- If `diversification_attempt_signals > 0`, trigger Auto-PASS 4
  evaluation in Stage 3

## 16. Stage 2 integration logic

Per ticker:

1. Multi-segment pre-check. If `primary_segment_exists == False`,
   exclude.
2. Compute proxies for all three axes.
3. Per-axis classification: each axis is PASS, FAIL, or NEED_LLM.
   PASS means proxies clearly meet conditions. FAIL means clearly
   fail. NEED_LLM means proxies are inconclusive or only partially
   determinative (e.g., New Frontier always except auto-rejection).
4. Apply hierarchy gate at quantitative level: if 2+ axes are PASS
   or NEED_LLM, AND a growth axis (New Frontier or Bottleneck) is
   among them, advance to Stage 3. Otherwise reject.

This funnels universe (typically several hundred to a thousand
candidates after multi-segment filter) down to fifty to one hundred
candidates suitable for LLM screening.

## 17. Limitations of automation

Several axis conditions resist quantitative automation:

- Moat: 4-buckets classification, erosion threat assessment, manager
  dependency check
- New Frontier: imitation evidence, analyst recognition
- Bottleneck: division-of-labor verification, dependency proof for
  thinly-disclosed B2B relationships, erosion progression assessment

These are explicitly delegated to Stage 3 and Stage 4 LLM stages.
This is not a defect; it is an honest design that acknowledges Pat
Dorsey's observation that pure quantitative moat detection is no
longer reliable in modern accounting environments.

The two-stage structure (quant filters then LLM) absorbs this
limitation cleanly. Quant narrows; LLM judges qualitatively.

---

# Part V — Constitution (C): LLM prompt templates

## 18. Stage 3 — Light qualitative screening

Purpose: per-ticker binary verdict on each axis. Reduce shortlist
to ten-to-fifteen Stage 4 candidates.

```
You are evaluating whether [TICKER] passes the user's investment 
rubric.

The user's rubric has three axes. Each axis has a precise definition.
Evaluate each axis independently and return PASS or FAIL.

[Axis definitions follow, copied verbatim from Sections 10-12 of the
constitution. Each axis includes its Pass conditions, Auto-PASS 
conditions, Explicit non-inclusions, and the provided quantitative 
data for this ticker.]

=== HIERARCHY GATE ===

Apply after evaluating all three axes:
- Count PASSes
- Check growth axis inclusion (New Frontier OR Bottleneck must be PASS)

Final classification:
- 2+ PASSes AND growth axis included → ADVANCE_TO_STAGE_4
- All other combinations → REJECT

=== OUTPUT FORMAT (JSON) ===

{
  "moat":        {"verdict": "...", "bucket": "...", "reasoning": "..."},
  "new_frontier":{"verdict": "...", "imitation_evidence": [...], 
                  "reasoning": "..."},
  "bottleneck":  {"verdict": "...", "type": "...", "reasoning": "..."},
  "hierarchy_decision": "ADVANCE_TO_STAGE_4" or "REJECT",
  "rejection_reason": "..." (if REJECT)
}

Be conservative. When uncertain, FAIL the axis. The user has 
explicitly accepted that the system will miss some good companies in 
exchange for not admitting bad ones.
```

Key design choices: definitions copied word-for-word from
constitution; quantitative data injected as inputs; final paragraph
explains the precision-over-recall trade-off so the LLM applies it
consistently.

## 19. Stage 4 — Skeptic prompt

```
You are the Skeptic. Attack the bull thesis on [TICKER] with 5 
specific attacks.

This company has been classified as having strengths in: [list of 
PASSED axes]. Your attacks must be aligned with those axes.

Attack distribution:
- 2 axes PASSED → 3 attacks on stronger axis, 2 on weaker
- 3 axes PASSED → 2 attacks on each of 3 axes (last attack on overall 
  thesis)

ATTACKS ON MOAT:
- Erosion (cite trend data)
- Mischaracterization (Dorsey trap — great product mistaken for moat)
- Already priced in
- Substitute technology emerging
- Manager dependency

ATTACKS ON NEW FRONTIER:
- Imitation hollow (imitators winning, originator losing)
- Frontier saturating
- TAM overestimated
- Defensibility against later large entrants

ATTACKS ON BOTTLENECK:
- Substitution emerging (downstream in-house alternative)
- Geopolitical exposure
- Technology obsolescence
- Competitor capacity expansion

Each attack must:
1. State a specific concern (not vague worry)
2. Cite supporting evidence with [Source: source_name] tags
3. Be falsifiable

Forbidden: generic concerns, macro hand-waving, hindsight reasoning.

Output 5 numbered attacks, each 3-5 sentences with citations.
```

Key design: attacks aligned to passed axes (rubric-aware), specific
attack types prevent dilution, forbidden patterns prevent narrative
attacks.

## 20. Stage 4 — Defender prompt

```
You are the Defender. Skeptic produced 5 attacks on [TICKER]. For 
each attack:

1. Decide DEFENDED or CONCEDED
2. If DEFENDED: provide evidence-backed counter-argument with 
   [Source: source_name] tags
3. If CONCEDED: explicitly acknowledge the attack as valid

Rules:
- DEFENDED requires citation. No defense without evidence.
- Citations must point to verifiable sources (10-K, fetched financial 
  data, recent news within 90 days)
- If you cannot find evidence, CONCEDE. Do not speculate.
- Weak defenses (citations exist but tangential) should be CONCEDED. 
  The audit catches tangential citations and downgrades them.

Output per attack:
- ATTACK: [restate]
- VERDICT: DEFENDED or CONCEDED
- IF DEFENDED: counter-argument with citations
- IF CONCEDED: which specific aspect is valid

Be honest. Defending an attack you cannot truly counter wastes the 
user's audit budget and damages reliability.
```

Key design: explicit rule that weak defenses should be conceded,
preventing citation inflation.

## 21. Stage 4 — Steward prompt

```
You are the Steward. Issue the final BUY/PASS verdict.

Inputs:
1. Stage 3 axis verdicts
2. Skeptic's 5 attacks
3. Defender's DEFENDED/CONCEDED per attack
4. Audit results: PASSED/DOWNGRADED/FAILED per defense

Apply these rules in order:

RULE 1 - Hierarchy gate (verify): 2+ axes PASSED + growth axis 
included. If not satisfied, return PASS.

RULE 2 - Defended ratio: count DEFENDED attacks where audit is 
PASSED (not DOWNGRADED or FAILED). Required: defended_ratio >= 3/5.

RULE 3 - Critical attack check: if any attack on a passed axis is 
CONCEDED, that axis is no longer reliably passed. If after CONCEDED 
attacks, fewer than 2 axes remain truly defended, return PASS.

RULE 4 - Audit downgrade impact: defenses with audit DOWNGRADED 
count as half-defended. FAILED counts as conceded.

Final:
- All four rules satisfied → BUY
- Any rule fails → PASS with specific reason

Output (JSON):
{
  "verdict": "BUY" or "PASS",
  "pass_reason": "..." (if PASS),
  "defended_ratio": "X/5",
  "axes_remaining_strong": [...],
  "summary": "2-3 sentences"
}
```

Key design: binary output aligned with Commitment 6; CONCEDED
attacks invalidate axis claims (Rule 3) so adversarial review has
real teeth.

## 22. Operational notes

**Model selection:** Stage 3 can run on local 7B-13B models (Llama
3.1 8B class) for cost reasons. Stage 4 benefits from larger models
(local 30B+ or Claude/GPT API) due to citation reasoning complexity.

**Language:** All prompts in English. Audit trail must be uniform.
User-facing output translation (ko/en/ja/zh) happens after Steward
verdict and only on summary text.

**Prompt versioning:** Constitution version is recorded with every
Stage 4 ledger record. When prompts change, the version bump goes
into the ledger. Future backtests can attribute outcomes to
constitution versions.

---

# Part VI — Implementation guidance

## 23. Work order

The original handoff proposed starting with the graph database. That
is the wrong starting point under the redefined system. The graph is
useful only after a pool of rubric-passing candidates exists, and
that pool requires a working screening pipeline, which requires the
constitution to be formalized first. The constitution is now
formalized in Parts III-V of this document. The remaining work
order:

1. **Implement Stage 2 quantitative pre-filter.** Build the proxy
   computations from Section 15 against Finnhub, EDGAR, DART, and
   FRED feeds already in the codebase. Implement multi-segment 30%
   filter. Output: ticker → axis-PASS/FAIL/NEED_LLM map. Estimated
   1-1.5 days at user's throughput.

2. **Implement Stage 3 light LLM screening.** Wire the prompt
   template from Section 18 to the existing LLM backend abstraction.
   Output: ten-to-fifteen Stage 4 candidates from a fifty-to-hundred
   shortlist. Estimated 0.5-1 day.

3. **Modify Stage 4 MAFIS engine.** Replace existing Skeptic/
   Defender/Steward prompts with Sections 19-21. Update tests to
   verify axis-aligned attacks are produced. Preserve existing audit
   infrastructure. The 475 existing tests must pass; new tests cover
   constitutional alignment. Estimated 1 day.

4. **Calibrate on small universe.** Take a known list of 30-50
   tickers (chosen for axis diversity, not user preference). Run the
   full pipeline. Compare system output to user's intuitive
   judgments. Where the system disagrees with the user, examine
   both: sometimes the user's intuition was sloppy; sometimes the
   constitution needs refinement. This calibration is the most
   important single activity in the cycle. Estimated 2-4 weeks of
   distributed effort, not concentrated coding time.

5. **Expand universe.** Once calibrated, scale to S&P 500 + KOSPI
   200. Infrastructure concerns (rate limits, caching, LLM cost
   budget) become primary. Estimated 0.5-1 day for scaling, ongoing
   monitoring.

6. **Build value chain global graph (Stage 5).** Now that survivors
   exist, position them on a global industry graph. The graph DB
   choice (Kuzu / NetworkX-on-SQLite / Neo4j) becomes relevant here.
   Estimated 2-3 days.

7. **Wire HRP (Stage 6).** Use riskfolio-lib's HRP function on the
   price-return matrix of survivors. Add post-hoc value-chain
   adjustment. Apply 1%-30% bounds. Compare against user's existing
   positions, emit incremental rebalancing recommendations. Estimated
   0.5 day.

8. **Modify tip channel.** Decouple tip ingestion from analysis
   triggering per Section 7. Implement tip log, ticker-mention
   annotation lookup, gap-analysis report. Estimated 0.5 day.

Total coding effort at user's demonstrated throughput (4 days for
MAFIS plus kor_based_jap): roughly 6-10 days of part-time work, plus
2-4 weeks of distributed calibration. Realistic full-system
availability: 1-1.5 months from start.

## 24. What this document does not decide

Several questions remain open and should not be settled until the
small-universe calibration produces evidence:

- Exact thresholds may need adjustment. The 5pp ROIC advantage,
  40% customer concentration, 5x downstream dependency ratio, HHI
  2500 — all of these are starting points. Calibration data may
  recommend changes.
- Stage 3 shortlist size (currently planned as 50-100) depends on
  LLM cost budget and user latency tolerance.
- The exact form of axis-aware Skeptic attack distribution (3-2 for
  two axes, 2-2-2-1 for three) may need adjustment based on whether
  this distribution produces useful signal.
- Whether US and Korean tickers run through the same pipeline or
  separate pipelines. Data sources differ (DART vs EDGAR vs Finnhub),
  reporting cadences differ. Combining at Stage 3 may introduce
  noise.
- Re-screening frequency (currently planned as weekly). Daily may be
  unnecessary for a long-horizon investor; monthly may suffice and
  significantly reduce LLM costs.

These are deliberately deferred to post-calibration because their
defaults are sensitive to the constitution's exact application and
should be adjusted with empirical data.

## 25. Closing note

This document was produced through a clarifying dialogue that
repeatedly revealed the user's actual intentions only by forcing
articulation. The original handoff was not wrong about MAFIS as it
exists; it was wrong about MAFIS as it should become. This document
is the correction and the constitution combined.

The most important moments in the dialogue were the points where the
user's commitment was tested. Tesla 2010-2015 — the user accepted
that the system would PASS such companies. The Dreamer module — the
user recognized it as a self-deception channel and rejected it.
Conviction levels — the user recognized that simplicity matched
master investors and rejected the complexity. Each test strengthened
the constitution because each test demanded that the user own the
trade-offs.

The next session should read this document in full, then begin with
Step 1 of Section 23. Resist the temptation to start with the graph
database. The constitution comes first, and the constitution is now
written.

If something during implementation reveals that the constitution is
wrong, that revelation is itself valuable. Constitutional change is
allowed, but only as an explicit version bump, not as a per-ticker
exception. The system cannot help the user beat his own
self-deception if the constitution becomes negotiable in the moment.

The user has chosen to externalize his willpower into this system.
The system's first job is to remember that, even when the user
forgets.
