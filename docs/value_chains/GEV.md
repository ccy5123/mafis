# GEV — Value Chain Brief (Phase 1C generalization test)

Last updated: 2026-04-23
Author: manual draft
Intended reader: Analyst / Valuer / Skeptic agent prompt context (not end-user facing)

GE Vernova (NYSE: GEV) was spun off from GE in April 2024 as a pure-play
energy-equipment company. Three segments: **Power** (gas turbines, hydro,
nuclear services), **Wind** (onshore + offshore), **Electrification**
(grid, power conversion, solar inverters, EV charging infra). This brief
captures the value chain the Analyst agent should reason over. Numbers
are order-of-magnitude only; the Analyst must pull precise figures via
tools and cite them.

## Peer Override

Finnhub's `/stock/peers` for GEV returns small-cap / pre-commercial
companies (Bloom Energy, NuScale, Babcock & Wilcox, Forgent, PSIX) whose
multiples are non-comparable (>500x EV/EBITDA or N/A for both ratios).
The real peer cohort is multi-national conglomerates in gas turbines,
wind, and grid equipment — none of which Finnhub auto-matches. The
tickers below will be merged into the peer table by the analysis
pipeline.

- SMNEY — Siemens Energy ADR (direct full-segment competitor)
- ETN — Eaton Corporation (electrification peer)
- ABBNY — ABB ADR (automation + electrification)
- VWDRY — Vestas Wind Systems ADR (wind-only pure play)
- HTHIY — Hitachi ADR (grid & power)

## Upstream — Suppliers

### Raw materials

- **Steel, copper, aluminum** — standard industrial commodity exposure.
  Grid equipment and wind tower components are copper-heavy; gas turbines
  are specialty steel.
- **Nickel superalloys** — high-temperature gas turbine blades. Sole
  qualified suppliers are few (PCC, Doncasters, Haynes); lead times 12-18
  months.
- **Rare earth magnets (neodymium-iron-boron)** — wind turbine permanent
  magnet generators. **China controls ~85% of processed rare earths** —
  the structural upstream chokepoint.
- **Carbon fiber** — offshore wind blades. Hexcel, Toray supply limited.

### Specialty components

- **Semiconductors** — Silicon Carbide (SiC) power electronics for grid
  inverters and HVDC transmission. Wolfspeed, STMicro, Infineon, Onsemi.
  Constrained during 2023-2024 capacity ramp.
- **Bearings** — SKF, Schaeffler for turbine main bearings; switching
  cost high.
- **Transformer cores** — grain-oriented electrical steel (GOES).

### Internal supply risk

- **Haliade-X offshore turbine platform** — if platform-level defects
  emerge (cf. Siemens Gamesa's Spain fleet warranty crisis in 2023-2024
  that cost Siemens Energy billions), single-product exposure is severe.

## Peers — Direct and adjacent competition

| Peer | Ticker | Overlap | Threat level (2026 H1) |
|------|--------|---------|-----------------------:|
| Siemens Energy | ENR.DE | Gas turbines + wind (direct) | High — full-segment competitor |
| Vestas | VWS.CO | Onshore wind | Medium-high — market leader in wind |
| Mitsubishi Power (MHI) | 7011.T | Gas turbines | High — esp. Asia markets |
| Hitachi Energy | (ABB spin) | Grid & transmission | High — direct in Electrification |
| Schneider Electric | SU.PA | Grid, power conversion | Medium — commercial + industrial focus |
| ABB | ABBN.SW | Electrification | Medium — automation-adjacent |
| Eaton | ETN | Electrical distribution | Medium |
| Goldwind | 2208.HK | Onshore wind (China) | Export-limited but cheap |
| Mingyang Smart Energy | 601615.SS | Offshore wind | Export-limited |

Most asymmetric threats: **Chinese OEMs in wind** (cost gap is large, but
US/EU trade barriers mostly offset in OECD markets) and **Siemens Energy
in gas turbines** (they have the same platform-risk exposure that makes
wins and losses bimodal).

## Downstream — Customers

### Revenue concentration (from most recent 10-K; Analyst must verify)

- No single customer >10% of revenue per 10-K disclosure (unlike NVDA's
  top-4 concentration).
- Customer base is fragmented across utilities, IPPs, governments.

### Customer categories

- **Utilities (US and international)** — NextEra, Iberdrola, Duke,
  Dominion, Southern, ENEL, EDF. Multi-year service contracts for gas
  turbines after the 20-30yr fleet reaches mid-life.
- **Independent Power Producers (IPPs)** — AES, Vistra, Calpine.
- **Hyperscale / data center operators** — **the new 2024-2026 demand
  story**. Microsoft, Google, Amazon, Meta are contracting for nuclear
  SMRs (via partners), gas turbine capacity, and grid equipment to
  support AI data center power needs. This is the segment inflecting.
- **Offshore wind developers** — Ørsted, Equinor, RWE, Avangrid.
  Bloodied segment — margin concessions widespread in 2023-2024.
- **Governments / sovereign programs** — UAE (Masdar), Saudi NEOM, India
  renewable targets.
- **Nuclear services** — existing PWR/BWR fleet extensions + SMR
  partnerships (BWRX-300 with Hitachi-GE).

## Infrastructure / Regulatory dependencies

- **US Inflation Reduction Act (IRA) tax credits** — solar, wind, nuclear
  PTC/ITC. **Uncertain under Trump administration** (2025-2028); partial
  modification possible, full repeal unlikely but non-zero. GEV Wind
  segment economics depend on the PTC.
- **EU RePowerEU** — €300B grid modernization program through 2030.
  Electrification segment tailwind.
- **Grid interconnection queues** — FERC-administered in US, 2+ year
  backlogs for large additions. Limits deployment velocity regardless
  of equipment availability.
- **Section 301 tariffs** — US tariffs on Chinese wind equipment limit
  competitive pressure but also limit component sourcing flexibility.
- **Offshore wind permitting** — BOEM in US, environmental litigation
  has delayed Vineyard Wind and Revolution Wind.

## Vulnerable links (Skeptic's attack surface)

When the Skeptic attacks a Bull thesis on GEV, these are the highest-
leverage points:

1. **Offshore wind losses** — GEV has taken material charges on Vineyard
   Wind and the Haliade-X platform; offshore segment has run unprofitable.
   Bull story depends on turn-around that has not yet arrived.
2. **Gas turbine orders = AI capex proxy** — the step-change in gas
   turbine backlog (2024-2025) reflects hyperscaler power procurement.
   If AI capex decelerates (scaling laws plateau, hyperscaler utilization
   saturates), this demand evaporates.
3. **IRA tax-credit uncertainty** — Wind segment IRR models depend on
   PTC. Modifications to the IRA (or interpretive changes by Treasury)
   compress project economics.
4. **Rare-earth supply chain (China concentration)** — PMDD generator
   designs depend on NdFeB magnets. Geopolitical cutoff is a severe
   low-probability / high-impact event.
5. **Platform defect risk** — Siemens Energy / Siemens Gamesa charged
   billions against wind turbine fleet defects in 2023. GEV has the
   Haliade-X and the Cypress onshore platform exposed. One systemic
   flaw = quarters of warranty expense.
6. **Reverse DCF implied growth** — if reverse_dcf returns an implied
   FCF growth rate consistent with sustained hyperscaler capex for 10
   years, ask whether capital goods have historically sustained such a
   rate, and what a cyclical downturn (1-2 year order pause) does to it.
7. **Grid interconnection bottleneck** — not GEV's problem directly but
   caps the addressable market. Equipment that cannot be deployed is not
   sold.
8. **Gas segment may be at cyclical peak** — gas turbine orders declined
   for 20+ years before the 2024 inflection. Mean reversion risk.
9. **Electrification margin is the equity story** — if Electrification
   margins plateau or decline (due to commodity steel/copper inflation
   or Chinese competition bleeding into non-wind markets), the premium
   multiple becomes hard to defend.

## Known unknowns (do not pretend to know)

- Segment-level quarterly orders breakdown by customer type (utility vs
  hyperscaler vs government).
- Exact gas turbine backlog value, duration, and pricing.
- Offshore wind margin trajectory toward 2027 (guidance framed
  conservatively; actual cadence uncertain).
- Electrification pricing power vs Schneider/Eaton/ABB.
- Impact of natural gas price on utility customer decisions between gas
  and renewables.
- Effect of Trump-administration IRA modifications (analysts model 0-30%
  credit-value compression scenarios; true outcome unknown).

Analyst should mark these as "data gap" in reports rather than estimate.
