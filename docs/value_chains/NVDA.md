# NVDA — Value Chain Brief (Phase 1B manual draft)

Last updated: 2026-04-22
Author: manual draft, to be validated/revised after each quarterly earnings call
Intended reader: Analyst agent prompt context (not end-user facing)

This document captures the upstream / peer / downstream / infrastructure map
around NVIDIA that the Analyst agent should reason over. It is intentionally
dense and biased toward *dependencies and vulnerabilities* rather than
marketing narrative. Numbers stated here are rough order-of-magnitude only;
the Analyst must pull precise figures via FMP tools and cite them.

## Peer Override

Finnhub's `/stock/peers` for NVDA already returns a solid mega-cap tech
cohort (GOOGL, MSFT, META, AMD, and similar). No override needed — the
auto-peer set is adequate for NVDA valuation comparisons. This section
is kept as a placeholder so the override mechanism is discoverable; if
a future earnings cycle surfaces a peer gap, add tickers here.

- (none)

## Upstream — Suppliers

### Chip fabrication

- **TSMC** — single external foundry for all leading-edge (N3, N4P) and
  expected N2 in 2026+. All Blackwell, Hopper, and Grace CPUs fabricated at
  TSMC. This is the single largest concentration risk in the entire value
  chain. Source: NVDA 10-K risk factors, TSMC capacity disclosures.
- **TSMC CoWoS advanced packaging** — distinct constraint from wafer
  capacity. CoWoS-S/L/R packages AI accelerators with HBM stacks. CoWoS
  shortage was the #1 supply constraint through 2024 and eased in 2025.

### Memory

- **SK hynix** — primary HBM3e supplier; first to qualify for Blackwell.
- **Samsung** — HBM3e qualification progress; incumbent but late to HBM4.
- **Micron** — third HBM supplier; smaller share but growing.

HBM pricing is effectively oligopolistic. Any HBM fab disruption (fire,
earthquake, labor action) bottlenecks NVDA shipments within one quarter.

### Design tooling

- **Synopsys, Cadence** — EDA software; monopolistic effectively, but cost
  is small fraction of NVDA's operating expense.
- **Arm** — IP licensing for Grace CPU. Royalty structure affects margin.

### Lithography (one layer up from fabs)

- **ASML** — EUV scanners; shipped only to TSMC/Samsung/Intel. Indirect
  dependency but any ASML production slowdown propagates.

### System assembly

- **Foxconn, Wiwynn, Quanta, Supermicro** — DGX/HGX/MGX systems
  integration. Not technology-critical but volume-critical during ramp.

## Peers — Direct competition in AI accelerators

| Peer | Ticker | Product | Threat level (2026 H1) |
|------|--------|---------|-----------------------:|
| AMD | AMD | MI300X, MI325X, MI350 roadmap | Medium — shipping at scale, software (ROCm) maturing |
| Intel | INTC | Gaudi 3, Falcon Shores | Low-medium — credible but trailing |
| Broadcom | AVGO | Custom ASICs for Google/Meta | High — hyperscalers shifting to in-house designs |
| Google | GOOGL | TPU v5/v6 internal + GCP | Internal threat, not sold commercially |
| AWS | AMZN | Trainium 2/3, Inferentia | Internal, increasing share of AWS AI workloads |
| Meta | META | MTIA | Internal, small but growing |
| Huawei | (private) | Ascend 910B/C | China-market substitute as NVDA is export-banned |

The most asymmetric peer threat is **custom ASICs from hyperscalers**. If
Google TPU, AWS Trainium, Meta MTIA collectively absorb 30%+ of their own AI
training/inference workloads, NVDA's top-4 customers become structurally
smaller buyers.

## Downstream — Customers

### Revenue concentration (from recent 10-K; Analyst must verify current numbers)

- Top 4 customers historically account for >40% of data center revenue.
- Microsoft, Meta, Amazon, Alphabet are the publicly discussed big four.
- Oracle, CoreWeave, Tesla are named sizable buyers in earnings calls.

### Customer categories

- **Hyperscale cloud** (Microsoft Azure, AWS, GCP, Oracle Cloud) — largest
  single segment; also the segment most likely to substitute with internal
  silicon.
- **Neoclouds** (CoreWeave, Lambda, Crusoe, Nebius) — GPU-as-a-service
  startups; backed by venture and infrastructure debt. Fragile in a rate-
  up or AI-demand-down scenario.
- **Enterprise AI** (financial firms, pharma, telecom) — growing but still
  small share.
- **Automotive** (Tesla, Mercedes-Benz, BYD for ADAS/autonomy compute) —
  small but strategic.
- **Sovereign AI** (UAE G42, Saudi HUMAIN, Japan METI programs, India,
  Singapore) — new category from 2024; politically durable.
- **Gaming** (GeForce RTX consumer line) — historical business, now minor
  share of revenue.
- **Professional visualization** (Quadro successor, Omniverse) — small.

## Infrastructure dependencies

- **Electricity** — AI data centers consume gigawatt-scale power. Deployment
  velocity is increasingly capped by grid interconnection, not GPU supply.
  US utilities are stating 2028-2030 timelines for new large loads.
- **Cooling** — liquid cooling vendors (Vertiv, Schneider, CoolIT) scaling
  to meet Blackwell's higher thermal design.
- **High-bandwidth networking** — InfiniBand (Mellanox, now NVDA) and
  Spectrum-X Ethernet; optical transceivers from Marvell, Coherent,
  Lumentum.
- **Site land and water rights** — ESG/community pressure rising in certain
  US states.

## Geopolitical / regulatory pressure points

- **US export controls on China** — H100/H200/Blackwell cannot legally ship
  to China. H20 (cut-down variant) was permitted then restricted then
  partially relaxed with Trump admin changes in 2025. Approximately 20-25%
  of historical data center revenue was China-exposed pre-controls.
- **Taiwan Strait risk** — Because 100% of leading-edge supply passes
  through TSMC in Taiwan, any PLA exercise/blockade/invasion is an
  existential operational disruption (not merely financial).
- **EU AI Act** — Compliance cost for high-risk AI systems; indirect effect
  on customer demand.
- **US/EU antitrust** — DOJ probe of NVDA's bundling / CUDA lock-in;
  monitor quarterly for formal action.

## Vulnerable links (Skeptic's attack surface)

When Skeptic is challenging a Bull thesis on NVDA, these are the highest-
leverage points:

1. **TSMC Taiwan single point** — supply, not demand. A week-long CoWoS
   outage could cost NVDA a full quarter of revenue.
2. **HBM supply oligopoly** — SK hynix production issues would bottleneck
   immediately.
3. **Customer concentration vs. in-house silicon** — if Google, AWS, Meta
   collectively take 40%+ of their own AI compute internal by 2028, the DC
   TAM available to NVDA shrinks even if absolute demand keeps growing.
4. **Power constraint** — AI data centers are becoming power-limited.
   Incremental GPU demand can be real while deployment capacity is capped.
5. **Software moat erosion** — ROCm and PyTorch improvements gradually
   lower CUDA switching cost. Ten-year durability of the moat is not the
   same as current moat strength.
6. **Reverse DCF implied growth** — if the Python tools return an implied
   FCF growth rate above 20%/year for 10 years, Skeptic should stress-test
   whether any semiconductor company has sustained that rate historically.
7. **Inventory cycle** — hyperscalers may over-order in a capacity panic
   (2024–2025) and under-order when normalization sets in.

## Known unknowns (do not pretend to know)

- Precise revenue share by customer (disclosed only in aggregate).
- Margin by product line (Blackwell vs Hopper vs gaming vs automotive).
- Forward CoWoS capacity allocations (TSMC does not publish).
- Exact impact of China restrictions on Q/Q revenue.

Analyst should mark these as "data gap" in reports rather than estimate.
