# MAFIS Usage Guide (English)

A walkthrough from fresh install to daily operations. For a high-level
overview see the [main README](../../README.md); for architecture see
[design-v2.2.md](../../design-v2.2.md); for the Phase 1 formal
evaluation see [docs/MVP_EVALUATION.md](../MVP_EVALUATION.md).

---

## 1. Setup (~10 min)

### 1.1 Requirements

- Python 3.13+
- [Ollama](https://ollama.com/) (local LLM runtime)
- Four free API keys (see 1.3)

### 1.2 Ollama + models

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# 16k-context models used by the crew
ollama pull qwen2.5:7b-16k
ollama pull llama3.1:8b-16k
```

### 1.3 API keys

| Service | Purpose | Link |
|---|---|---|
| **Finnhub** | US fundamentals / quotes | [finnhub.io](https://finnhub.io/) |
| **FRED** | Macro indicators (Economist) | [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys) |
| **OpenDART** | Korean fundamentals | [opendart.fss.or.kr](https://opendart.fss.or.kr/mngInfo/mngInfoMain.do) |
| **Telegram** (optional) | Push alerts | `/newbot` via `@BotFather` |

Copy `.env.example` to `.env` and fill in the keys:

```bash
cp .env.example .env
# edit .env
FINNHUB_API_KEY=...
FRED_API_KEY=...
DART_API_KEY=...            # required only for Korean tickers
TELEGRAM_BOT_TOKEN=...      # optional
TELEGRAM_CHAT_ID=...        # optional
```

### 1.4 Python environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 1.5 Verify

```bash
python scripts/verify_env.py    # checks API keys + Ollama reachability
pytest                          # should print "480 passed"
```

---

## 2. First run — NVDA end-to-end

### 2.1 Onboard the ticker (auto value-chain draft)

```bash
python scripts/onboard_ticker.py NVDA --tier 1 --notes "First target"
```

Takes ~3–5 min. What it does:

1. Pulls Finnhub profile / industry / peer list
2. Downloads the latest 10-K from SEC EDGAR and indexes it in ChromaDB
3. Fetches geopolitical news (GDELT + Google News)
4. Qwen 2.5 7B drafts an 8-heading value chain brief
5. Saves to `docs/value_chains/NVDA.draft.md`
6. Adds an entry to `config/tickers.yaml` under Tier 1

### 2.2 Review the draft (human gate, 2–3 min)

Open `docs/value_chains/NVDA.draft.md`:

- Pay particular attention to the **Vulnerable links** section — the
  Skeptic agent's primary fuel
- Entries prefixed `[?UNCERTAIN]` are LLM self-flags of low
  confidence — resolve or delete
- Once reviewed, rename to activate:

```bash
mv docs/value_chains/NVDA.draft.md docs/value_chains/NVDA.md
```

### 2.3 Run the crew

```bash
python scripts/run_crew.py NVDA
```

~15–20 min (six agents sequential). Produces:

- `reports/NVDA_YYYYMMDD_HHMM.crew.md` — six-part report plus audit
  blocks
- `reports/NVDA_YYYYMMDD_HHMM.crew.meta.txt` — per-agent timing + model
  info
- Auto-inserted row in the `paper_trades` table of `portfolio.sqlite`
- Optional Telegram push with the Korean summary

### 2.4 Korean tickers follow the same flow

```bash
python scripts/onboard_ticker.py 005930 --tier 1   # Samsung Electronics
python scripts/run_crew.py 005930
```

The DART dispatcher auto-detects 6-digit KRX codes and routes through
OpenDART for fundamentals.

---

## 3. Reading a report

### 3.1 Six-part structure

| Part | Agent | Role |
|---|---|---|
| 1 | Economist | Fed rate, CPI, FX — macro backdrop |
| 2 | Analyst | 7-section business + financial-health report |
| 3 | Valuer | PER / EV-EBITDA / reverse-DCF implied growth |
| 4 | Skeptic | Five rebuttals on the Bull thesis (Llama, different model) |
| 5 | Defender | Bull-side replies: DEFENDED or CONCEDED per rebuttal |
| 6 | Steward | Final verdict — BUY / HOLD / PASS + conviction 1–5 |

### 3.2 System Audit block (end of report)

Auto-appended after Steward. Example:

```
### System Audit — Discipline Matrix Enforcement
- Raw Steward labels: NEUTRALIZED=1, SURVIVED=1.
- Defender labels (authoritative): DEFENDED=1, CONCEDED=4.
- Steward mis-translated Defender labels.
- Effective labels: NEUTRALIZED=1, SURVIVED=4.
- Reported Verdict: BUY / Conviction: 4.
- Matrix ceiling: PASS / Conviction 1.
- VIOLATION: reported Verdict=BUY exceeds matrix ceiling PASS.
```

**How to read**: the Steward LLM issued BUY C4, but the Defender only
defended 1 of 5 Skeptic rebuttals — by the discipline matrix this
collapses to PASS C1. The Steward narrative is preserved verbatim,
but downstream consumers (paper ledger, Telegram summary) use the
audit-corrected verdict.

### 3.3 Citation Grounding block

Scans every `[Source: edgar.*]` citation to verify the cited number
actually appears in the retrieved 10-K passage. Flags hallucinations:

```
## System Audit — Citation Grounding
1 ungrounded claim(s):
- Claim '15-20%' in section 'mdna_highlights' not found in retrieved passages.
```

Meaning: the LLM attached a plausible-looking citation to a fabricated
number — treat as unverified.

### 3.4 Verdict semantics

- **BUY C5**: highest confidence, all 5 Skeptic rebuttals DEFENDED.
  Sizing 5–8%.
- **BUY C3–4**: Bull majority with some concessions. Sizing 2–5%.
- **HOLD C2**: balanced. Keep existing exposure, no new add.
- **HOLD C1**: close call — effectively a soft PASS.
- **PASS C1**: Bear majority. No position.

---

## 4. Daily operations

### 4.1 Portfolio tracking

```bash
# Add a position
python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1

# List everything
python scripts/portfolio_cli.py list

# Current weights at live quotes
python scripts/portfolio_cli.py weights

# Compare a Steward sizing band to current weight
python scripts/portfolio_cli.py gap NVDA --low 3 --high 5
# Output: "Already at 4.2% (suggestion 3.0-5.0% — within band, no action)"
```

### 4.2 Paper-trade performance tracking

Crew runs auto-insert rows into `paper_trades`. After a few weeks:

```bash
python scripts/paper_ledger.py list                 # all recorded verdicts
python scripts/paper_ledger.py returns              # mark-to-market
python scripts/paper_ledger.py summary              # win rates, audit effect
```

Sample `summary`:

```
By verdict
  BUY:  n=5  avg=+4.20%  win rate=60.0%
  HOLD: n=2  avg=+1.50%
  PASS: n=3  avg=-2.80%

By conviction
  C4: n=3  avg=+6.10%
  C2: n=2  avg=-0.30%

Audit effect (original BUY verdicts)
  BUYs that cleared audit: +7.50%
  BUYs downgraded by audit: -3.20%
```

The last block is the key validation: **if BUYs the audit downgraded
actually underperformed, the Python discipline layer is earning
alpha** — an objective signal the system is working.

### 4.3 Chain alerts (news → position cascades)

When any node in the value chain graph (e.g. TSMC, ASML, Siemens)
appears in the news, every Tier 1 target reachable within N hops
gets an alert:

```bash
# One-off scan
python scripts/scan_chain_alerts.py --dedup --hops 2

# Hourly cron during market hours
0 9-16 * * 1-5  cd ~/MAFIS && /path/to/.venv/bin/python \
    scripts/scan_chain_alerts.py --dedup --telegram \
    >> /var/log/mafis_alerts.log 2>&1
```

`--dedup` uses the SQLite ledger so the same (target, node, title)
does not alert twice within 48 h.

### 4.4 Tier 3 promotion recommendations (pre-filter)

Surface Tier 3 names that deserve a closer look based on news
activity:

```bash
python scripts/prefilter_scan.py --graph-context --semantic
```

- Stage 1: keyword match (symbol / company name)
- Stage 2: value-chain context (mentions of suppliers / peers of Tier 1 targets)
- Stage 3: Qwen relevance filter (drops immaterial routine PR)

### 4.5 Regression-safe prompt / model tweaks

After changing a prompt or bumping the model, compare old vs new
report on structural quality:

```bash
python scripts/regression_compare.py \
    reports/NVDA_20260424_1715.crew.md \
    reports/NVDA_20260425_0900.crew.md
```

Classifies every metric (citation_rate, refusal_count, audit
violation counts, edgar citation counts, verdict, conviction) as
IMPROVED / REGRESSED / NEUTRAL. Use `--fail-on-regression` in CI.

---

## 5. Common issues

### 5.1 `FINNHUB_API_KEY not set`

Ensure `.env` is in the project root and the key has no surrounding
whitespace.

### 5.2 SEC EDGAR returns 403

SEC fair-use policy requires the User-Agent to include an email. The
default is `MAFIS research ccy5123ccy@gmail.com` — change to your
own in `src/wise_investor/rag/edgar.py`:

```python
USER_AGENT = "YourName research your@email.com"
```

### 5.3 DART returns `status: 013`

Usually means `corp_code` mismatch or no filing for that year.
Debug:

```bash
python scripts/probe_dart.py 005930 --year 2024 --dump
```

### 5.4 Crew takes > 20 min

- Check `ollama ps` for GPU / CPU utilization
- Ollama may have run out of memory — `ollama stop && ollama serve`
- Skeptic (Llama) ↔ Defender/Steward (Qwen) model swap costs
  30 s – 2 min; this is expected

### 5.5 Chain alerts empty

```bash
# Rebuild the value chain graph
python scripts/build_value_chain_graph.py

# Manually probe one symbol
python scripts/probe_geopolitics.py NVDA --timespan 3days
```

---

## 6. Directory map

```
MAFIS/
├── src/wise_investor/         # main package
│   ├── agents/                # 6 crew agents
│   ├── data/                  # Finnhub / DART / FRED clients
│   ├── rag/                   # SEC EDGAR 10-K ChromaDB
│   ├── geopolitics/           # GDELT + Google News
│   ├── alerts/                # chain alerts + dedup ledger
│   ├── filters/               # pre-filter 3 stages
│   ├── onboarding/            # ticker onboarding automation
│   ├── portfolio/             # position SQLite
│   ├── paper_trading/         # Steward verdict performance tracker
│   └── regression/            # report diff tool
├── scripts/                   # CLI entry points
├── docs/value_chains/         # manual + auto-drafted briefs
├── config/tickers.yaml        # 3-Tier registry
└── data/                      # portfolio.sqlite, chroma/, edgar_cache/, facts_cache/
```

---

## 7. Core principles

- **Local-first, API-last**: zero external LLM spend. Finnhub / FRED /
  GDELT / DART are all free public APIs.
- **LLM is judgment, Python is calculation**: every number in the
  report comes from a Python tool. The LLM only writes narrative.
- **Reproducibility**: `temperature=0`, `seed=42` → byte-identical
  agent outputs on the same facts cache.
- **Multi-layer audit**: discipline matrix + speculative-language
  detector + Defender-aware correction + citation grounding + Skeptic
  mandate compliance.
- **Paper-trade before real money**: weeks of `paper_ledger.py
  summary` output is the only objective answer to "does this system
  make money?".

---

## 8. More

- [Design document](../../design-v2.2.md)
- [MVP formal evaluation](../MVP_EVALUATION.md)
- [GitHub repo](https://github.com/ccy5123/mafis)

File issues on GitHub when something breaks.
