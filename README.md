# MAFIS — Wise Investor System

**Local-first, multi-agent equity research crew** for long-term
fundamental investment decisions. Runs entirely on your machine; no
cloud LLM API spend.

**Usage guide**:
[한국어](docs/usage/ko.md) ·
[English](docs/usage/en.md) ·
[日本語](docs/usage/ja.md) ·
[简体中文](docs/usage/zh.md)

Design doc: [design-v2.2.md](design-v2.2.md)
Formal MVP evaluation: [docs/MVP_EVALUATION.md](docs/MVP_EVALUATION.md)

---

## What it does

Feed a ticker in. A 6-agent crew produces a cited research note in
~15–20 minutes:

```
Economist → Analyst → Valuer → Skeptic → Defender → Steward
                                          │           │
                          (debate round) ─┘           └─ BUY / HOLD / PASS + conviction
```

Every number in the report traces to a Python-computed source
(Finnhub/FRED/DART/SEC EDGAR). A deterministic Python audit downgrades
any verdict that violates the discipline matrix — the LLM can't
overclaim. A separate citation-grounding audit flags any `[Source:
edgar.*]` citation whose number doesn't actually appear in the
retrieved 10-K passage.

Supported markets:
- **US equities** via Finnhub + SEC EDGAR (via direct ChromaDB RAG)
- **Korean equities** via OpenDART (KRX 6-digit codes; `.KS` / `.KQ`
  suffixes stripped automatically)

---

## Current state (Phases 1–4)

| Phase | Status | What's in it |
|-------|--------|--------------|
| Phase 1 MVP | ✅ Complete | 5 agents + quality metrics — formal GO verdict |
| Phase 2 | ✅ 98% | Economist + Steward + Defender + debate round + 3-layer audit + portfolio SQLite + auto-onboarding |
| Phase 3 | ✅ 98% | 3-Tier registry + SEC EDGAR RAG + DART + chain alerts + pre-filter stages 1–3 + dedup ledger |
| Phase 4 | 🟡 75% | Paper trading ledger + auto-record on crew completion + regression-diff tool |
| LLM backend abstraction | ✅ | Pluggable backends (Ollama / OpenAI-compat / MLX / llama.cpp) + per-agent model+sampling routing via `config/agent_models.yaml` |

**Sampling policy**: each agent uses its model's published recommended
sampling (e.g. Qwen 2.5 → `temperature=0.7`/`top_p=0.8`). Same prompt
twice can produce different outputs. See
[`docs/llm_backends.md`](docs/llm_backends.md) for the full policy
discussion and how to opt back into deterministic mode per-agent.

---

## Setup

### Requirements

- Python 3.13+
- [Ollama](https://ollama.com/) running locally
- Finnhub API key (free; [finnhub.io](https://finnhub.io/))
- FRED API key (free; [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org/apikeys))
- OpenDART API key (free, for Korean stocks; [opendart.fss.or.kr](https://opendart.fss.or.kr/mngInfo/mngInfoMain.do))
- Telegram bot (optional, for push notifications)

### 1. Install Ollama + models (default backend)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

# 16k context variants used by the crew
ollama pull qwen2.5:7b-16k
ollama pull llama3.1:8b-16k
```

Want a different backend? See
[**Alternative LLM backends**](docs/llm_backends.md) — MAFIS supports
Apple Silicon (MLX), GGUF (llama.cpp), and any OpenAI-compatible
server (vLLM, LM Studio, mlx_lm.server, …) without changing the
agent code.

### 2. Configure `.env`

Copy `.env.example` to `.env` and fill in:

```bash
FINNHUB_API_KEY=...
FRED_API_KEY=...
DART_API_KEY=...              # required only for Korean tickers
TELEGRAM_BOT_TOKEN=...        # optional
TELEGRAM_CHAT_ID=...          # optional
```

### 3. Python env

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 4. Verify

```bash
python scripts/verify_env.py      # API keys + Ollama reachable
pytest                            # should report 750+ passed
```

---

## Daily workflow

### Add a new ticker (60-min hand-authoring → 3-min auto-draft)

```bash
python scripts/onboard_ticker.py AMD --tier 2 --notes "GPU peer of NVDA"
```

This pulls Finnhub profile + peers, downloads the latest 10-K,
indexes it into ChromaDB, drafts a value chain brief via Qwen, and
registers the ticker in `config/tickers.yaml`. Output:
`docs/value_chains/AMD.draft.md` — review the **Vulnerable links**
section, then:

```bash
mv docs/value_chains/AMD.draft.md docs/value_chains/AMD.md
```

Korean tickers work the same way — the dispatcher detects 6-digit
codes and routes through DART:

```bash
python scripts/onboard_ticker.py 005930 --tier 1  # Samsung Electronics
```

### Run the full crew

```bash
python scripts/run_crew.py NVDA                   # US
python scripts/run_crew.py 005930                 # Korean
```

Output:
- `reports/<SYMBOL>_YYYYMMDD_HHMM.crew.md` — six-section report +
  audit block
- `reports/<SYMBOL>_...meta.txt` — timing / char counts / models used
- Auto-inserted row in `data/portfolio.sqlite` paper-trades table
- Optional Telegram push of the Korean summary

### Inspect the portfolio

```bash
python scripts/portfolio_cli.py add NVDA --shares 10 --cost 5000 --tier 1
python scripts/portfolio_cli.py weights                  # live Finnhub quotes
python scripts/portfolio_cli.py gap NVDA --low 3 --high 5
```

### Track paper-trade P&L over time

```bash
python scripts/paper_ledger.py list                      # all recorded verdicts
python scripts/paper_ledger.py returns                   # mark-to-market
python scripts/paper_ledger.py summary                   # win rate, audit effect
```

### Monitor news → chain alerts

```bash
# One-off scan (prints alerts; won't fire duplicates when --dedup)
python scripts/scan_chain_alerts.py --dedup --hops 2

# Cron-friendly (with Telegram push)
0 9-16 * * 1-5  cd ~/MAFIS && /path/to/.venv/bin/python \
    scripts/scan_chain_alerts.py --dedup --telegram \
    >> /var/log/mafis_alerts.log 2>&1
```

### Promote Tier 3 → Tier 2 based on news activity

```bash
python scripts/prefilter_scan.py --graph-context --semantic
```

Runs Stages 1 (keyword), 2 (value-chain context), and 3 (Qwen
materiality filter) against the news pool and recommends promotions.

### Validate prompt / model tweaks didn't regress quality

```bash
python scripts/regression_compare.py \
    reports/NVDA_20260424_1557.crew.md \
    reports/NVDA_20260425_0900.crew.md \
    --fail-on-regression
```

---

## Architecture

```
src/wise_investor/
├── agents/                  # crew: analyst, valuer, skeptic, defender, steward, economist
│   ├── steward_audit.py     # discipline matrix + speculative-language + Defender-aware
│   └── runner.py            # pre_gather_facts dispatcher (US → Finnhub, KR → DART)
├── data/
│   ├── finnhub.py           # US fundamentals
│   ├── dart.py              # Korean fundamentals (OpenDART)
│   ├── dart_facts.py        # KR → crew facts adapter (KRW→USD via FRED)
│   ├── fred.py              # macro snapshot (Economist)
│   └── cross_validate.py
├── rag/
│   ├── edgar.py             # SEC EDGAR downloader + cache
│   ├── sections.py          # Business / Risk Factors / MD&A / Quant Market Risk extractor
│   ├── index.py             # ChromaDB persistent store
│   └── integration.py       # crew pre_gather hook
├── geopolitics/
│   ├── gdelt.py             # GDELT DOC 2.0 client
│   ├── google_news.py       # RSS parser
│   └── snapshot.py          # per-symbol geopolitical context
├── alerts/
│   ├── chain_alerts.py      # value-chain graph × news → target alerts
│   └── ledger.py            # SQLite dedup + cooldown
├── filters/
│   ├── pre_filter.py        # Stages 1 (keyword) + 2 (graph context)
│   └── semantic.py          # Stage 3 Qwen materiality filter
├── onboarding/
│   ├── brief_generator.py   # Finnhub + 10-K + geo → Qwen-drafted value chain brief
│   └── tickers_yaml.py      # 3-Tier registry CRUD
├── portfolio/
│   └── store.py             # positions + sizing-gap helper
├── paper_trading/
│   ├── ledger.py            # paper_trades table + performance metrics
│   └── report_parser.py     # parse Steward verdict + audit flag from crew report
├── regression/
│   └── compare.py           # structured crew-report diff tool
├── value_chain/
│   ├── graph.py             # NetworkX-backed typed DiGraph
│   └── parser.py            # docs/value_chains/*.md → graph
├── quality/
│   ├── metrics.py           # 6 automated quality scores
│   └── citation_audit.py    # edgar.* grounding + Skeptic mandate audit
└── notify/
    └── telegram.py

scripts/                     # CLI entry points for every component above
docs/value_chains/           # hand-curated + auto-drafted briefs (*.md vs *.draft.md)
data/                        # portfolio.sqlite, chroma/, edgar_cache/, facts_cache/
tests/                       # 750+ tests (offline; live ones marked -m network)
```

---

## Core principles

- **Local-first, API-last**: Phase 1 runs with $0 LLM spend. Finnhub /
  FRED / GDELT / DART are free public APIs.
- **LLM is judgment, Python is calculation**: every dollar value, ratio,
  and growth rate is computed by `src/wise_investor/tools/` or `data/`
  and fed to the LLM as prepared facts. The LLM synthesizes narrative,
  never arithmetic.
- **Sampling follows model recommendations**: each agent uses the
  sampling profile published by its model author (Qwen 2.5: 0.7/0.8;
  Llama 3.x: 0.7/0.9; Qwen3 thinking: 0.6/0.95/min_p=0). Two runs of
  the same crew may differ; the audit + citation system enforce
  within-run consistency, not run-to-run reproducibility. Opt back
  into deterministic mode per agent in
  [`config/agent_models.yaml`](docs/llm_backends.md#re-enabling-deterministic-output).
- **Multi-layer audit**: discipline matrix (verdict vs labels) +
  speculative-language detector + Defender-aware correction + edgar
  citation grounding + Skeptic mandate compliance. The LLM can emit
  any narrative; Python enforces the rules.
- **Paper trading before real trading**: every Steward verdict is
  automatically recorded with entry price. `paper_ledger.py summary`
  tells you whether BUY verdicts actually outperform PASS verdicts
  over time — the only objective answer to "is this system useful?".

---

## Telegram push (optional)

1. Create a bot with [@BotFather](https://t.me/BotFather) → copy the
   token.
2. Send any message to your bot (creates the chat).
3. Visit `https://api.telegram.org/bot<TOKEN>/getUpdates` → copy the
   `chat.id`.
4. Add to `.env`:
   ```
   TELEGRAM_BOT_TOKEN=...
   TELEGRAM_CHAT_ID=...
   ```
5. `run_crew.py` auto-pushes a Korean summary; `scan_chain_alerts.py
   --telegram` pushes chain alerts.

No configuration → silent skip, no errors.

---

## Limitations

- Korean-ticker crew runs share the English agent prompts; the
  Analyst will produce English analysis of Korean financials. A
  follow-up will branch the prompts by source country.
- Value chain graph auto-update from 10-K text is not yet
  implemented. Briefs are either hand-curated or onboarding-
  drafted and then hand-reviewed.
- No paper-trade position sizing — the ledger records Steward
  verdicts only; actual position sizing per trade is manual.
- No OpenClaw integration (design §8.1); Telegram covers the
  equivalent role.

See [docs/MVP_EVALUATION.md](docs/MVP_EVALUATION.md) for the
Phase 1 formal evaluation and Phase 2+ priorities.
