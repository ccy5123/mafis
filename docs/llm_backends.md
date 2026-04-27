# LLM Backends & Per-Agent Model Routing

MAFIS does its analysis through six agents (Economist, Analyst, Valuer,
Skeptic, Defender, Steward) plus a few utility tools (semantic news
filter, tip classifier, value-chain brief generator, report
translator). Every one of those LLM calls flows through a pluggable
**backend layer** — you pick the runtime that fits your hardware,
and a YAML config picks the model + sampling per agent.

This document covers:

1. [The reproducibility policy change](#reproducibility-policy-change)
2. [Picking a backend](#picking-a-backend)
3. [Hardware-aware model recommendations](#hardware-aware-model-recommendations)
4. [Backend setup recipes](#backend-setup-recipes)
5. [Per-agent customization (`config/agent_models.yaml`)](#per-agent-customization)
6. [Thinking-mode models (Qwen3, DeepSeek-R1)](#thinking-mode-models)
7. [Re-enabling deterministic output](#re-enabling-deterministic-output)
8. [How `regression_compare.py` and paper trading change](#regression_compare-and-paper-trading)
9. [Troubleshooting](#troubleshooting)

---

## Reproducibility policy change

**Old policy (pre-Phase-5)**: every LLM call was pinned to
`temperature=0, seed=42`. Two crew runs over the same facts cache
produced byte-identical reports.

**New policy**: each agent uses its model's **published recommended
sampling**. Two runs over the same inputs may differ.

Why the change:

- Modern instruction-tuned models (Qwen3, Llama 3.x, DeepSeek-R1) are
  trained at non-zero temperature and degrade noticeably at
  `temperature=0`, particularly for long-form prose and reasoning.
- Pinning seeds across providers is a fragile abstraction — vLLM,
  llama.cpp, MLX, and Ollama implement seed semantics differently.
- The audit / citation-grounding system inside MAFIS already enforces
  internal consistency per run; reproducibility-via-pinning was
  defending the wrong invariant.

What this means in practice:

- A single re-run is a sample from a distribution, not a recovery of
  the same output. Treat `regression_compare.py` as a *delta surveyor*,
  not a strict equality check.
- Statistical conclusions about model behavior (e.g. "verdict
  stability for NVDA") need multiple runs, not one.
- If your workflow genuinely needs byte-identical re-runs (academic
  paper, regulator-style backtest), see
  [Re-enabling deterministic output](#re-enabling-deterministic-output)
  — you opt back in per agent in YAML.

The audit, citation grounding, and facts cache all keep working. They
care about within-run consistency, which is independent of run-to-run
reproducibility.

---

## Picking a backend

Set `LLM_BACKEND` in `.env`. Default is `ollama`.

| Backend | Process model | Hardware | Best for |
|---|---|---|---|
| `ollama` | Local server | Any (CPU / NVIDIA / Apple) | Default — works everywhere with one binary install |
| `openai_compat` | Remote HTTP | Any | vLLM / LM Studio / mlx_lm.server / SGLang clusters |
| `mlx` | In-process | **Apple Silicon only** | Native Metal acceleration on M-series Macs |
| `llamacpp` | In-process | Any (best with build flags) | Direct GGUF; preferred for tight CPU-only setups |

CrewAI's tool-calling loop (used by the Analyst) requires a backend
that exposes an OpenAI-shape chat completions endpoint. That means:

- **Ollama** ✅ — native bridge, default.
- **OpenAI-compat** ✅ — uses CrewAI's `hosted_vllm/` provider tag.
- **MLX** ❌ in-process — run `mlx_lm.server` and point
  `openai_compat` at it instead.
- **llama.cpp** ❌ in-process — run `python -m llama_cpp.server` and
  point `openai_compat` at it instead.

The non-Analyst utility calls (semantic filter, tip classifier,
report translator, brief generator) work natively on every backend.

---

## Hardware-aware model recommendations

These are **starting points**, not benchmarks. Run `scripts/run_crew.py`
once on a few tickers and judge for yourself.

### NVIDIA GPU 24 GB+ (RTX 4090, A100, H100)

- 30B-class instruction model (e.g. Qwen 2.5 32B, Qwen3 30B-A3B,
  DeepSeek-R1-Distill 32B for Skeptic).
- Backend: `ollama` for ergonomics, `openai_compat` + vLLM for
  throughput across multiple agents.
- 32K context fits comfortably; 10-K RAG benefits.

### NVIDIA GPU 8–16 GB (RTX 3070 / 4060 Ti / A2000)

- 14B-class with 4-bit quantization (Qwen 2.5 14B Q4_K_M, Llama 3.1
  Nemotron 8B for Skeptic).
- Backend: `ollama` or `llamacpp` with `LLAMACPP_N_GPU_LAYERS=-1`.

### NVIDIA GPU 6 GB (RTX 2060, 3060)

- 7–8B-class with 4-bit (Qwen 2.5 7B, Llama 3.1 8B). One model in
  VRAM at a time — the runner handles model swaps with `keep_alive=0`.
- Backend: `ollama` (this is what MAFIS' default `.env` targets).

### Apple Silicon 32 GB+ (M2 Pro, M3 Max)

- 14B–32B class via MLX 4-bit weights, or 32B Q4 via llama.cpp Metal.
- Backend: `mlx` for in-process speed; `openai_compat` pointing at
  `mlx_lm.server` if you want CrewAI tool-calling for the Analyst.

### Apple Silicon 16 GB

- 7–8B class via MLX 4-bit. Save VRAM headroom for the model swap
  during the Skeptic phase.
- Backend: `mlx` or `ollama` (Ollama uses Metal under the hood).

### CPU-only with 32 GB+ RAM

- MoE models with active-parameter counts under your CPU's reach
  (Qwen 2.5 7B Q4_K_M is the floor; Phi-3.5 14B Q4 if you have time).
- Backend: `llamacpp` with default `LLAMACPP_N_GPU_LAYERS=0`.
- A crew run will take 60–90 minutes. Plan accordingly.

### CI / 8 GB dev box

- 3–4B class (Qwen 2.5 3B, Phi-3 mini). Quality drops below the
  audit-pass threshold often — fine for CI smoke, not for actual
  research output.

---

## Backend setup recipes

### Ollama (default)

```bash
# Install Ollama from https://ollama.com
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
# .env stays as the bundled .env.example — no LLM_BACKEND line
# needed (ollama is the default).
```

### OpenAI-compatible (vLLM example)

```bash
# Start the server (single GPU example)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# .env
LLM_BACKEND=openai_compat
OPENAI_COMPAT_BASE_URL=http://localhost:8000/v1
# OPENAI_COMPAT_API_KEY left at the default 'local'
```

LM Studio: point `OPENAI_COMPAT_BASE_URL` at `http://localhost:1234/v1`.
mlx_lm.server: `mlx_lm.server --port 8080` then
`OPENAI_COMPAT_BASE_URL=http://localhost:8080/v1`.

### MLX (Apple Silicon)

```bash
pip install -e ".[mlx]"
# .env
LLM_BACKEND=mlx
# MLX_MODEL_CACHE_DIR optional (defaults to ~/.cache/huggingface/hub)
```

In `config/agent_models.yaml` use HuggingFace repo IDs:

```yaml
defaults:
  backends:
    mlx: "mlx-community/Qwen2.5-7B-Instruct-4bit"
```

### llama.cpp (any OS)

```bash
# CPU-only:
pip install -e ".[llamacpp]"

# NVIDIA GPU:
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install -e ".[llamacpp]"

# Apple Metal:
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -e ".[llamacpp]"
```

Then download a GGUF (e.g. from `bartowski` or `TheBloke` on HF):

```bash
# .env
LLM_BACKEND=llamacpp
LLAMACPP_MODEL_PATH=/path/to/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLAMACPP_N_GPU_LAYERS=-1   # -1 = all layers on GPU; 0 = CPU only
LLAMACPP_N_CTX=8192
```

llama.cpp pins one GGUF per backend instance — to switch models per
agent you'd run multiple instances and point `openai_compat` at a
local llama.cpp server (`python -m llama_cpp.server`).

---

## Per-agent customization

`config/agent_models.yaml` resolves model + sampling per agent.

```yaml
defaults:
  # Leave model unset to defer to .env (ANALYST_MODEL etc.).
  # Set it to centralize the default in this file:
  # model: "qwen2.5:7b"

  # Backend-specific overrides — handy when one backend uses HF repo
  # IDs and another uses Ollama tags for the same weights.
  # backends:
  #   openai_compat: "Qwen/Qwen2.5-7B-Instruct"
  #   mlx: "mlx-community/Qwen2.5-7B-Instruct-4bit"

agents:
  economist:
  analyst:
  valuer:
  skeptic:
  defender:
  steward:
```

Each agent block can override:

```yaml
  skeptic:
    model: "qwen3:14b"
    sampling:
      enable_thinking: true
    backends:
      openai_compat: "Qwen/Qwen3-14B"
```

**Resolution precedence** (most specific wins):

1. `agents.<name>.backends.<backend>`
2. `agents.<name>.model`
3. `defaults.backends.<backend>`
4. `defaults.model`
5. Legacy `.env` (`ANALYST_MODEL`, `SKEPTIC_MODEL`, etc.)

Sampling resolution starts from
`get_recommended_sampling(model, enable_thinking=…)` (see
`src/wise_investor/llm/utils/sampling.py`) and overlays the
`defaults.sampling` block, then the agent's `sampling:` block.

---

## Thinking-mode models

Qwen3 (when prompted to think) and DeepSeek-R1 emit a
`<think>...</think>` block before the final answer. MAFIS strips
that block automatically (`wise_investor.llm.utils.thinking`) so
the report sections never contain reasoning leakage.

To turn thinking on for a single agent:

```yaml
agents:
  skeptic:
    model: "qwen3:14b"   # whatever your Ollama tag is
    sampling:
      enable_thinking: true
```

The recommended sampling for thinking-mode Qwen3 (temp 0.6, top_p 0.95,
min_p 0) is auto-applied. Don't override these unless you have a
specific reason — Qwen's authors are explicit that low-temperature
thinking-mode generation degrades sharply.

DeepSeek-R1 distill series doesn't take an `enable_thinking` flag —
it always emits the thinking block. The same `<think>` stripping
applies. Set the model name and let the sampling defaults handle
the rest.

---

## Re-enabling deterministic output

The pre-Phase-5 contract was `temperature=0, seed=42`. To restore it
for a specific agent (or all of them):

```yaml
defaults:
  sampling:
    temperature: 0.0
    seed: 42

agents:
  steward:
    sampling:
      temperature: 0.0
      seed: 42
```

When you should consider this:

- **Steward only**: keeps the verdict strictly reproducible while the
  upstream agents still benefit from their model's recommended
  sampling for prose quality.
- **All agents**: regulator-style backtests, paper-revision diffs.

When you should NOT do this:

- General research use. Modern instruction-tuned models genuinely
  produce worse output at temperature 0.
- Paper trading evaluation — the sample diversity is informative.

Note: backend support for `seed` is uneven. Ollama and OpenAI-compat
servers usually honor it. MLX and llama.cpp support it too but the
exact reproducibility guarantee depends on the build (BLAS thread
count, etc.). Verify with two runs before relying on it.

---

## `regression_compare` and paper trading

Both tools predate Phase 5 and assumed bit-exact re-runs. They still
work, but the interpretation changes:

### `scripts/regression_compare.py`

- **Old**: "diff is a real change in behavior."
- **New**: "diff might be sampling noise; only large structural
  changes (verdict flip, citation drop, section restructure) are
  signal."

If you're tuning a prompt and want to detect "did this change make
runs better/worse on average", run each side N times (5–10) and
compare distributions, not single outputs. A future helper might
formalize this; for now `paper_trading/ledger.py` aggregates BUY/HOLD/
PASS counts, which is the right shape for that question.

### `paper_trading/ledger.py`

- A single BUY → PASS audit downgrade is no longer a deterministic
  consequence of the inputs; it's one sample of the model's behavior.
- `audit_effect` (`scripts/paper_ledger.py summary`) is still valid
  *as an aggregate*: across many runs, audit-downgraded BUYs vs
  clean BUYs reveals whether the discipline matrix actually catches
  losers.
- For a single ticker decision: run two or three times, take the
  modal verdict, note the spread. Don't treat one BUY/PASS as a
  reproducible classification.

---

## Troubleshooting

**"OpenAICompatBackend requires a base URL"**: set
`OPENAI_COMPAT_BASE_URL` in `.env`. The factory deliberately fails
loudly rather than silently falling back to Ollama — a typo there
should surface immediately.

**"MLX backend requires mlx-lm, which only runs on Apple Silicon"**:
expected on Windows / Linux / Intel Mac. Use `LLM_BACKEND=ollama` or
run `mlx_lm.server` on a Mac and use `LLM_BACKEND=openai_compat`.

**"GGUF file not found"**: `LLAMACPP_MODEL_PATH` points at a non-
existent file. Download a GGUF first (e.g. `huggingface-cli download
bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf`)
and set the absolute path.

**Out-of-memory on a 6 GB VRAM card**: the crew runner swaps models
between Skeptic (`keep_alive="0"`) and the others to keep one model
in memory at a time. If it still OOMs, switch to a smaller
quantization (Q4_K_S, Q4_0) or drop to a 3B-class model for the
Skeptic seat only:

```yaml
agents:
  skeptic:
    model: "qwen2.5:3b"
```

**Audit failure rate jumps after switching models**: the discipline
matrix is sensitive to a model's calibration around `BUY/HOLD/PASS`
language. Some models lean dovish (more BUYs that audit downgrades)
or hawkish (more PASSes). This is observed behavior, not a regression
— check `paper_ledger.py summary --days 30` to see whether audit
downgrades are catching real losers or rejecting good calls.

**`regression_compare.py` shows differences after a no-op refactor**:
expected — same prompt twice can produce different outputs under the
new sampling policy. If you need a binary "did the refactor change
behavior" signal, temporarily pin both runs with
`temperature: 0, seed: 42` in `defaults.sampling`, run both, then
revert the override.
