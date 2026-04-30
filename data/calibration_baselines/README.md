# Calibration baselines

Curated snapshots of calibration ledger entries that subsequent
calibration runs measure themselves against. Each file in this
directory captures one run that's structurally meaningful — a
post-fix baseline, a model-size sweep, a rubric-revision evidence
point — and is preserved across `git pull` so cross-revision
comparison works on a clean clone.

This is distinct from `/data/calibration_ledger/` (untracked,
per-run output): a baseline is *the* ledger you want to keep, a
ledger entry is just the latest run's bytes.

## File naming convention

```
v{N}_post-{phase}_{model}_{date}.json
```

- `v{N}` — baseline number (incremented per curated snapshot)
- `post-{phase}` — what the calibration was measuring after
  (e.g. `post-P3-5` = the first calibration after Stage 3 LLM
  wiring landed)
- `{model}` — the LLM backend / model used for Stage 3, with
  `none` if Stage 2 only
- `{date}` — ISO date the run was captured

## Current baselines

### `v5_post-P3-5_qwen2.5-7b-16k_2026-04-30.json`

The first LLM-enabled calibration run. Measures the joint behavior
of:

- Constitution v2.0
- Option A IC formula (commit b28b3c5)
- All P0/P1/P3-1..3 infrastructure fixes
- RAG signals enabled (lookahead bias caveat)
- Stage 3 LLM = qwen2.5:7b-16k (Ollama, TEMP=0.0, SEED=42 deterministic)

Topline metrics (Stage 3 final decision):

| Metric            | Value           |
|-------------------|-----------------|
| n evaluated       | 30              |
| Final advance     | 0               |
| Final reject      | 30 (29 + SPY)   |
| TP / FP / TN / FN | 0 / 0 / 9 / 21  |
| Precision         | undefined (TP=0)|
| Recall            | 0.0%            |
| Accuracy          | 30.0%           |

The 21 false negatives include NVDA (+541.7% over benchmark), ASML
(+226.1%), TSM (+147.9%), LLY (+432.7%) — high-conviction outperformers
that the 7B model rejected. Stage 3 produced 5 INVALID-axis records
(T, INTC, GE, COST, 005380.KS) where the LLM couldn't render its
verdict in the required JSON shape; per Commitment 3 these auto-route
to REJECT.

This baseline is **not a constitution-revision trigger**. The honest
diagnosis is that qwen2.5:7b-16k can't carry the §17/§18 qualitative
load consistently. Larger model variants (14B, thinking-class) are
the natural next sweep before adjusting any threshold.

### Future baselines (placeholder)

- `v6_post-model-sweep_qwen2.5-14b_*.json` — 14B comparison
- `v7_post-prompt-revision_*.json` — if Stage 3 prompts get
  improved
