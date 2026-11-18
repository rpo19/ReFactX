# RL efficiency plan

Status: proposed, not applied to the production run `4605057`.
Scope: make the mixed GRPO/GSPO run finish within the 72 h SLURM limit without
changing the algorithm or the reward.

## Baseline (job `4605057`, 4B SFT adapter, single process, 2 GPUs requested)

Measured from `logs/trl_qwen35_4b_sft_mix_4605057.err`,
`wandb/latest-run/files/output.log`, and the training progress bar:

| Metric | Value |
|---|---|
| `step_time` | 90–310 s per optimizer step |
| `completions/mean_length` | ~1272–1328 tokens |
| `completions/clipped_ratio` | 53–56% at `max_completion_length=2048` |
| `completions/mean_terminated_length` | ~390 tokens |
| `eval/fact_calls_mean` (p90) | 35 (80) |
| progress bar | `30/8000 [1:29:21<275:15:41]` |
| GPUs actually used | 1 (script starts one `python utils/run_trl.py`) |

Root causes, in order of impact:

1. Constrained generation is CPU-bound: every token goes through the Python
   trie/logits processor, so the GPU idles between tokens.
2. `max_completion_length=2048` while successfully terminated completions
   average ~390 tokens: over half the generated tokens run to the cap.
3. Only one of the two H100s is used.
4. `num_generations=8` and a synchronous custom eval every 30 steps (32
   constrained generations) create the multi-minute stalls.

## Goals / non-goals

- Goal: cut wall-clock enough to finish one epoch inside 72 h.
- Goal: keep the reward, dataset, adapter, and constrained decoding semantics.
- Non-goal: changing the objective (`loss_type`, KL, normalization) — see the
  orthogonal note at the end.
- Non-goal: rewriting the constrained decoder or migrating to vLLM (Phase 3,
  optional).

## Metrics to track

Primary (speed):

- `step_time` and steps/hour from the TRL log line;
- `completions/mean_length`, `completions/clipped_ratio`;
- tokens generated per second (`num_tokens` / `step_time`);
- projected finish time (`remaining steps * mean step_time`).

Guardrails (quality; must not regress materially):

- `eval/answer_accuracy`, `eval/normalized_accuracy`;
- `eval/truncation_rate`, `eval/format_accuracy`;
- sentinel/count metrics: `eval/sentinel_rate`,
  `eval/tool/count_usage_on_count_type`;
- `frac_reward_zero_std` (lower is better signal).

## Phase 0 — baseline and smoke harness

On the test server:

```bash
# 1. same dataset/adapter/tables as production
ls "${WS_PATH}/datasets/rl_train_mix.jsonl"
ls "${WS_PATH}/sft_output_supported_mix/adapter_config.json"

# 2. short smoke run (few steps) before any long job
RL_CONFIG=configs/trl_qwen35_4b_sft_grpo_mix_fast.json \
  sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh
# then, once it starts, cancel after ~5 optimizer steps and inspect step_time
```

Record the first 5 `step_time` values as the baseline for that server.

## Phase 1 — config-only changes (no code)

Create `configs/trl_qwen35_4b_sft_grpo_mix_fast.json` (added in this change) from
`trl_qwen35_4b_sft_grpo_mix_cluster.json` with:

| Key | Production | Fast | Rationale / risk |
|---|---|---|---|
| `max_completion_length` | 2048 | **1024** | Biggest cheap win: >50% of samples clip at 2048. Risk: longer enumeration chains truncate; watch `eval/truncation_rate`. A 768 setting is an even faster fallback. |
| `num_generations` | 8 | **4** | Halves sequences generated per prompt. Risk: higher-variance advantages; watch `frac_reward_zero_std`. Keep 8 if quality drops. |
| `custom_eval_steps` | 30 | **100** | Each eval runs 32 constrained generations synchronously. |
| `max_eval_samples` | 32 | **16** | Halves eval cost; still covers all four datasets. |
| `save_steps` | 60 | **100** | Fewer Lustre checkpoint stalls; combined with `save_total_limit=2`. |
| `experiment_name` | `..._all4_cluster` | `..._all4_fast` | Separate W&B runs. |
| `output_dir` / `generation_output` | `.../all4` | `.../all4_fast` | Do not overwrite production artifacts. |

`steps_per_generation` is derived from `num_generations` in `utils/run_trl.py`,
so it follows automatically (4).

Run:

```bash
RL_CONFIG=configs/trl_qwen35_4b_sft_grpo_mix_fast.json \
  sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh
```

Expected: roughly 2.5–4x fewer generated tokens per step, plus fewer eval stalls.
Validate on the first ~20 steps before committing to a full run.

Acceptance for Phase 1:

- `step_time` down by >= 2x vs Phase 0;
- `completions/clipped_ratio` materially lower;
- `eval/answer_accuracy` and `eval/format_accuracy` within a few points of the
  production baseline at the same optimizer step.

## Phase 2 — use both GPUs (needs small code changes)

Current launch is a single process, so one H100 is idle. TRL's `GRPOTrainer`
supports DDP via `accelerate`, but the custom bits are not rank-aware yet:

Required code changes in `utils/run_trl.py`:

1. Guard evaluation and generation-file writes to the main process:
   - `CustomMetricsCallback.on_step_end` / `on_log` should run `evaluate_policy`
     and append to `args.generation_output` only on rank 0
     (`trainer.is_world_process_zero()` / `state.is_world_process_zero`).
   - Otherwise every rank appends to the same JSONL and races.
2. Reduce/aggregate tool counters across ranks (decoder counters and
   `triple_counter` are per-process); either log per-rank or `all_reduce` them.
3. Confirm index/DB access per rank:
   - each process builds its own `RoutedIndex` from `INDEX` (the per-job
     Postgres overlay socket on the same node), so both ranks can connect;
   - memory doubles (RoutedIndex + model shard); check node RAM.
4. Keep `output_dir`/checkpointing to the Trainer (already rank-0 only).

Launch:

```bash
"$PYTHON" -m accelerate.commands.launch --num_processes 2 --mixed_precision bf16 \
  utils/run_trl.py --config configs/trl_qwen35_4b_sft_grpo_mix_fast.json
```

(Add this as an opt-in in `hpc/trl_qwen35_4b_sft_grpo_mix.sh`, e.g. a
`NUM_PROCESSES=2` env var, defaulting to 1.)

Acceptance for Phase 2:

- near-linear throughput improvement (>= 1.6x) with unchanged step semantics;
- no duplicate/missing lines in `generations.jsonl`;
- W&B run still logged once (rank 0).

Risk: DDP interactions with the patched `model.generate` are per-process and
should be safe, but verify routing stats per rank (`index routing ... misses=0`).

## Phase 3 — optional rollout backend (vLLM)

Potential large speedup, but not a drop-in:

- TRL `use_vllm` bypasses the patched `model.generate`, so the per-row index
  routing (`RoutedIndex` + `constrained_generate`) would need a vLLM-side hook;
- constrained/guided decoding inside vLLM must support the same trie
  constraints (`<fact>` triple generation, `<count>`, sentinel);
- evaluate whether correctness (routing, sentinel, count callbacks) is
  preserved before adopting.

Treat as a separate project; do not block Phases 1–2 on it.

## Orthogonal: GSPO loss type (not an efficiency fix)

`gspo: true` sets `importance_sampling_level="sequence"` but leaves TRL's default
`dapo` loss; reproducing the GSPO paper objective needs `loss_type="grpo"`.
`utils/run_trl.py` does not expose `loss_type` yet. Decide separately from the
speed work; document the choice in the run notes.

## Rollback

- Phase 1: revert to `configs/trl_qwen35_4b_sft_grpo_mix_cluster.json`; artifacts
  go to a separate `_fast` output dir, so production output is untouched.
- Phase 2: set `NUM_PROCESSES=1` (or call the script without accelerate).
- Phase 3: keep the Transformers rollout path as the default.

## Test-server checklist

1. Deps import quickly (consider `hpc/setup_venv.sh`; `env.sh` uses
   `$WS_PATH/venvs/refactx` when present).
2. Dataset, SFT adapter, and both KB tables exist.
3. `bash -n hpc/trl_qwen35_4b_sft_grpo_mix.sh`.
4. Smoke run with `max_steps` small, confirm `step_time`, routing `misses=0`,
   and W&B project `refactx_grpo`.
5. Apply Phase 1, re-measure; only then consider Phase 2.
6. Keep `teleclinotify` on so failures are reported.
