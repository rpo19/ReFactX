# TODO: stop current RL run, merge efficiency branch, switch to GSPO loss, relaunch

Context for the agent picking this up. Repo: `ReFactX` (cluster login node).
Current branch: `feat/sentinel-count-branches` at `2bd5aa4`. Working tree should
be clean except untracked scratch files.

Active job: `4605057` (`hpc/trl_qwen35_4b_sft_grpo_mix.sh`,
config `configs/trl_qwen35_4b_sft_grpo_mix_cluster.json`). It is very slow
(`step_time` 90-310 s) and uses a **hybrid loss**: `importance_sampling_level="sequence"`
with TRL's default `loss_type="dapo"` (length-weighted, not GSPO).

Goal (the "1 and 2" from the review):
1. Wire `loss_type` through the config and set `"loss_type": "grpo"` (pure GSPO)
   in the fast config.
2. Stop `4605057` and relaunch with the fast config.
Plus: decide whether `origin/feat/constrained-decoding-efficiency` is useful and,
if so, merge it **before** the relaunch.

## 0. Safety checks

```bash
git --no-pager status --short           # only untracked files expected
git --no-pager log --oneline -1         # expect 2bd5aa4 (or newer)
sacct -j 4605057 --format=JobID,State,Elapsed
```

Do not overwrite production outputs: production uses
`${WS_PATH}/grpo_output_supported_mix_all4`; the fast config writes
`.../grpo_output_supported_mix_all4_fast`.

## 1. Stop the current run

```bash
scancel 4605057
sacct -j 4605057 --format=JobID,State,Elapsed
```

Expected: the new `EXIT` trap sends `teleclinotify "rl error ..."` on cancel —
that is expected, not a bug. Confirm no `trl_*` job from this user is still
RUNNING before relaunching.

## 2. Evaluate and merge the efficiency branch

Branch: `origin/feat/constrained-decoding-efficiency`, one commit `9cbc2d0`
"Speed up constrained decoding and add batched vLLM path", forked from the same
base `0094ffb` as this branch. `git merge-tree --write-tree HEAD origin/...`
reports **no conflicts** (checked).

Why it is useful (see `SPEED_IMPROVEMENTS_REPORT.md` on the branch):

- **Re-enables the KV cache during rollouts/eval.** Current `utils/run_trl.py`
  sets `model.config.use_cache = False`; generation without cache is
  dramatically slower. This is likely the dominant slowdown, more than the
  constrained processor itself (report measures ~8-11% processor overhead at
  steady state).
- Cheaper cache/index: tuple keys, canonical per-prefix subtrees, no full
  `scores`-shaped mask when nothing is constrained, O(1) per-row state updates.
- New knobs: `attn_implementation`, `dataset_num_proc`, `use_generation_cache`,
  `parallel_workers` (`parallel_workers` is disabled automatically for
  PostgreSQL/HTTP indexes, so it will not help the Postgres-backed run).
- Optional vLLM adapter (`refactx/vllm_proc.py`, `examples/vllm_offline.py`,
  `[project.optional-dependencies].vllm`). **Do not install/use vLLM for this
  relaunch**; keep the default HuggingFace path.
- Tests: `tests/test_efficiency.py`, `tests/test_vllm_proc.py` (skipped without
  vllm), benchmarks: `utils/benchmark_efficiency.py`,
  `utils/constrained_throughput.py`.

Merge:

```bash
git fetch origin
GIT_EDITOR=true git merge --no-ff --no-edit origin/feat/constrained-decoding-efficiency
```

If the merge unexpectedly conflicts (it should not), abort and fall back to
Step 3 + fast config only; note the conflict in the final report.

## 3. Wire `loss_type` and switch the fast config to GSPO

`utils/run_trl.py` currently never sets `loss_type`. Add it in the `defaults`
dict in `parse_args()`:

```python
"loss_type": config.get("loss_type", "dapo"),
```

and pass it to `GRPOConfig(...)`:

```python
loss_type=args.loss_type,
```

Then edit `configs/trl_qwen35_4b_sft_grpo_mix_fast.json` (already contains the
efficiency knobs: `max_completion_length=1024`, `num_generations=4`,
`custom_eval_steps=100`, `max_eval_samples=16`, `save_steps=100`) and add:

```json
"loss_type": "grpo",
"use_generation_cache": true,
"attn_implementation": "sdpa",
"dataset_num_proc": 8,
```

Keep the production config (`..._mix_cluster.json`) unchanged.

Optional: after the smoke run, try `"attn_implementation": "flash_attention_2"`.
Check `flash_attn_2_cuda` is importable first; revert to `sdpa` on any issue.

After the merge + edit, confirm the TRL warning
"When using `importance_sampling_level='sequence'`, the `'dapo'` loss ..."
no longer appears in the new job's `.err`.

## 4. Tests before submitting

Use the workspace venv (has the deps; falls back handled by `hpc/env.sh`):

```bash
PY=/data/horse/ws/ripo631h-quokka/venvs/refactx/bin/python
"$PY" tests/test_efficiency.py
"$PY" tests/test_routed_index.py
"$PY" tests/test_run_trl_reward.py
"$PY" tests/test_run_trl_tool_metrics.py
"$PY" tests/test_tool_callbacks.py
"$PY" tests/test_features.py
"$PY" -m py_compile utils/run_trl.py
```

`tests/test_vllm_proc.py` may skip (no vllm) — that is fine.

Optional benchmark on the target server:

```bash
"$PY" utils/benchmark_efficiency.py processor
"$PY" utils/constrained_throughput.py --model Qwen/Qwen3.5-0.8B --table qwen36 \
  --max-tokens 128 --output logs/throughput_constrained.json
```

## 5. Smoke run, then full relaunch

Submit the fast config:

```bash
RL_CONFIG=configs/trl_qwen35_4b_sft_grpo_mix_fast.json \
  sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh
JOB=$(squeue -u "$USER" -h -o '%i %j' | awk '/trl_qwen35_4b_sft_mix/ {print $1}' | tail -1)
echo "$JOB"
```

Watch `logs/trl_qwen35_4b_sft_mix_<JOB>.{out,err}` and W&B. Acceptance for the
smoke phase (first ~5 optimizer steps):

- no `loss_type`/`sequence` warning; W&B run name
  `trl_qwen35_4b_sft_supported_mix_grpo_all4_fast` in project `refactx_grpo`;
- `index routing ... misses=0` and both tables loaded
  (`qwen36`, `qwen36fbwikiprops`);
- KV cache enabled for rollouts (no O(T^2) regeneration; cross-check
  `step_time`);
- `step_time` clearly below the 90-310 s baseline (expect a large drop from the
  KV cache + shorter completions; record the measured number);
- `completions/clipped_ratio` lower than 0.53-0.56.

If the smoke is good, let it continue as the full run. If not, cancel and report
the failing metric/log line; do not silently revert the loss change.

## 6. Commit

Commit the merge and the code/config changes (keep the merge commit):

```bash
git add -A
GIT_EDITOR=true git commit -m "Switch fast RL config to GSPO loss and merge efficiency branch"
```

(If the merge was already committed, just commit the `loss_type` wiring and the
fast-config edits.)

## Guardrails

- Do not modify or restart the production config/artifacts.
- Do not install vLLM or change `pyproject` extras for this run.
- Keep `teleclinotify` enabled; an `rl error` on `scancel` is expected.
- If the merged branch's vLLM path is imported accidentally, ensure the default
  HF rollout is used (no `use_vllm` in the config).
- Note the exact measured `step_time` before (90-310 s on `4605057`) and after,
  in the final report.
