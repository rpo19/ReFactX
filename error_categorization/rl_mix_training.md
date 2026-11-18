# Mixed RL training across all four datasets

This prepares a single GRPO/GSPO run over Mintaka, 2WikiMultihopQA, WebQSP and
CWQ, with the KB table carried per example.

## KB tables

| Datasets | Table | Notes |
|---|---|---|
| Mintaka, 2Wiki | `qwen36` | Wikidata |
| WebQSP, CWQ | `qwen36fbwikipops` | Freebase, as present in the local DB |

The Freebase table name is inconsistent in the repository: `hpc/populate_fb.sh`
and the eval configs use **`qwen36fbwikiprops`** (with `r`), while the local
database has **`qwen36fbwikipops`** (without `r`). The live local table
`qwen36fbwikiprops` does not exist. The builder takes `--freebase-table`, so use
the name that actually exists on the target cluster:

```bash
# local / this DB
python -m utils.build_rl_dataset --freebase-table qwen36fbwikipops ...
# cluster, if hpc/populate_fb.sh was actually used
python -m utils.build_rl_dataset --freebase-table qwen36fbwikiprops ...
```

The RL config does not hardcode table names: `utils/run_trl.py` infers the set
from the `index_table` column and preloads exactly those tables.

## Build the dataset

```bash
python -m utils.build_rl_dataset \
  --output logs/rl_train_mix.jsonl \
  --max-per-dataset 2000 \
  --val-per-dataset 50 \
  --count-target 1400 \
  --allow-overlap-types count \
  --exclude-from sft_training_mix_supported.jsonl.gz \
  --freebase-table qwen36fbwikipops
```

- Uses train/validation splits only.
- Answers are normalized to lists (RoG string-lists are parsed).
- `question_type`, `requires_full_enumeration`, `dataset` and `index_table` are
  stored per record.
- `--exclude-from` subtracts questions already used by SFT. If the file is
  absent, the builder warns and RL may reuse SFT questions. The default files
  `sft_training_mix_supported.jsonl.gz`,
  `logs/sft_training_mix_supported.jsonl[.gz]` and `logs/sft_training_mix.jsonl`
  are auto-used when present; pass `--no-default-exclude` to ignore them.
- `--count-target 1400 --allow-overlap-types count` keeps the Mintaka count
  pool (the only dataset with `count` questions). Count questions are allowed to
  overlap SFT; every other type remains strictly disjoint.
- Validation questions are selected first so they cannot also appear in train.
- A manifest is written next to the output (`*.manifest.json`).

The final build with `sft_training_mix_supported.jsonl.gz` (6,966 unique SFT
questions excluded) produced **8,200 records**:

| Split | Mintaka | 2Wiki | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| train | 2,000 | 2,000 | 2,000 | 2,000 |
| validation | 50 | 50 | 50 | 50 |

Index split: 4,100 `qwen36` + 4,100 `qwen36fbwikipops`. All 8,200 questions are
unique, train/validation are disjoint, and there are no empty questions or
answers.

**Count policy:** `--count-target 1400 --allow-overlap-types count` deliberately
keeps the count questions. The output has **1,399 count-type train questions**
(5 validation), of which **1,140 overlap the SFT mixture** (all `count`). Every
non-count question is strictly disjoint from SFT (0 overlap). This raises count
coverage from 61 to ~1,400 while preserving a clean non-count split.

If strict SFT disjointness matters more than count coverage, build with
`--count-target 0` instead; in that case only 61 count questions remain.

## Run RL

Local smoke (0.8B, both tables, one step):

```bash
python -m utils.run_trl --config configs/trl_mix_smoke.json
```

Production config: `configs/trl_qwen35_4b_sft_grpo_mix.json` (absolute alpha
paths).

```bash
python utils/run_trl.py --config configs/trl_qwen35_4b_sft_grpo_mix.json
```

Cluster config: `configs/trl_qwen35_4b_sft_grpo_mix_cluster.json`. It is the same
run with every location rooted at `${WS_PATH}`, which `utils/run_trl.py` expands
for `adapter`, `dataset`, `prompt`, `output_dir`, `generation_output` and
`resume_from_checkpoint`. `hpc/env.sh` already exports `WS_PATH` for the current
cluster, so the same file works on capella (`/data/cat/ws/...`) and alpha
(`/data/horse/ws/...`). If a variable is missing, `run_trl.py` warns and leaves
the literal `${...}` in the path.

Locally (the extracted adapter is under `logs/`):

```bash
python utils/run_trl.py \
  --config configs/trl_qwen35_4b_sft_grpo_mix.json \
  --adapter logs/sft_output_supported_mix/sft_output_supported_mix \
  --report-to none
```

The validation rows are shuffled across datasets, so `max_eval_samples: 32`
still covers all four datasets rather than only the first one in file order.

Cluster:

The job reads the mixture from `${WS_PATH}/datasets/rl_train_mix.jsonl`. Upload
the locally built file once instead of rebuilding on the cluster, so the run uses
exactly the validated 8,200 records (the script rebuilds only when the file is
absent or `REBUILD=1`).

```bash
# from the local repo root (pick the login node and the workspace that the
# cluster actually mounts; $WS_PATH is printed by hpc/env.sh)
scp logs/rl_train_mix.jsonl \
  ripo631h@<LOGIN_NODE>:/data/horse/ws/ripo631h-quokka/datasets/rl_train_mix.jsonl
# capella workspace instead of alpha:
# /data/cat/ws/ripo631h-horse/datasets/rl_train_mix.jsonl

# optional, only for provenance
scp logs/rl_train_mix.jsonl.manifest.json \
  ripo631h@<LOGIN_NODE>:/data/horse/ws/ripo631h-quokka/datasets/rl_train_mix.jsonl.manifest.json
```

Create the directory first if it does not exist:

```bash
ssh ripo631h@<LOGIN_NODE> 'mkdir -p /data/horse/ws/ripo631h-quokka/datasets'
```

Then submit:

```bash
cd /home/ripo631h/ReFactX

# uses configs/trl_qwen35_4b_sft_grpo_mix_cluster.json and the uploaded dataset
sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# rebuild the mixture on the cluster instead of reusing the uploaded one
REBUILD=1 sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# the SFT mixture is still needed when rebuilding (--exclude-from)
scp sft_training_mix_supported.jsonl.gz ripo631h@<LOGIN_NODE>:/home/ripo631h/ReFactX/

# If the cluster Freebase table has the "r":
FREEBASE_TABLE=qwen36fbwikiprops sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# Strict SFT-disjoint mixture (drops count coverage to ~61):
COUNT_TARGET=0 REBUILD=1 sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# different config for a one-off run:
RL_CONFIG=configs/trl_qwen35_4b_sft_grpo_mix.json sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh
```

The job reuses `${WS_PATH}/datasets/rl_train_mix.jsonl` when present (or builds
it), checks the SFT adapter, then runs GRPO/GSPO. It uses the same per-job
PostgreSQL overlay as the other HPC scripts. SLURM logs are
`logs/trl_qwen35_4b_sft_mix_<jobid>.{out,err}`; checkpoints and generations go to
`${WS_PATH}/grpo_output_supported_mix_all4`.

## GSPO loss type

`"gspo": true` sets `importance_sampling_level="sequence"` in `GRPOConfig` but
leaves `loss_type` at TRL's default. TRL warns that with sequence-level
importance sampling the default `dapo` loss sums per-token contributions, which
weights each sequence by its completion length instead of optimizing the
per-sequence objective. To reproduce the GSPO paper's objective, pass
`loss_type="grpo"` to `GRPOConfig` (the config does not expose it yet). This is
intentionally left unchanged for now; decide before comparing runs against the
paper.

## Throughput notes

The constrained decoder is the bottleneck: every generated token goes through a
Python trie/logits processor, and the SFT policy emits long tool chains. On the
first mixed 4B run (`4605057`) completions averaged ~1,300 tokens with 53-56%
clipped at `max_completion_length=2048` and ~35 fact calls per sample; logged
`step_time` was 90-310 s, which projects well beyond the 72 h limit. Levers,
cheapest first:

- lower `max_completion_length` (successfully terminated completions averaged
  ~390 tokens, so most of the clipped ones waste generation);
- lower `num_generations` (8 -> 4) and cut validation cost
  (`custom_eval_steps`, `max_eval_samples`);
- use both GPUs: the script starts a single `python utils/run_trl.py` process,
  so one H100 is idle; `accelerate launch --num_processes 2` would double
  throughput if the per-rank index/DB setup is acceptable;
- a vLLM rollout is much faster but bypasses the patched `model.generate`, so
  the per-row index routing would need a vLLM-side hook.

## HPC submission checklist

1. `hpc/env.sh` provides `$WS_PATH`, `$POSTGRES_CONNECTION`, and the overlay.
2. `$WS_PATH/sft_output_supported_mix/adapter_config.json` exists and matches
   the adapter path in `configs/trl_qwen35_4b_sft_grpo_mix_cluster.json`.
3. `$WS_PATH/datasets/rl_train_mix.jsonl` is uploaded (or `REBUILD=1` is used so
   the builder creates it).
4. The Freebase table exists; use `FREEBASE_TABLE` to override the default
   (`qwen36fbwikiprops`).
5. The SFT mixture is visible as
   `logs/sft_training_mix_supported.jsonl[.gz]` or
   `sft_training_mix_supported.jsonl.gz` in the repo root, so the builder can
   subtract it (counts are allowed to overlap).
6. Submit with `sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh`.

## What changed

- `refactx/index.py`: new `RoutedIndex` selects a table from
  `state.index_table` on every `next_tokens` call.
- `utils/build_rl_dataset.py`: mixed dataset builder with per-sample
  `index_table`, SFT-overlap exclusion, and a count quota
  (`--count-target`/`--allow-overlap-types`).
- `utils/run_trl.py`:
  - local JSONL loading with a `split` column;
  - preloads all tables referenced by `index_table`;
  - generates with one logits processor per actual generation batch and assigns
    one table per row (including `num_return_sequences` expansion);
  - per-example `dataset` tag for normalized reward and enumeration gating;
  - `eval/dataset/<name>/accuracy` metrics;
  - T4-safe fp16 fallback and `remove_unused_columns=False`;
  - fixed a `parse_args` crash for config-only keys;
  - expands `${VAR}`/`~` in path-valued config keys, so the cluster config only
    needs `WS_PATH`.
- `configs/trl_qwen35_4b_sft_grpo_mix_cluster.json`: the production run with
  `${WS_PATH}` paths for capella/alpha and the dataset under
  `$WS_PATH/datasets`.
- `hpc/trl_qwen35_4b_sft_grpo_mix.sh`: reuses an uploaded dataset instead of
  always rebuilding (`REBUILD=1` forces a rebuild) and defaults to the cluster
  config (`RL_CONFIG` overrides).

## Tests run

```bash
python tests/test_routed_index.py      # 6 passed
python tests/test_build_rl_dataset.py  # 9 passed
python tests/test_run_trl_reward.py  # 33 passed
python tests/test_run_trl_tool_metrics.py  # 7 passed
python tests/test_features.py          # 45 passed
python tests/test_tool_callbacks.py    # 4 passed
```

Smoke tests:

- 0.8B, both indexes: routed `qwen36` + `qwen36fbwikipops`, 0 misses.
- 4B + SFT adapter, one step: reward received per-example datasets
  (`['cwq','cwq','2wiki','2wiki']`) and the mixed batch loaded both indexes.

## Caveats

- Routing recovers the table from the prompt text. The builder deduplicates
  normalized questions across datasets; if the same prompt still maps to two
  tables, `run_trl.py` warns and uses `default_index_table`.
- The multi-index wrapper patches `model.generate`, so it applies to the
  regular Transformers generation path. A vLLM rollout path would need its own
  routing hook (the configs here use the default path).
- `qwen36` and the Freebase table must share the tokenizer and trie settings;
  `RoutedIndex` rejects mismatched `switch_parameter`/`rootkey`.
