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

Production config: `configs/trl_qwen35_4b_sft_grpo_mix.json`.

```bash
python utils/run_trl.py --config configs/trl_qwen35_4b_sft_grpo_mix.json
```

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

```bash
cd /home/ripo631h/ReFactX
sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# If the cluster Freebase table has the "r":
FREEBASE_TABLE=qwen36fbwikiprops sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh

# Strict SFT-disjoint mixture (drops count coverage to ~61):
COUNT_TARGET=0 sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh
```

The job builds `logs/rl_train_mix.jsonl` first, checks the SFT adapter, then
runs GRPO/GSPO. It uses the same per-job PostgreSQL overlay as the other HPC
scripts. SLURM logs are `logs/trl_qwen35_4b_sft_mix_<jobid>.{out,err}`.

## HPC submission checklist

1. `hpc/env.sh` provides `$WS_PATH`, `$POSTGRES_CONNECTION`, and the overlay.
2. `$WS_PATH/sft_output_supported_mix/adapter_config.json` exists and matches
   the adapter path in `configs/trl_qwen35_4b_sft_grpo_mix.json`.
3. The Freebase table exists; use `FREEBASE_TABLE` to override the default
   (`qwen36fbwikipops`).
4. The SFT mixture is visible as
   `logs/sft_training_mix_supported.jsonl[.gz]` or
   `sft_training_mix_supported.jsonl.gz` in the repo root, so the builder can
   subtract it (counts are allowed to overlap).
5. Submit with `sbatch hpc/trl_qwen35_4b_sft_grpo_mix.sh`.

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
  - fixed a `parse_args` crash for config-only keys.

## Tests run

```bash
python tests/test_routed_index.py      # 6 passed
python tests/test_build_rl_dataset.py  # 9 passed
python tests/test_run_trl_reward.py    # 30 passed
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
