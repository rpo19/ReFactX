# Evaluation plan: supported-mix SFT

Evaluation of the adapter produced by `hpc/sft_train_supported_mix.sh`
(`$WS_PATH/sft_output_supported_mix`, 2 epochs, final eval loss 0.3166).

## Preliminary comparison: random 100 (vanilla 4B vs SFT-4B)

A shared random 100-example sample per dataset, generated with
`utils/make_sample_ids.py` (`seed=42`) from the labeled split. Both models load
the same JSON list via `sample_index_file`, so they see exactly the same
questions in the same order.

| Dataset | Split | Sample file | id key | Index table |
|---|---|---|---|---|
| Mintaka | test | `configs/samples/mintaka_test_100_seed42.json` | `id` | `qwen36` |
| 2WikiMultihopQA | validation (dev) | `configs/samples/2wiki_validation_100_seed42.json` | `_id` | `qwen36` |
| WebQSP | test | `configs/samples/webqsp_test_100_seed42.json` | `id` | `qwen36fbwikiprops` |
| CWQ | test | `configs/samples/cwq_test_100_seed42.json` | `id` | `qwen36fbwikiprops` |

Configs:

- `configs/config_prelim100_{mintaka,2wiki,webqsp,cwq}_vanilla4b.json`
- `configs/config_prelim100_{mintaka,2wiki,webqsp,cwq}_sft4b.json`

All use `Qwen/Qwen3.5-4B`, `prompt_qwen36_angular2_nothink.yaml`,
`sentinel: true`; the SFT configs add
`adapter_path: /data/horse/ws/ripo631h-quokka/sft_output_supported_mix`.

```bash
# Two-GPU paired launcher (vanilla + SFT together, one dataset at a time)
sbatch hpc/eval_prelim100.sh

# Or submit 8 independent one-GPU jobs and let Slurm schedule them
bash hpc/submit_eval_prelim100_4b.sh
```

The paired launcher runs vanilla + SFT for each dataset in parallel on two GPUs.
The submit script instead calls `sbatch` eight times (one job per config, each
requesting `--gres=gpu:1`), so Slurm decides how many run concurrently.
Outputs are `logs/prelim100_{dataset}_{vanilla4b,sft4b}.out.0`.

## 200-example checkpoint comparison

The 200-example configs are kept for continuity with the checkpoint-720 GSPO
comparison in `error_categorization_report.md`:

| Dataset | Split | Config | Index table |
|---|---|---|---|
| Mintaka | test | `configs/config_mintaka_sft_test.json` | `qwen36` (Wikidata) |
| 2WikiMultihopQA | validation | `configs/config_2wiki_sft_test.json` | `qwen36` (Wikidata) |
| WebQSP | test | `configs/config_webqsp_sft_test.json` | `qwen36fbwikiprops` (Freebase) |
| CWQ | test | `configs/config_cwq_sft_test.json` | `qwen36fbwikiprops` (Freebase) |

All four use `Qwen/Qwen3.5-4B` + `adapter_path` (no merged checkpoint),
`prompt_qwen36_angular2_nothink.yaml`, `sentinel: true`, and `n: 200`.

```bash
sbatch hpc/eval_sft_test.sh
```

The script evaluates Mintaka + 2Wiki (Wikidata) in parallel on two GPUs, then
WebQSP + CWQ (Freebase). Logs are `logs/{experiment_name}.out.0`.

For a single dataset use `hpc/eval_mintaka_sft.sh` or `hpc/eval_2wiki_sft.sh`.

## Reference numbers

From `error_categorization/error_categorization_report.md` (all strict
evaluator, 200-example snapshots unless noted):

| Model (Mintaka test) | Examples | Complete | Correct | Correct acc. | Macro F1 | Don't know |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B + SFT (4,000) | 4,000 | 997 (24.9%) | 640 | 16.0% | 16.3% | 0.4% |
| Qwen3.5 4B + SFT + GSPO (ckpt 720) | 200 | 81 (40.5%) | 49 | 24.5% | 16.1% | 3.2% |

| Model (2Wiki validation) | Examples | Complete | Correct | Correct acc. | Macro F1 | Don't know |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B + SFT + GSPO (ckpt 720) | 200 | 170 (85.0%) | 125 | 62.5% | 62.5% | 0.5% |

No WebQSP/CWQ test scores are recorded in the reports, so the new runs establish
the first SFT numbers for those two datasets. Run the corresponding base-model
configs on the same 200 examples if a matched baseline is needed.

## The old 100-example logs (not 4B)

Two 100-example logs exist in Git history (recoverable from local Git LFS):

- `logs/mintaka_test_sft.out.0`
- `logs/twowiki_test_sft.out.0`

Their headers are:

```json
{"model_name": "/data/horse/ws/ripo631h-quokka/sft_merged",
 "dataset": "AmazonScience/mintaka", "n": 100,
 "prompt": "prompts/prompt_qwen36_mini2.txt",
 "date": "25/06/2026 08:00:52 UTC"}
```

| Log | Examples | Macro F1 | Correct acc. | Don't know |
|---|---:|---:|---:|---:|
| `mintaka_test_sft.out.0` | 100 | 17.0% | 17.0% | 0.0% |
| `twowiki_test_sft.out.0` | 100 | 55.0% | 55.0% | 0.0% |

These are **not Qwen3.5-4B results**. They evaluate the old `sft_merged`
checkpoint, whose `config.json` is the 0.8B architecture
(`hidden_size: 1024`), and they use a different dataset repo
(`AmazonScience/mintaka`) and the now-missing `prompt_qwen36_mini2.txt`, so they
are not directly comparable to the current eval setup.

## 4B 100-example slice from the archived full run

`/data/horse/ws/ripo631h-quokka/eval_outputs.zip` contains full 4,000-example
Mintaka logs for vanilla `Qwen/Qwen3.5-4B` and for the older 4B SFT adapter
(`sft_output_qwen35_4b`), both on `rpozzi/mintaka` with
`prompt_qwen36_angular2_nothink.yaml`. Prefix slices:

| Log | Slice | Complete | Correct | Macro F1 | Truncated |
|---|---|---:|---:|---:|---:|
| vanilla 4B | first 100 | 23 (23.0%) | 14 (14.0%) | 0.140 | 76 |
| vanilla 4B | first 200 | 33 (16.5%) | 17 (8.5%) | 0.085 | 163 |
| 4B + old SFT adapter | first 100 | 33 (33.0%) | 25 (25.0%) | 0.250 | 67 |
| 4B + old SFT adapter | first 200 | 56 (28.0%) | 35 (17.5%) | 0.177 | 144 |

Use these only as a sanity check: they are arbitrary prefix slices of the full
files, not dedicated 100-example benchmarks. The current configs use `n: 200`
to stay aligned with the GSPO checkpoint comparison; switch to `n: 100` if you
want the legacy 100-example convention.

## 27B test evaluations

What exists today:

- **Mintaka test: yes.** `/data/horse/ws/ripo631h-quokka/eval_outputs.zip`
  contains `mintaka_test_eval_qwen38_27b.out.0`: 4,000 examples,
  `rpozzi/mintaka` test, `prompt_qwen36_angular2_nothink.yaml`, no adapter.
  13.4% correct (536/4,000), 21.0% complete, mean F1 0.135. Prefix slices:
  23% correct on the first 100 and 18.0% on the first 200.
- **2Wiki: no test labels.** 2Wiki test is unlabeled, so the only usable 27B
  script points at the *train* config (`hpc/test_2wiki_qwen38_27b.sh` uses
  `configs/config_2wiki_qwen38_27b_train.json`); there is no
  `config_2wiki_qwen38_27b_test.json`.
- **WebQSP / CWQ: configs and sbatch scripts exist**
  (`configs/config_{webqsp,cwq}_qwen38_27b_test.json`,
  `hpc/eval_{webqsp,cwq}_test_qwen38_27b.sh`), but I found no completed evaluation
  logs for them in the repo, Git history, or the workspace archive.
- No dedicated 100-example 27B run exists; the Mintaka slices above are the only
  small-sample numbers available.

27B configs on the same preliminary 100-ID sample are now provided (they reuse
the exact sample files and ID keys):

- `configs/config_prelim100_mintaka_27b.json`
- `configs/config_prelim100_2wiki_27b.json` (validation/dev)
- `configs/config_prelim100_webqsp_27b.json`
- `configs/config_prelim100_cwq_27b.json`

They use `Qwen/Qwen3.8-27B`, `device_map: balanced`, 4x24 GiB GPU memory plus
80 GiB CPU offload, and `batch_size: 8`, so each run needs 4 GPUs. Run them one
at a time, e.g.:

```bash
python -m utils.eval --config configs/config_prelim100_mintaka_27b.json
```

## Caveats

- The preliminary 100-example sample is random (`seed=42`) but not stratified;
  per-type cells will be tiny and should be treated as descriptive.
- The 200-example configs take the **first** 200 rows, not a stratified sample.
  The previous reports note the resulting per-type imbalance; compare overall
  metrics and treat small per-type cells as descriptive.
- The evaluator is strict set matching (`utils/eval.py`); normalization only
  happens in the offline analysis, not in the logged `evaluation` block.
- Mintaka/2Wiki use the Wikidata table and WebQSP/CWQ the Freebase table from
  the same PostgreSQL service; the table is selected per config via `tablename`.
- These configs replace the old `prompts/prompt_qwen36_mini2.txt` reference,
  which does not exist in the repository.
