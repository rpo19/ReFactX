# Qwen3.5-0.8B ReFactX SFT/RL experiments

Living log of the 0.8B work on 2×T4. All accuracy numbers use the project's
strict evaluator on the fixed 100-ID sample files
(`configs/samples/*_100_seed42.json`), with the first N rows used unless noted.

## Summary of best results

| Benchmark | Vanilla 0.8B | Best adapter | Budget | N | Accuracy |
|---|---:|---|---:|---:|---:|
| Mintaka | 16% | `sft_output_08b_combined/checkpoint-100` | 12 | 50 | **38%** |
| 2Wiki | 18% | `sft_output_08b_combined/checkpoint-100` | 12 | 50 | **54%** |
| WebQSP | 2% (4% on 100) | `sft_output_08b_clean/checkpoint-200` | 40 | 50 | **18%** |
| CWQ | 12% (9% on 100) | `sft_output_08b_combined/checkpoint-100` | 12 | 50 | **20%** |

The combined adapter is best on Mintaka/2Wiki/CWQ. WebQSP is still better with
the earlier no-mask teacher adapter, so the recommended setup is per-benchmark.

## 0. What was wrong initially

- SFT targets ended at `</answer>` and never contained the chat end token
  `<|im_end|>`, so the model never learned to stop.
- The original prompt was ~1,540 tokens (three long examples), leaving little
  generation room; 21% of teacher traces were truncated by `max_length`.
- The "supported" SFT mixture contained 662 traces that were strictly wrong on
  the benchmark (mostly 2Wiki date-format mismatches) and ~27% traces with
  malformed exhausted-retrieval calls such as `<relation <no further records>`.
- 72% of Mintaka failures were retrieval loops/truncation, not retrieval errors.
- T4 training used bf16/DoRA/long sequences, causing OOM and ~46 s/optimizer step.

## 1. Compact prompt

- File: `prompts/prompt_compact.yaml`
- Same `<fact>`, `<count>`, `<answer>` tool syntax as the training traces, but
  ~490 tokens instead of ~1,540.

## 2. Clean, correct-only SFT data (`logs/sft_training_mix_v3.jsonl`)

Builder: `utils/build_clean_sft.py`

```bash
python -m utils.build_clean_sft \
  --input logs/sft_training_mix_supported.jsonl.gz \
  --output logs/sft_training_mix_v3.jsonl \
  --prompt prompts/prompt_compact.yaml \
  --max-tokens 1280 \
  --require-correct
```

What it does:

- keeps only `evaluation.correct == 1` (drops 662 strictly wrong traces);
- repairs malformed sentinel calls and drops remaining malformed fact blocks;
- rebuilds each prompt with the compact template;
- appends `<|im_end|>` to every target;
- drops sequences longer than 1,280 tokens.

Kept **5,561** traces:

| Dataset | Traces |
|---|---:|
| Mintaka | 2,748 |
| 2Wiki | 1,558 |
| CWQ | 798 |
| WebQSP | 457 |
| synthetic count | 1,000 |
| natural | 4,561 |

## 3. How `sft_output_08b_clean/checkpoint-200` was obtained

Base: `Qwen/Qwen3.5-0.8B`, LoRA r=16, alpha=32, dropout=0.05, plain LoRA
(DoRA disabled), gradient checkpointing, fp16 (T4-safe).

Config: `configs/sft_config_qwen35_08b_clean.json`

- data: `logs/sft_training_mix_v3.jsonl`
- max_length 1280, batch 2/GPU, grad-accum 8 (effective batch 32 across 2 GPUs)
- lr 5e-5, 2 epochs declared, but trained to 200 optimizer steps (~1.2 epochs)
- save/eval every 100 steps

Initial run:

```bash
accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
  --config configs/sft_config_qwen35_08b_clean.json \
  --output-dir sft_output_08b_clean
```

This saved `checkpoint-100`. It was then resumed:

```bash
accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
  --config configs/sft_config_qwen35_08b_clean.json \
  --output-dir sft_output_08b_clean \
  --resume-from-checkpoint sft_output_08b_clean/checkpoint-100 \
  --max-steps 200
```

`checkpoint-200` is the adapter after those 200 optimizer steps (final training
loss ~0.49). It is the no-mask teacher adapter used both for the WebQSP result
above and as the initialization for the combined adapter.

## 4. Decoder-side retrieval budget

Change in `refactx/generate.py`: after `max_tool_calls` completed fact/count
calls the decoder forces `\n<answer>\n` and then blocks new `<fact>`/`<count>`
starts. Exposed through `max_tool_calls` in eval/RL configs.

Effect at checkpoint-100 (30 samples, Mintaka/2Wiki):

| Run | Mintaka | 2Wiki |
|---|---:|---:|
| no budget | 20.0% | 46.7% |
| budget 16 | **26.7%** | **50.0%** |

Truncation dropped from 17/30 to 7/30 (Mintaka) and 11/30 to 7/30 (2Wiki).

## 5. Masking experiments

The original `locate_fact_ranges` searched for `Fact:`, but the current data uses
`<fact>`, so masking was a no-op for the new traces. Fixed in
`utils/sft_train.py`:

- supports `<fact>...</fact>`, `<count>...</count>` and legacy `Fact:` lines;
- token offset mapping instead of per-prefix re-tokenization;
- `--mask-triples` masks whole tool calls;
- `--mask-objects` masks only the object slot;
- per-sample `mask_mode` (`none` / `objects` / `triples`) for mixed datasets.

Mask-triples-only adapter (`sft_output_08b_masktriples/checkpoint-165`,
1 epoch on the clean data):

| Benchmark | No-mask ckpt-200 | Mask-triples ckpt-165 |
|---|---:|---:|
| Mintaka | 32% (50) | 26.7% (30) |
| 2Wiki | 52% (50) | 40% (30) |

Conclusion: masking whole triples alone is worse. It is useful as a first stage
when followed by an unmasked/grounded stage, which matches the earlier
SFT-masked + RL-unmasked observation.

## 6. Answer-conditioned, KB-verified path data

Builder: `utils/build_path_sft.py`

For each RoG train question:

1. search the `graph` field for a directed path from a question entity to each
   gold answer;
2. map every edge to a fact that exists in the actual Freebase trie
   `qwen36fbwikipops` (the `qwen36fbwikiprops` spelling in older configs/logs
   does not exist locally);
3. keep the question only when every answer is reachable through verified facts.

Run in 8 shards per dataset:

```bash
python -m utils.build_path_sft \
  --dataset rmanluo/RoG-webqsp --output logs/path_shards/webqsp_${S}.jsonl \
  --num-shards 8 --shard ${S}
# same for rmanluo/RoG-cwq with --max-samples 500
```

Yields:

- `logs/path_webqsp.jsonl`: **1,116** traces (39.5% of 2,826 questions)
- `logs/path_cwq.jsonl`: **1,388** traces (34.7% of the 4,000 sampled)
- dropped questions: no graph path, no KB subject, or an edge absent from the KB

Path-only adapter (`sft_output_08b_path/checkpoint-80`, initialized from the
masked adapter) was bad: WebQSP 8%, CWQ 6% on 50 samples — it learned to emit a
single fact and answer its object.

## 7. Combined adapter (best for Mintaka/2Wiki/CWQ)

Dataset: `logs/sft_combined_v3_path.jsonl` = 5,561 clean teacher traces
(`mask_mode="none"`) + 2,504 verified path traces (`mask_mode="objects"`).

Config: `configs/sft_config_qwen35_08b_combined.json`

- init: `sft_output_08b_clean/checkpoint-200`
- lr 1e-5, max_length 1280, batch 2/GPU, grad-accum 8, 2×T4 DDP
- 100 optimizer steps
- output: `sft_output_08b_combined/checkpoint-100`

```bash
accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
  --config configs/sft_config_qwen35_08b_combined.json \
  --output-dir sft_output_08b_combined
```

Results (50 samples):

| Benchmark | Init ckpt-200 | Combined ckpt-100 |
|---|---:|---:|
| Mintaka | 32% | **38%** |
| 2Wiki | 52% | **54%** |
| WebQSP | **18%** | 12% |
| CWQ | 16% | **20%** |

## 8. Recommended configs

- Mintaka: `configs/config_prelim100_mintaka_sft08b_clean_budget.json`
- 2Wiki: `configs/config_prelim100_2wiki_sft08b_clean_budget.json`
- WebQSP: `configs/config_prelim100_webqsp_sft08b_clean_budget.json`
- CWQ: `configs/config_prelim100_cwq_sft08b_clean_budget.json`

Each uses the best adapter above and the budget that was evaluated (12 for
Mintaka/2Wiki/CWQ, 40 for WebQSP).

## 9. Tests

```bash
pytest -q tests/test_sft_masking.py tests/test_retrieval_budget.py \
  tests/test_features.py tests/test_efficiency.py tests/test_run_trl_reward.py
```

Last run: **95 passed**.

## 10. Not done yet

- No GRPO/RL run on the combined adapter.
- No isolated mask-objects-only adapter evaluation.
- WebQSP path data currently hurts the combined adapter; a stricter semantic
  path filter or a WebQSP-specific mix is the next thing to try.

## 11. Dataset quality assessment (added after reviewing `kqa_pro.md`)

This is a practical assessment for *this* ReFactX setup (constrained generation
over the local prefix tree), not a general judgement of the datasets.

| Dataset | Strengths | Main quality/compatibility problems |
|---|---|---|
| 2WikiMultihopQA | Clean template-based multi-hop questions; answers are Wikidata entities; best observed accuracy (54%) | Date/answer formatting is inconsistent (`May 26, 1900` vs `20 October 2008`); strict exact matching rejects otherwise correct dates |
| Mintaka | Human-written, many question types, widely used | Low compatibility with a fact-only decoder: count-with-filter, difference, ordinal, superlative, comparative, multilingual, long-tail; many answers are derived or absent from the KB; strict date/answer normalization issues |
| WebQSP | Real user questions, Freebase, standard in literature | KB coverage and answer completeness problems; Freebase is outdated and many answers are MIDs/labels not directly usable; list answers; only ~39.5% of train questions had a fully verified path to all answers |
| CWQ | Compositional, standard in literature | Noisiest in practice: ambiguous questions, incomplete/noisy answer sets, heavy KB/schema mismatch; only ~34.7% of the sampled train questions had a fully verified path; low and unstable scores |
| KQA Pro | Gold KoPL program, gold SPARQL, verifiable single answer, 94k train questions, count/verify/multi-hop/qualifier coverage | Small KB (~17k entities, 376k base triples) so the model can memorise it; synthetic/paraphrased language; every question has 10 choices and the answer is always among them, so passing `choices` leaks the task |

### Which datasets are lowest quality?

- **Lowest in practice for ReFactX: CWQ.** It is not useless — it is heavily
  used in the literature — but it has the worst combination of noisy answers,
  incomplete answer sets and KB mismatch. Report it, but do not tune the whole
  pipeline on it.
- **Mintaka is not the “dirtiest” dataset, but it is the least compatible** with
  a pure fact-constrained decoder. Many questions need arithmetic/filtering or
  answer forms that ReFactX cannot derive from triples alone. It is hard for
  structural reasons, not only label noise.
- **2Wiki is the highest-quality benchmark** for fact-grounded multi-hop in this
  setup; its main issue is answer/date normalisation, not the graph facts.
- **WebQSP is medium:** cleaner questions than CWQ, but KB coverage and answer
  completeness still cap the achievable score.

### KQA Pro as a training set

KQA Pro is likely a **better SFT source than Mintaka**, but not a drop-in
replacement for the literature benchmarks:

- its gold program (`program`) and SPARQL make it possible to generate traces
  deterministically and verify every fact/count against the index;
- it has enough data (94k train) and the right skills (multi-hop, count,
  verify, qualifier);
- but the KB is small, so mix it with Mintaka/2Wiki/WebQSP/CWQ for
  generalisation;
- do **not** pass `choices` during SFT, and evaluate both with and without
  choices to separate reasoning from 10-way selection;
- use `val` for evaluation (test has no public labels).

Recommended next experiment: build 5–10k KQA Pro traces from the gold KoPL
program, verify facts against `kqapro_qwen36.pkl`, mix with the existing clean
teacher traces, and continue SFT from `sft_output_08b_combined/checkpoint-100`.
Keep WebQSP/CWQ as held-out literature benchmarks.

## 12. Replacing Mintaka with KQA Pro for counts and type study

Prepared pipeline (details in `kqa_pro.md` section 13):

- sample balanced KQA Pro train ids (`utils/sample_kqa_pro.py`);
- generate teacher traces with Qwen3.8-27B on HPC
  (`hpc/eval_kqa_pro_train_qwen38_27b.sh`, 4 GPUs);
- build verified SFT traces (`utils/build_kqa_sft.py`), analyse question types
  (`utils/analyze_kqa_types.py`), and mix them into the clean teacher data while
  dropping Mintaka count traces (`utils/mix_kqa_sft.py`);
- continue 0.8B SFT from `sft_output_08b_combined/checkpoint-100`
  (`hpc/sft_train_kqa_08b.sh`).

KQA Pro teacher generation hides the 10 choices on purpose. The local KQA Pro
index must be built with the teacher tokenizer; Qwen3.5/3.6/3.8 share the token
vocabulary (verified for 3.5-0.8B and 3.6-27B), so the same index works for the
0.8B student.
