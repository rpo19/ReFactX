# Error categorization guide for Mintaka and 2Wiki evaluations

This guide accompanies the following JSONL evaluation logs:

| File | Model / variant |
|---|---|
| `mintaka_test_eval_qwen35_4b.out.0` | Qwen3.5 4B |
| `mintaka_test_eval_qwen35_4b_sft.out.0` | Qwen3.5 4B + SFT |
| `mintaka_test_eval_qwen38_27b.out.0` | Qwen3.8 27B |
| `mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0` | Qwen3.5 4B + SFT + GSPO (checkpoint 720) |
| `twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1` | Qwen3.5 4B + SFT + GSPO (checkpoint 720), 2Wiki validation |

The model labels above follow the experiment description supplied with these files. When comparing runs, verify the metadata record—especially `model_name`, `adapter_path`, `dataset`, `dataset_split`, and `experiment_name`—rather than relying only on the filename.

The three larger baseline Mintaka logs are also JSONL evaluation files and should be reviewed with the same extraction procedure. They may be tens of megabytes larger because each example contains the prompt, retrieved triples, the full model generation, and the full formatted sample. Avoid opening any of these files in an editor or printing them with `cat`. Use the commands below to extract only the fields needed for analysis.

## File layout

The checkpoint-720 Mintaka and 2Wiki files contain 202 JSON objects, one per line. The larger baseline Mintaka files use the same JSONL record convention; use the metadata/metrics inspection command below to confirm their exact record counts before assuming the same line numbers.

| Lines | Record | Meaning |
|---:|---|---|
| 1 | Metadata | Experiment, dataset, model, generation settings |
| 2–201 | Example records | 200 evaluated examples |
| 202 | Macro metrics | Aggregate metrics for the 200 examples |

For the checkpoint-720 files, records are zero-indexed in Python as follows: `rows[0]` is metadata, `rows[1:-1]` are examples, and `rows[-1]` is the metrics record. The baseline files should be handled the same way, but determine their length programmatically rather than assuming 202 records.

### Dataset distinction

- Mintaka is configured as `dataset_split: "test"` and has `gt_answer`/`answerText` for the examples in this log.
- The 2Wiki log is configured as `dataset_split: "validation"`; it is the ground-truth-bearing 2Wiki validation set used here. Do not infer that a separate 2Wiki **test** split has ground truth. For a no-ground-truth split, correctness metrics cannot be interpreted as answer accuracy; review answer format, evidence, and reasoning instead.

## Record schema

Each example record has these main fields:

| Field | Use |
|---|---|
| `question` | Normalized question shown to the model. |
| `gt_answer` | Reference answer when available. |
| `input_sample` | Original dataset example. Contains dataset-specific fields such as Mintaka `answerText`, or 2Wiki `context`, `supporting_facts`, `evidences`, and `answer`. |
| `triples` | Retrieved knowledge-base triples supplied during generation. |
| `prediction` | Parsed answer, usually a list of answer strings; `null` means no answer was parsed. |
| `full_prediction` | Complete model output. Read this only for a selected example. |
| `answer_complete` | Whether the evaluator found a complete/parseable answer. |
| `evaluation` | Per-example scores: `precision`, `recall`, `f1`, `correct`, and `dont_know`. |
| `reached_max_tokens` | Whether generation stopped at the configured token limit. |
| `new_tokens_generated` | Number of generated tokens. |
| `prompt` / `full_sample` | Very large diagnostic strings. Avoid extracting these during triage unless needed. |

The final metrics record contains keys such as `macro_precision`, `macro_recall`, `macro_f1`, `macro_correct_accuracy`, `macro_dont_know`, and the corresponding `macro_answered_*` values.

## Safe inspection commands

Run these commands from the repository root. They parse the JSONL but do not print the large prompts or full generations.

For generated summary tables, `utils/make_latex_tables.py` can produce Markdown tables as well as LaTeX tables. Markdown is the default; use `--output tables.md` to save Markdown, or add `--latex` (and use a `.tex` output path) for LaTeX. For example:

```bash
python utils/make_latex_tables.py logs/*.out --output tables.md
```

### 1. Show metadata and final metrics

```bash
python - <<'PY'
import json
from pathlib import Path

for path in [
    Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0"),
    Path("logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1"),
]:
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    print(f"\n{path}")
    print("metadata:", json.dumps(rows[0], ensure_ascii=False, indent=2))
    print("metrics:", json.dumps(rows[-1], ensure_ascii=False, indent=2))
PY
```

### 2. Print a compact table of all examples

This is the recommended first pass. It prints one short row per example and omits prompts and generations. Change the `paths` list in the command to compare any or all of the five logs, for example:

```python
paths = [
    Path("logs/mintaka_test_eval_qwen35_4b.out.0"),
    Path("logs/mintaka_test_eval_qwen35_4b_sft.out.0"),
    Path("logs/mintaka_test_eval_qwen38_27b.out.0"),
    Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0"),
    Path("logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1"),
]
```

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0"),
    Path("logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1"),
]

for path in paths:
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    print(f"\n### {path}")
    print("idx\tcomplete\tcorrect\tdk\tmax_tokens\tprediction\tgt_answer\tquestion")
    for idx, row in enumerate(rows[1:-1], 1):
        ev = row.get("evaluation", {})
        def short(value, limit=100):
            text = "" if value is None else str(value).replace("\n", " ")
            return text if len(text) <= limit else text[:limit - 3] + "..."
        print("\t".join([
            str(idx),
            str(row.get("answer_complete")),
            str(ev.get("correct")),
            str(ev.get("dont_know")),
            str(row.get("reached_max_tokens")),
            short(row.get("prediction")),
            short(row.get("gt_answer")),
            short(row.get("question")),
        ]))
PY
```

The `idx` in this table is the example number starting at 1. The corresponding physical JSONL line is `idx + 1` because line 1 is metadata.

### 3. Compare headline metrics across model variants

Use the final JSON object from each file rather than scanning the full generations:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    Path("logs/mintaka_test_eval_qwen35_4b.out.0"),
    Path("logs/mintaka_test_eval_qwen35_4b_sft.out.0"),
    Path("logs/mintaka_test_eval_qwen38_27b.out.0"),
    Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0"),
]
for path in paths:
    with path.open() as f:
        rows = [json.loads(line) for line in f]
    meta, metrics = rows[0], rows[-1]
    print(path.name, {
        "records": len(rows),
        "model": meta.get("model_name"),
        "adapter": meta.get("adapter_path"),
        "split": meta.get("dataset_split"),
        "macro_f1": metrics.get("macro_f1"),
        "macro_correct_accuracy": metrics.get("macro_correct_accuracy"),
        "macro_dont_know": metrics.get("macro_dont_know"),
    })
PY
```

Compare errors by example number only when the examples are aligned across runs; verify the questions or dataset IDs first. A changed sampling order or dataset revision can make row numbers refer to different questions.

### 4. Select only failures or truncations

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0")
with path.open() as f:
    rows = [json.loads(line) for line in f]

for idx, row in enumerate(rows[1:-1], 1):
    ev = row.get("evaluation", {})
    if (
        ev.get("correct") != 1
        or row.get("reached_max_tokens")
        or row.get("prediction") is None
    ):
        print(json.dumps({
            "idx": idx,
            "question": row.get("question"),
            "gt_answer": row.get("gt_answer"),
            "prediction": row.get("prediction"),
            "evaluation": ev,
            "answer_complete": row.get("answer_complete"),
            "reached_max_tokens": row.get("reached_max_tokens"),
            "new_tokens_generated": row.get("new_tokens_generated"),
            "triple_count": len(row.get("triples", [])),
        }, ensure_ascii=False))
PY
```

Change the `path` value to inspect the other dataset.

### 5. Inspect one selected example in detail

Replace `wanted = 17` with the example number from the compact table. This prints the question, reference, parsed prediction, evaluation, retrieved triples, and full generation—but only for that one example.

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0")
wanted = 17
with path.open() as f:
    rows = [json.loads(line) for line in f]
row = rows[wanted]

for key in [
    "question", "gt_answer", "prediction", "evaluation",
    "answer_complete", "reached_max_tokens", "new_tokens_generated",
    "triples", "full_prediction",
]:
    print(f"\n--- {key} ---")
    print(json.dumps(row.get(key), ensure_ascii=False, indent=2))
PY
```

For 2Wiki, also inspect the dataset-specific evidence fields:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("logs/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1")
wanted = 17
with path.open() as f:
    rows = [json.loads(line) for line in f]
row = rows[wanted]
for key in ["question", "gt_answer", "prediction", "evaluation", "triples", "input_sample"]:
    print(f"\n--- {key} ---")
    print(json.dumps(row.get(key), ensure_ascii=False, indent=2))
PY
```

### 6. Use `jq` for a quick compact projection

`jq` can stream one JSON object at a time. The `.[1:-1]` form below is safe for these files because the first and last records are metadata and metrics.

```bash
jq -c '.[1:-1][] | {
  question, gt_answer, prediction, answer_complete,
  evaluation, reached_max_tokens, new_tokens_generated,
  triple_count: (.triples | length)
}' logs/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0
```

The Python commands are preferable for very large files because they make it easier to select a single record and avoid accidentally printing `prompt`, `full_sample`, or `full_prediction` for every example.

## Suggested error taxonomy

Assign one **primary** category per failed example, and optionally one or more secondary categories. Use the evidence in the record rather than guessing from the score alone.

1. **Correct / no error**
   - `evaluation.correct == 1` and the answer is supported by the available evidence.

2. **Generation truncation / non-termination**
   - `reached_max_tokens == true`, usually `new_tokens_generated == 1024` in these runs.
   - `prediction == null` or the answer block is incomplete.
   - Check whether the model spent too many tokens listing facts or reasoning instead of answering.

3. **Answer-format / parser failure**
   - The generation appears to contain the right answer, but `prediction` is `null` or differs because the required answer format was not followed.
   - Check the final `<answer>` block in `full_prediction`, including JSON/list syntax, extra prose, spelling, and multiple answers.

4. **Abstention / unsupported-answer behavior**
   - The model returns an explicit “I don't know” or an empty answer despite sufficient evidence.
   - Distinguish intentional abstention from truncation; use `evaluation.dont_know` and `reached_max_tokens`.

5. **Retrieval failure / missing supporting triple**
   - The required fact or relation is absent from `triples`, so the model could not reasonably derive the answer from the supplied knowledge base.
   - For 2Wiki, compare with `input_sample.supporting_facts` and `input_sample.evidences` when available.

6. **Evidence selection or grounding failure**
   - Relevant triples are present, but the model ignores them, uses an unrelated triple, or asserts a fact not supported by `triples`.
   - Mark hallucinated facts separately as a secondary note when useful.

7. **Multi-hop / compositional reasoning failure**
   - The needed intermediate entity is found, but the model fails to apply the next relation, reverses a relation, stops after one hop, or composes the wrong entities.
   - This is especially important for 2Wiki questions such as “X of the Y of Z”.

8. **Entity resolution / ambiguity failure**
   - The model selects the wrong entity, confuses similarly named entities, or fails to resolve aliases, parentheticals, or pronouns.

9. **Answer extraction / normalization mismatch**
   - The semantic answer is arguably correct, but differs in capitalization, punctuation, aliases, multi-answer ordering, granularity, or another evaluator-normalization detail.
   - Record the model answer and reference answer exactly in notes.

10. **Question or dataset issue**
    - Ambiguous, malformed, inconsistent, or apparently mislabeled question/reference.
    - Use this sparingly and include a short explanation; do not use it merely because the model was wrong.

## Recommended review worksheet

For each selected error, record:

| Field | Value |
|---|---|
| Dataset / file | Mintaka or 2Wiki validation |
| Example number | The `idx` from the compact table |
| Primary category | One taxonomy category above |
| Secondary category | Optional |
| Question | From `question` |
| Reference | From `gt_answer`, if available |
| Parsed prediction | From `prediction` |
| Complete? | `answer_complete` |
| Truncated? | `reached_max_tokens` and `new_tokens_generated` |
| Evidence available? | Whether `triples` contain the needed fact(s) |
| Evidence used? | Whether `full_prediction` uses the relevant fact(s) |
| Notes | One or two sentences explaining the decision |

For consistency, first classify mechanical failures (truncation and parser/format failures), then evidence availability, then reasoning/entity errors. Do not classify a no-ground-truth example as “incorrect” solely because there is no `gt_answer`; mark it as unscored and review the generation qualitatively.

## Snapshot of these two logs

These counts are a starting point for triage, not a substitute for manual categorization:

| File | Complete | Incomplete | Parsed prediction is `null` | Hit max tokens | Correct |
|---|---:|---:|---:|---:|---:|
| Mintaka | 81 | 119 | 119 | 119 | 49 |
| 2Wiki validation | 170 | 30 | 30 | 30 | 125 |

The Mintaka incomplete cases and the 2Wiki incomplete cases all hit the 1,024-token generation limit in this run. The remaining incorrect cases should be separated into retrieval, grounding, multi-hop reasoning, entity resolution, and answer-normalization errors using the selected-example procedure above.
