# Error categorization report

This report applies the taxonomy in `ERROR_CATEGORIZATION_GUIDE.md` to the following evaluation logs:

- `logs/22_sept_2026/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0`
- `logs/22_sept_2026/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1`

The logs were inspected using the guide's JSONL workflow and the table-generation utility:

```bash
python utils/make_latex_tables.py \
  logs/22_sept_2026/twowikimultihop_test_eval_qwen35_4b_grpo_checkpoint720.out.1 \
  logs/22_sept_2026/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0 \
  --output tables.md
```

Each log contains 200 evaluated examples. Mechanical truncation was classified first and was not double-counted as a parser or semantic error.

## Summary

| Primary category | Mintaka | 2Wiki validation | Total |
|---|---:|---:|---:|
| Generation truncation / non-termination | 119 | 30 | 149 |
| Answer-format / parser failure | 0 | 1 | 1 |
| Abstention / unsupported-answer behavior | 3 | 0 | 3 |
| Retrieval failure / missing supporting triple | 6 | 9 | 15 |
| Evidence selection / grounding failure | 0 | 1 | 1 |
| Multi-hop / compositional reasoning failure | 2 | 1 | 3 |
| Entity resolution / ambiguity failure | 1 | 2 | 3 |
| Answer extraction / normalization mismatch | 16 | 31 | 47 |
| Question/dataset or temporal inconsistency | 4 | 0 | 4 |
| **Total failed examples** | **151** | **75** | **226** |
| Correct according to evaluator | 49 | 125 | 174 |

## Model comparison across the 22 September runs

The comparison below uses the four Mintaka test logs in `logs/22_sept_2026/`:

- Qwen3.5 4B vanilla
- Qwen3.5 4B + SFT
- Qwen3.8 27B vanilla
- Qwen3.5 4B + SFT + GRPO, checkpoint 720

The first three runs contain 4,000 examples each. The checkpoint-720 GRPO run contains 200 examples, so percentages are more informative than raw counts and the GRPO result should be interpreted with greater uncertainty. All four runs use the Mintaka test split; the metadata identifies the model and adapter paths used for each run.

### Headline metrics

| Model / variant | Examples | Complete | Correct | Correct accuracy | Macro F1 | Answered accuracy | Don't know |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5 4B vanilla | 4,000 | 562 (14.1%) | 294 | 7.4% | 10.3% | 10.4% | 1.8% |
| Qwen3.5 4B + SFT | 4,000 | 997 (24.9%) | 640 | 16.0% | 16.3% | 16.3% | 0.4% |
| Qwen3.8 27B vanilla | 4,000 | 840 (21.0%) | 536 | 13.4% | 13.0% | 13.2% | 2.0% |
| Qwen3.5 4B + SFT + GRPO (checkpoint 720) | 200 | 81 (40.5%) | 49 | 24.5% | 16.1% | 16.7% | 3.2% |

`Correct accuracy` is the evaluator's overall correctness metric, while `answered accuracy` and `macro F1` are computed over answered examples as reported in the metrics record. The GRPO run has the highest observed completion and correctness rates in this snapshot, but its 200-example evaluation is not directly comparable in statistical precision to the 4,000-example baselines. Its overall macro F1 is approximately tied with SFT and slightly below SFT's, which indicates that the higher correctness rate should not be interpreted as a uniform improvement across all scored examples.

### Completion and truncation behavior

| Model / variant | Incomplete / null prediction | Hit max tokens | Mean generated tokens |
|---|---:|---:|---:|
| Qwen3.5 4B vanilla | 3,438 (85.9%) | 3,123 (78.1%) | 820.5 |
| Qwen3.5 4B + SFT | 3,003 (75.1%) | 3,005 (75.1%) | 844.3 |
| Qwen3.8 27B vanilla | 3,160 (79.0%) | 3,163 (79.1%) | 879.1 |
| Qwen3.5 4B + SFT + GRPO (checkpoint 720) | 119 (59.5%) | 119 (59.5%) | 757.1 |

Completion improved substantially from vanilla Qwen3.5 to SFT, and further in the GRPO checkpoint run. The 27B vanilla model also completes more examples than the 4B vanilla model, but still truncates more often than SFT. The exact `reached_max_tokens` and `answer_complete` counts are not identical for every baseline record, so they should be treated as separate diagnostics rather than assumed to be interchangeable.

### Interpretation across training stages

- **Vanilla Qwen3.5 4B** is the weakest of the four Mintaka runs: it has the lowest completion rate, lowest correctness, and the highest incomplete/null-prediction rate.
- **SFT** produces the clearest large-sample improvement over the same 4B base model: completion rises from 14.1% to 24.9%, correct accuracy from 7.4% to 16.0%, and the don't-know rate falls from 1.8% to 0.4%.
- **Vanilla Qwen3.8 27B** outperforms vanilla 4B on completion and correctness, but underperforms SFT on both metrics despite its larger parameter count. It also has the highest mean generation length, consistent with a greater risk of spending the generation budget before producing a parseable answer.
- **SFT + GRPO checkpoint 720** has the strongest observed completion and correctness on the 200-example Mintaka snapshot. However, its macro F1 is essentially the same as SFT's, and the run has a higher don't-know rate; a larger matched evaluation is needed before claiming a robust improvement.

The 2Wiki validation log currently available in this comparison set is only for the SFT + GRPO checkpoint-720 model. It reports 200 examples, 170 complete answers, 125 correct answers, 62.5% correct accuracy, 62.5% macro F1, and 0.5% don't-know. There is therefore no apples-to-apples 2Wiki comparison across all four model variants in the available logs.

## Typed metrics by dataset

The datasets use different type schemas, so type names must not be compared across datasets:

- **Mintaka:** `complexityType`, with `generic`, `comparative`, `count`, `difference`, `intersection`, `multihop`, `ordinal`, `superlative`, and `yesno`.
- **2Wiki:** `type`, with `compositional`, `comparison`, `bridge_comparison`, and `inference`.

The tables report evaluator correctness accuracy. Each cell is `correct / examples (accuracy)`. The baseline Mintaka runs have the same 400-example allocation for each non-generic type and 800 generic examples. The GRPO Mintaka run has only 200 examples with a different type distribution, so its per-type percentages—especially generic (`n=2`) and yes/no (`n=1`)—are descriptive rather than reliable model comparisons.

### Mintaka correctness by complexity type

| Complexity type | Qwen3.5 4B | Qwen3.5 4B + SFT | Qwen3.8 27B | Qwen3.5 4B + SFT + GRPO |
|---|---:|---:|---:|---:|
| generic | 87/800 (10.9%) | 157/800 (19.6%) | 123/800 (15.4%) | 0/2 (0.0%) |
| comparative | 102/400 (25.5%) | 174/400 (43.5%) | 167/400 (41.8%) | 20/36 (55.6%) |
| count | 30/400 (7.5%) | 39/400 (9.8%) | 29/400 (7.3%) | 6/34 (17.6%) |
| difference | 6/400 (1.5%) | 14/400 (3.5%) | 6/400 (1.5%) | 3/39 (7.7%) |
| intersection | 5/400 (1.3%) | 44/400 (11.0%) | 45/400 (11.3%) | 13/38 (34.2%) |
| multihop | 21/400 (5.3%) | 48/400 (12.0%) | 43/400 (10.8%) | 5/43 (11.6%) |
| ordinal | 5/400 (1.3%) | 34/400 (8.5%) | 36/400 (9.0%) | 2/7 (28.6%) |
| superlative | 6/400 (1.5%) | 14/400 (3.5%) | 6/400 (1.5%) | n/a |
| yesno | 32/400 (8.0%) | 116/400 (29.0%) | 81/400 (20.3%) | 0/1 (0.0%) |

The stable large-sample pattern is that **comparative questions are easiest** for all three 4,000-example runs, while `difference`, `ordinal`, and `superlative` questions are difficult. SFT improves every baseline-comparable type over vanilla Qwen3.5 4B, with particularly large gains for comparative, yes/no, intersection, and ordinal questions. Qwen3.8 27B is stronger than vanilla 4B on most types but does not consistently exceed SFT. The apparently high GRPO scores for intersection, ordinal, and difference should not be over-interpreted because their sample sizes are 38, 7, and 39 respectively.

### Mintaka completion by complexity type

Completion is also type-dependent. For the large-sample runs, the complete-answer rates by type are:

| Complexity type | Qwen3.5 4B | Qwen3.5 4B + SFT | Qwen3.8 27B | GRPO checkpoint 720 |
|---|---:|---:|---:|---:|
| generic | 21.1% | 34.9% | 29.5% | 50.0% (n=2) |
| comparative | 31.5% | 51.5% | 51.3% | 75.0% (n=36) |
| count | 26.5% | 12.5% | 12.8% | 29.4% (n=34) |
| difference | 4.0% | 6.8% | 3.0% | 10.3% (n=39) |
| intersection | 2.5% | 15.5% | 13.5% | 36.8% (n=38) |
| multihop | 16.0% | 34.0% | 25.5% | 53.5% (n=43) |
| ordinal | 4.5% | 18.8% | 13.5% | 28.6% (n=7) |
| superlative | 3.3% | 7.3% | 6.0% | n/a |
| yesno | 10.0% | 33.3% | 25.5% | 0.0% (n=1) |

The count category is an important exception: SFT and Qwen3.8 have higher correctness than vanilla 4B but lower completion rates, indicating that truncation alone does not explain all count failures. For GRPO, comparative and multihop questions have enough examples to be informative; the other type-level completion rates should be treated cautiously.

### 2Wiki validation metrics by type

Only the GRPO checkpoint-720 2Wiki validation run is available:

| 2Wiki type | Examples | Complete | Correct | Correct accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|
| compositional | 90 | 78 (86.7%) | 47 | 52.2% | 52.2% |
| comparison | 48 | 42 (87.5%) | 37 | 77.1% | 77.1% |
| bridge_comparison | 40 | 30 (75.0%) | 29 | 72.5% | 72.5% |
| inference | 22 | 20 (90.9%) | 12 | 54.5% | 54.5% |
| **overall** | **200** | **170 (85.0%)** | **125** | **62.5%** | **62.5%** |

The 2Wiki model performs best on `comparison` and `bridge_comparison`, while `compositional` and `inference` are lower. These results should not be compared directly with Mintaka's `comparative` or `multihop` labels because the datasets define their type fields differently.

## Normalization and alias-handling recommendations

The evaluator currently compares lowercased prediction and reference sets exactly in `utils/eval.py`. This is appropriate for a strict benchmark score, but it treats semantically equivalent forms as errors. The Mintaka records show that normalization should be dataset- and answer-type-aware rather than based on a single global fuzzy matcher.

### Observed Mintaka answer conventions

The following inventory was computed from the 4,000-example vanilla Qwen3.5 Mintaka log and its `gt_answer`/`input_sample.answerText` fields. The categories overlap neither with one another nor with every possible date or quantity form; they are a conservative pattern-based inventory:

| Reference form | Count | Examples |
|---|---:|---|
| Year only (`YYYY`) | 209 | `1981`, `1989`, `2016` |
| Day-month-year (`d-Mon-yy`) | 41 | `2-Aug-70`, `17-Nov-42`, `20-Jan-80` |
| Month-name date (`Month d, yyyy`) | 2 | `December 25, 1776` |
| Day-month-name date (`d Month yyyy`) | 1 | `30 November 1874` |
| Feet/inches | 18 | `6'6\"`, `5'7\"`, `7'2\"` |
| Number plus unit-like text | 37 | `17 weeks`, `115.72 m`, `166 pounds`, `82.8 seconds` |

There is no single default Mintaka date format. In particular, forcing the model to output ISO dates in the prompt would make the output more uniform, but it would not match the existing gold format directly. The evaluator should therefore canonicalize both sides. Questions explicitly asking “which year” generally have year-only references, while birth-date questions often use the abbreviated `d-Mon-yy` format.

### Entity and alias information in Mintaka

Mintaka examples include structured entity information in `input_sample`:

- `questionEntity` contains entity IDs, canonical labels, and the text mention in the question.
- `answerEntity` contains entity IDs and labels when the answer is linked to one or more entities.
- The observed log has non-empty `answerEntity` data for 2,816 of 4,000 examples.
- In 1,773 of those records, the reference answer is exactly one of the supplied entity labels after case-folding; in the remaining records, the answer is often a count or a list summary, so a mismatch with the labels is not automatically an alias.

This provides a useful route to aliases: use the stable Wikidata `Q`-ID as the internal identity, and treat the dataset answer text, `answerEntity.label`, question mention, and approved external labels as surface forms. For example, `Monsters Inc` and `Monsters, Inc.` can resolve to the same entity. However, the local records do not contain a complete alias list. If broader aliases are needed, retrieve Wikidata aliases for the `Q`-IDs offline and cache them; do not perform an uncontrolled network lookup during evaluation. Alias matches should be applied only to entity-valued answers, not to arbitrary strings or numeric counts.

### Recommended evaluator changes

Keep the current strict score, but add a second semantic score rather than silently replacing the benchmark metric:

1. **Robust extraction:** parse `<answer> ... </answer>` with whitespace tolerance, require valid JSON, and verify that the result is a list of strings.
2. **Basic text normalization:** Unicode NFKC normalization, case-folding, whitespace normalization, and carefully controlled punctuation handling.
3. **Date normalization:** parse both the gold and prediction into typed values. Support year-only values, `d-Mon-yy`, natural-language dates, and ISO dates. Compare complete dates exactly. If the reference is year-only and the prediction is a full date with the same year, count it as semantically equivalent; do not automatically give full credit when a full-date reference is reduced to a year.
4. **Quantity normalization:** normalize known units separately. Convert feet/inches, metric lengths, weights, durations, and similar quantities into a common unit with an explicit tolerance. Do not use general string similarity for measurements.
5. **Entity aliases:** when `answerEntity` supplies a Q-ID, compare entity IDs or a curated alias set rather than only surface strings. Keep list cardinality and count answers separate from entity-label matching.
6. **Answer cardinality:** preserve set comparison for unordered multi-answer questions, but penalize extra entities and missing entities. Do not let substring matching turn `Winnipeg` and `Winnipeg, Manitoba` into unconditional equivalents.

Recommended output metrics are `strict_correct`, `normalized_correct`, and `partial_match`. Store the original prediction, normalized prediction, and the normalization rule used so that every newly accepted match remains auditable. This will distinguish true model errors from formatting, alias, precision, and evaluator-normalization mismatches.

### Executable implementation

These rules are persisted as an opt-in implementation in [`utils/answer_normalization.py`](../utils/answer_normalization.py). The module provides:

- `parse_date()` for year-only, ISO, `d-Mon-yy`, and natural-language date forms;
- precision-aware `dates_match()`;
- conservative `parse_quantity()` and `quantities_match()` for supported units and feet/inches;
- Mintaka-only use of `input_sample.answerEntity` labels for non-count entity answers;
- `compare_answers()` for unordered answer lists with cardinality checks and an auditable rule name.

The historical strict evaluator is intentionally unchanged. Call `compare_answers(..., dataset="mintaka", question=..., input_sample=...)` to obtain the additional semantic comparison. Passing `dataset="2wiki"` enables the shared date/quantity rules but does not invent entity aliases because 2Wiki records do not provide Mintaka-style answer Q-IDs. The focused regression tests are in [`tests/test_answer_normalization.py`](../tests/test_answer_normalization.py).

### Prompt guidance

The prompt can reduce—but cannot eliminate—these mismatches. Add answer-type-specific instructions such as:

```text
Return only a JSON list inside <answer>...</answer>.
For a question asking for a year, return YYYY.
For a complete date, return YYYY-MM-DD.
For a measurement, include the numeric value and unit.
Do not add explanations, alternative aliases, or extra entities.
```

The evaluator must still support the historical Mintaka formats because existing references include `2-Aug-70`, year-only answers, feet/inches, and descriptive answer strings. Prompt changes should be evaluated with both strict and normalized metrics.

## Count-tool usage and its effect on count questions

The `<count>` tool is active in all constrained runs, but the models use it at very different rates. Usage is measured by the presence of `<count>` in `full_prediction`, and accuracy is the evaluator's `correct` flag. The GRPO Mintaka numbers are based on only 200 examples, so the percentages are descriptive rather than precise.

| Model (Mintaka) | Examples using `<count>` | Count-type questions using `<count>` (tool-usage ratio) | All count-type questions | Count-type with `<count>` | Count-type without `<count>` |
|---|---:|---:|---:|---:|---:|
| Qwen3.5 4B vanilla (4,000) | 557 (13.9%) | 207/400 (51.8%) | 30/400 (7.5%) | 29/207 (14.0%) | 1/193 (0.5%) |
| Qwen3.5 4B + SFT (4,000) | 58 (1.5%) | 27/400 (6.8%) | 39/400 (9.8%) | 17/27 (63.0%) | 22/373 (5.9%) |
| Qwen3.8 27B vanilla (4,000) | 86 (2.2%) | 44/400 (11.0%) | 29/400 (7.3%) | 14/44 (31.8%) | 15/356 (4.2%) |
| Qwen3.5 4B + SFT + GRPO (200) | 4 (2.0%) | 3/34 (8.8%) | 6/34 (17.6%) | 2/3 (66.7%) | 4/31 (12.9%) |

The **tool-usage ratio** column is the share of count-type questions on which the model emitted `<count>`. It shows the opposite ordering from accuracy: vanilla 4B calls the tool on just over half of count questions but is the least accurate, while SFT calls it on under 7% of count questions yet is far more accurate overall.

### Interpretation

- **Yes, GRPO used the count tool, but rarely.** Only 4 of 200 Mintaka GRPO examples contain `<count>`, and only 3 of those are `count`-type questions. For comparison, vanilla Qwen3.5 4B invoked the tool on 207 of its 400 count-type questions.
- **When any model does use the tool, count accuracy is much higher than when it does not.** This holds for every run: 14.0% vs 0.5% (vanilla), 63.0% vs 5.9% (SFT), 31.8% vs 4.2% (Qwen3.8), and 66.7% vs 12.9% (GRPO).
- **This comparison is correlational, not causal.** The model chooses when to call the tool, so tool-using questions may be the ones where it already recognizes a countable entity-relation pair. The correct interpretation is that the tool is associated with better count accuracy, not yet proven to cause it.
- **GRPO's count improvement is better explained by broad gains than by tool use.** GRPO count accuracy (17.6%) exceeds the baselines, yet it used the tool in only 3 count-type examples. Its overall count gain therefore does not come from heavy tool use.
- **The clearest under-use signal is SFT and GRPO.** Both achieve high accuracy on the few count questions where they call the tool, but call it far less often than vanilla 4B. Encouraging tool use on count questions is a plausible, testable improvement.

A controlled test should prompt or force `<count>` on matched count-type questions and compare accuracy against the same questions answered without the tool; the current logs cannot isolate the causal effect.

The 2Wiki GRPO log contains only one `<count>` call, on a `compositional` question rather than a counting question; the 2Wiki validation split has no `count`-type questions, so the 2Wiki data cannot provide a count-tool ratio.

## Exhausted-retrieval behavior (`<no further records>`)

The intended sentinel mechanism is disabled in every log examined here. The metadata for all five logs records no `sentinel` key, meaning the default `sentinel=False` was used. Consequently, the `<no further records>` string can only appear when a model emits it from the prompt instruction rather than from the tool.

| Log | Examples | Examples containing `<no further records>` | Share |
|---|---:|---:|---:|
| Mintaka Qwen3.5 4B vanilla | 4,000 | 3 | 0.08% |
| Mintaka Qwen3.5 4B + SFT | 4,000 | 0 | 0.00% |
| Mintaka Qwen3.8 27B | 4,000 | 1 | 0.03% |
| Mintaka GRPO checkpoint 720 | 200 | 0 | 0.00% |
| 2Wiki GRPO checkpoint 720 | 200 | 0 | 0.00% |

The few occurrences are concentrated in question types that require enumerating or bounding a subject-relation set:

| Log | Question type | Example | Correct |
|---|---|---|---|
| Mintaka Qwen3.5 4B vanilla | `count` | “How many U.S. states have a population of less than 1 million?” | 0 |
| Mintaka Qwen3.5 4B vanilla | `count` | “How many heads of state attended the 1945 Yalta Conference?” | 0 |
| Mintaka Qwen3.5 4B vanilla | `superlative` | “Which actor has starred in the most movies and TV shows?” | 0 |
| Mintaka Qwen3.8 27B | `count` | “How many books are there in the Torah?” | 0 |

Two observations follow:

1. **The sentinel was not exercised as a tool behavior.** Because `sentinel=False`, these strings are model text, not tool-emitted exhausted-branch markers. This is why the GRPO report shows zero occurrences even though the prompt describes the marker.
2. **Where it appears, it clusters on `count` and other set-enumeration types.** That is consistent with the intended design: knowing when a retrieval set is exhausted matters most when a question requires counting or bounding the full result set. All four observed cases were scored incorrect, so these few occurrences are not evidence of a benefit.

Enabling `sentinel: true` in the checkpoint-720 configs (documented below) is the change needed to test the tool-emitted behavior on `count`, `superlative`, and other complete-enumeration questions.

## GRPO retrieval-tool configuration

The checkpoint-720 GRPO evaluation configs now explicitly set:

```json
"unconstrained_generation": false,
"sentinel": true,
"count_pattern": "<count>"
```

This configuration has the following execution path:

1. `utils/eval.py` reads `sentinel` and `count_pattern` and passes them to `refactx.get_constrained_logits_processor()`.
2. The public wrapper in `refactx/__init__.py` explicitly forwards `sentinel` to `refactx.generate.get_constrained_logits_processor()`.
3. The factory registers `FactGeneration` with `sentinel=True` and registers `CountBranchesGeneration` with the `<count>` pattern.
4. `FactGeneration` emits `<no further records>` when an exhausted retrieval branch is selected, while `CountBranchesGeneration` computes and emits the exact count for a `<count>` call.

The existing 22 September logs were generated before this configuration change: the Mintaka GRPO log contains four `<count>` calls and the 2Wiki GRPO log contains one, but neither contains `<no further records>`. New evaluations must be rerun to measure the effect of sentinel-enabled retrieval; changing the config does not retroactively change those logs.

The count handler was already active in the previous constrained GRPO runs—it is registered by the factory whenever constrained generation is enabled, even when `count_pattern` is not explicitly present in the JSON. The explicit config entry now makes that behavior auditable rather than relying on the default.

## Incentivizing tool use during training

The evidence above argues against rewarding tool use directly. A flat bonus for emitting `<count>` or `<no further records>` would push the policy toward vanilla Qwen3.5 4B's behavior, which calls `<count>` on 51.8% of count questions but has the lowest count accuracy of the four Mintaka runs. The recommended design instead rewards the **outcome** and treats tool use as a small, gated multiplier on top of the existing answer reward.

### Gated multiplier design

The current reward lives in `exact_reward()` in [`utils/run_trl.py`](../utils/run_trl.py) and scores facts (0.3), a single answer block (0.3), valid JSON (0.5), and answer-set IoU. It contains no tool term. The recommended addition is a per-example multiplier:

```text
reward = base_reward * (1 + alpha)   if is_count_question     and used_count    and answer_correct
reward = base_reward * (1 + beta)    if requires_enumeration and used_sentinel and answer_correct
reward = base_reward                 otherwise
```

Properties that make this safe:

- **Correct-answer gating removes random use.** Calling a tool on a question where it does not help cannot improve the answer, so no multiplier is paid.
- **Question-type gating removes off-target use.** The bonus applies only where the tool is designed to help.
- **Multiplier, not additive bonus.** A correct answer already carries a large reward, so a proportional multiplier keeps the incentive modest and lets TRL's group-relative advantage learn *when* to call from differences within the rollout group.
- **Use the structured markers, not raw substrings.** Detect real calls such as `<count> ... = N </count>` and printed sentinels rather than any mention of the tag.

A counterfactual variant is stronger but costlier: for a subset of questions, force the tool on one branch and compare against the unforced branch for the same question, rewarding the decision only when the tool changes a wrong answer into a correct one.

### Condition for the count tool

Use the dataset's own type field as the gate—`complexityType == "count"` for Mintaka, and any equivalent counting category for other datasets. This maps one type to one behavior and is the cleanest signal.

### Condition for the sentinel: list-answer datasets such as WebQSP

WebQSP answers are frequently lists of entities, which makes it a good source of sentinel training signal, but the gate must be narrower than "the answer is a list".

- **Good sentinel use:** one subject-relation pair whose full object set is the answer, so the model must know the enumeration is complete before answering (for example, "what languages does X speak").
- **Weak sentinel use:** multi-hop compositions where the answer list comes from composing small relations rather than exhausting one relation.

The recommended metadata flag is therefore `requires_full_enumeration`, approximated from single-relation fan-out where the object count matches the answer cardinality, rather than a generic list-answer test. This avoids rewarding sentinel calls that do not matter and reduces the risk of terminating a branch the model still needs.

Use a smaller `beta` than `alpha`. The sentinel has a real downside: firing it on a branch that is still needed can destroy an otherwise-correct answer, so its usage should be discouraged relative to the count tool.

### Prerequisites and sequencing

Before either incentive can help:

1. **Enable the sentinel in training rollouts and verify it fires.** Every log examined here still has `sentinel=False`; if the sentinel never triggers during rollouts, the reward term is dead. [`utils/run_trl.py`](../utils/run_trl.py) previously did not forward `sentinel` to `get_constrained_logits_processor()` at all, so training always used the default `sentinel=False`. It now accepts a `--sentinel` flag (and a `sentinel` config key) and forwards it, together with `count_pattern`, to both constrained-processor construction sites (the training batch and the per-example validation batch).
2. **Add the metadata at data preparation time.** Derive `is_count_question` for Mintaka and `requires_full_enumeration` for WebQSP, and carry them into the training examples.
3. **Keep training and evaluation protocols aligned.** If sentinel use is rewarded in training, evaluation should also run with `sentinel: true`.

Run the two tools as separate experiments so their effects can be attributed:

| Experiment | Data | Gate | Reward |
|---|---|---|---|
| A | Existing datasets | `is_count_question` | count multiplier `alpha` on correct answers |
| B | Add WebQSP and other list-answer datasets | `requires_full_enumeration` | smaller sentinel multiplier `beta` on correct answers |

For both, track the tool-usage ratio **and** task accuracy together. The goal is accuracy rising while the usage ratio stays moderate, rather than the ratio climbing toward vanilla's over-calling behavior.

### WebQSP representation caveat

WebQSP answers are Freebase entities, so a list may be returned as labels, IDs, or both, and answer sets can be large. Before using correctness as the reward gate, confirm the evaluator handles that representation; otherwise the multiplier would be withheld from correct-but-differently-formatted answers. The opt-in normalization module described above is the relevant lever.

## Alternating tool calls with free reasoning

A plausible hypothesis is that models answer better when they alternate constrained tool calls with free reasoning—retrieve, reason, retrieve again—rather than emitting long runs of adjacent calls with no prose in between. This section measures that pattern.

The measurement is implemented in [`utils/measure_alternation.py`](../utils/measure_alternation.py):

```bash
python utils/measure_alternation.py logs/22_sept_2026/*.out.0 --min-free 4
```

The utility splits each `full_prediction` into constrained tool spans (`<fact>...</fact>`, `<count>...</count>`, and `<no further records>`) and the free-text spans between them, excludes the final `<answer>` block, and uses whitespace-delimited words as a token proxy. Only spans **between two consecutive tool calls** are considered: the lead-in before the first call and the lead-in to `<answer>` are excluded. An example counts as **alternating** when the smallest between-call span is at least `min_free` words. This is a strict definition: an example with one bare call and several well-reasoned calls still counts as non-alternating.

### Headline comparison

The comparison below uses the checkpoint-720 GRPO logs, because they are the only runs where the constrained tool path is active in both datasets.

| Dataset | Alternating (min 4 free words) | Accuracy, alternating | Accuracy, not alternating | Overall accuracy |
|---|---:|---:|---:|---:|
| Mintaka GRPO | 38/200 (19.0%) | 23.7% | 24.7% | 24.5% |
| 2Wiki validation GRPO | 106/200 (53.0%) | 54.7% | 71.3% | 62.5% |

**The two datasets disagree.** On Mintaka at the smallest threshold, alternating and non-alternating examples are statistically indistinguishable (23.7% vs 24.7%). On 2Wiki, alternation is clearly *worse*: 54.7% vs 71.3%.

### Threshold sensitivity

The two datasets move in opposite directions as the criterion becomes stricter, and the Mintaka comparison changes sign with the threshold:

| `min_free` | Mintaka alternating | Mintaka acc. alternating | Mintaka acc. not alternating | 2Wiki alternating | 2Wiki acc. alternating | 2Wiki acc. not alternating |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 38 (19.0%) | 23.7% | 24.7% | 106 (53.0%) | 54.7% | 71.3% |
| 4 | 38 (19.0%) | 23.7% | 24.7% | 106 (53.0%) | 54.7% | 71.3% |
| 8 | 31 (15.5%) | 29.0% | 23.7% | 103 (51.5%) | 53.4% | 72.2% |
| 12 | 21 (10.5%) | 38.1% | 22.9% | 87 (43.5%) | 60.9% | 63.7% |

On Mintaka, the alternating group only pulls ahead once the free span is required to be long (38.1% vs 22.9% at `min_free=12`). On 2Wiki, heavier reasoning correlates with *lower* accuracy at `min_free` 2–8, and the two groups converge near 61–64% at `min_free=12`. The threshold sweep therefore flips the sign of the Mintaka comparison, which is itself an important caution.

### Per-type breakdown (`min_free=4`)

Mintaka:

| Type | Alternating ratio | Acc. alternating | Acc. not alternating |
|---|---:|---:|---:|
| comparative | 8.3% | 33.3% | 57.6% |
| multihop | 37.2% | 25.0% | 3.7% |
| intersection | 21.1% | 50.0% | 30.0% |
| count | 20.6% | 0.0% | 22.2% |
| difference | 7.7% | 0.0% | 8.3% |
| ordinal | 14.3% | 0.0% | 33.3% |

2Wiki:

| Type | Alternating ratio | Acc. alternating | Acc. not alternating |
|---|---:|---:|---:|
| compositional | 83.3% | 56.0% | 33.3% |
| bridge_comparison | 35.0% | 71.4% | 73.1% |
| inference | 50.0% | 45.5% | 63.6% |
| comparison | 12.5% | 16.7% | 85.7% |

### Interpretation

- **The headline result is mixed, not a clean win for alternation.** Mintaka is a wash at low thresholds and only favors alternation at a strict threshold; 2Wiki consistently favors less interleaving.
- **Sign flips with the threshold.** On Mintaka, alternation looks neutral at `min_free=2` and favorable at `min_free=12`. That instability is a strong warning that the split is capturing something other than a stable reasoning style—most likely question difficulty.
- **The 2Wiki `comparison` case is the striking one.** Only 12.5% of comparison questions alternate, yet those that do score 16.7% versus 85.7% for those that do not. Here, extended interleaved reasoning looks like a symptom of the model being on a wrong or ambiguous path, not a cause of failure.
- **Multihop is the clearest positive case for alternation.** Mintaka `multihop` alternates at 37.2% and scores 25.0% vs 3.7%. That is the pattern the hypothesis predicts: interleaving retrieval with reasoning helps compose multiple hops.
- **`count` does not benefit from more free reasoning here.** Mintaka `count` alternating examples score 0.0%; this is consistent with the earlier finding that the count tool, not longer prose, is what raises count accuracy.
- **This is observational, not causal.** The number of tool calls and the amount of reasoning are chosen by the model, so they co-vary with question difficulty and with whether retrieval is already failing. A high-accuracy alternating group may simply be questions where the model recognized a solvable multi-step path.
- **Mintaka has only 200 GRPO examples**, so per-type cells are small and should be read as directional.

### How to use this measurement

1. Run the utility on a candidate run before and after any prompt or reward change:

   ```bash
   python utils/measure_alternation.py logs/22_sept_2026/<run>.out.0 --min-free 4 --json alternation.json
   ```

2. Track the alternating ratio and both accuracy columns together; a change in the ratio alone is not evidence of improvement.
3. Do not reward alternation directly, for the same reason as the tool-use bonuses above: it is a format property whose usefulness is dataset- and type-dependent. If it is trained on at all, gate it on the question types where the correlation is positive, and confirm with a controlled comparison.

## Free reasoning versus constrained generation

This section studies how much free reasoning the models produce relative to constrained tool calls, whether interleaving helps, and how both interact with question complexity. The analysis is implemented in [`utils/analyze_alternation.py`](../utils/analyze_alternation.py):

```bash
python utils/analyze_alternation.py logs/22_sept_2026/<run>.out.0 --min-free 4 --json analysis.json
```

Each `<fact>` block contains exactly one triple in these runs, so the number of fact calls equals the number of retrieved triples. "Free words" are whitespace-delimited words in the between-call spans only; the lead-in before the first call and the tail before `<answer>` are reported separately.

### Volume balance: free words versus constrained words

| Run | Mean fact calls | Mean free words | Mean constrained words | Free/constrained ratio |
|---|---:|---:|---:|---:|
| Mintaka Qwen3.5 4B vanilla (n=3,648 with calls) | 28.9 | 23.8 | 318.6 | 0.12 |
| Mintaka Qwen3.5 4B + SFT (n=3,994) | 24.8 | 75.2 | 266.7 | 0.40 |
| Mintaka Qwen3.8 27B (n=3,999) | 26.7 | 51.4 | 296.5 | 0.35 |
| Mintaka GRPO 720 (n=200) | 20.3 | 95.6 | 209.9 | 0.69 |
| 2Wiki GRPO 720 (n=200) | 5.3 | 63.7 | 65.2 | 0.85 |

The models differ sharply in balance. Vanilla Mintaka is almost purely retrieval-driven (median free words 0). SFT and Qwen3.8 27B add moderate reasoning, and GRPO reasons the most. The 2Wiki run retrieves far less and reasons proportionally much more, which fits its different question structure.

### Accuracy falls monotonically with the number of fact calls

The single strongest pattern in the data is that correctness collapses as fact calls increase. This holds in every Mintaka run:

| Fact calls | Qwen3.5 4B vanilla | Qwen3.5 4B + SFT | Qwen3.8 27B | GRPO 720 |
|---|---:|---:|---:|---:|
| 1-2 | 59.2% | 71.2% | 70.8% | 68.0% |
| 3-5 | 50.0% | 62.6% | 65.8% | 59.1% |
| 6-10 | 42.1% | 56.9% | 53.2% | 52.9% |
| 11-20 | 6.4% | 15.4% | 16.7% | 30.0% |
| 21+ | 0.0% | 0.5% | 0.3% | 0.9% |

Examples needing more than ~20 fact calls are essentially never answered correctly. This is the clearest expression of the retrieval-loop failure mode described in the truncation section: the model keeps asking for facts instead of concluding. It also implies that fact-call count is a strong difficulty or "stuck" proxy, which matters for interpreting alternation.

### Is alternation useful?

Splitting each fact-count bucket into alternating (≥4 free words between calls) and non-alternating examples shows that alternation is beneficial only while the model is still succeeding:

| Fact calls | SFT alternating acc. | SFT non-alternating acc. | 27B alternating | 27B non-alternating |
|---|---:|---:|---:|---:|
| 1-2 | 51.5% | 74.8% | 45.7% | 76.5% |
| 3-5 | 59.6% | 63.5% | 61.9% | 68.4% |
| 6-10 | 48.4% | 58.3% | 45.9% | 58.1% |
| 11-20 | 1.2% | 32.9% | 10.5% | 19.9% |
| 21+ | 0.0% | 0.5% | 0.0% | 0.3% |

Alternation does not rescue hard examples. In the 1-2 fact bucket it actually scores *worse* than flat retrieval, and in the 11-20 bucket it is far worse. The consistent reading is that extended interleaved reasoning is a **symptom of being lost**, not a fix: the model reasons most when it is already on the wrong path. The 2Wiki `comparison` result in the previous section is the extreme version of the same effect.

The one remaining useful framing is "reason after N facts". The data support reasoning between a *small* number of calls (1-5) but not as a general rule, and never as a rescue strategy for large fan-outs.

### Accuracy peaks at a moderate amount of reasoning per fact

| Free words per fact | Qwen3.5 4B vanilla | SFT | 27B |
|---|---:|---:|---:|
| 0-5 | 2.8% | 7.8% | 5.7% |
| 5-15 | 9.8% | 17.4% | 26.9% |
| 15-40 | 5.0% | 16.0% | 24.8% |
| 40+ | 4.5% | 7.7% | 0.0% |

A moderate reasoning budget per fact—roughly 5 to 15 words—is associated with the best accuracy, and very heavy reasoning per fact is consistently worse. Combined with the fact-count result, the pattern is: reason briefly after a few calls, retrieve selectively, and stop.

### Complexity by question type

Difficulty tracked by baseline type accuracy, with reasoning volume in the GRPO run:

| Tier | Mintaka types | Typical baseline accuracy | GRPO free words/fact |
|---|---|---:|---:|
| Easy | comparative | 25-44% | 4.0 |
| Medium | generic, yes/no, count | 8-20% | 0.9-8.0 |
| Hard | multihop, difference, intersection, ordinal, superlative | 1-13% | 2.4-12.4 |

The model spends the most reasoning per fact on `multihop` (12.4 words/fact in GRPO) and the least on `yes/no` (1.5) and `comparative` (4.0). This is the expected shape—multi-hop composition requires interleaved reasoning—but it does not translate into better outcomes for the hardest types, consistent with the symptom-not-cause reading.

For 2Wiki, reasoning volume is highest for `bridge_comparison` (11.5 words/fact) and `compositional` (10.7), which are also the types with the most facts retrieved.

### How many facts before reasoning? (best N)

If the guidance is "reason freely after every N facts", the logs can estimate N by grouping examples by the **longest run of adjacent fact calls** (calls with fewer than one free word between them). A run of N means the model emitted N fact calls back-to-back before reasoning.

| Longest adjacent run | Vanilla 4B | SFT | 27B | Mintaka GRPO 720 | 2Wiki GRPO 720 |
|---|---:|---:|---:|---:|---:|
| 1 (reason every call) | 17.2% | 24.2% | 36.8% | 31.0% | 54.6% |
| **2** | **48.2%** | **52.9%** | **58.7%** | **47.2%** | **78.9%** |
| 3-5 | 39.1% | 35.7% | 35.1% | 45.5% | 53.3% |
| 6-10 | 34.1% | 32.5% | 22.4% | 18.8% | 50.0% |
| 11+ | 0.2% | 1.2% | 0.2% | 1.4% | 0.0% |

**N = 2 is the best value, and it is consistent across all four model variants and both datasets.** Reasoning after every single call (N = 1) is clearly worse than pulling two facts and then reasoning, and accuracy degrades monotonically beyond N = 2 up to the collapse at N ≥ 11.

Two caveats:

- This is observational. The run length is chosen by the model and correlates with difficulty; the 11+ bucket is dominated by the unsolvable retrieval-loop cases.
- The N = 2 advantage is large enough and consistent enough to be worth a controlled test: prompt the behavior explicitly ("retrieve two facts, then reason") and compare against the current policy on matched questions.

### Abstention: the models almost never say "I don't know"

A second concern is calibration. The model should recognize when the retrieved facts do not support an answer and abstain, rather than guessing. The **abstention profile** (implemented as `abstention_profile` in the analysis utility) measures this directly.

| Run | Abstain rate | Answered rate | Accuracy when answered | Confident-wrong rate | Answered with 0 triples |
|---|---:|---:|---:|---:|---:|
| Mintaka vanilla 4B (4,000) | 0.9% | 14.1% | 52.3% | 5.9% | 137 (23.4% acc.) |
| Mintaka SFT (4,000) | 0.5% | 24.9% | 64.2% | 8.4% | 4 (50.0% acc.) |
| Mintaka 27B (4,000) | 1.9% | 21.0% | 63.8% | 5.7% | 5 (60.0% acc.) |
| Mintaka GRPO 720 (200) | 2.0% | 40.5% | 60.5% | 14.0% | 0 |

Findings:

- **Abstention is essentially absent.** Vanilla abstains on 0.9% of examples, SFT 0.5%, 27B 1.9%, GRPO 2.0%. The models almost always commit to an answer, correct or not.
- **Confident-wrong answers dominate the answered set.** For GRPO, 28 of 81 answered examples (14.0% of all examples, 34.6% of answered) are wrong and not abstentions. For SFT the figure is 8.4% of all examples.
- **The model answers even without evidence.** Vanilla gives 137 answers with zero retrieved triples and still scores 23.4% on them, i.e. it answers from parametric knowledge when the tool returns nothing. This is the behavior the "insufficient evidence" rule is meant to prevent.
- **The "empty" bucket is truncation, not abstention.** The large `prediction: null` rates (86% vanilla, 59.5% GRPO) are generation-limit failures, not deliberate abstentions. Any abstention metric must therefore key on an explicit "I don't know" answer, which is what the profile does, and must not count `null` predictions as abstentions.

This is why "reason over the facts found, then decide" matters: the missing behavior is not longer reasoning, it is an explicit **sufficiency check** that can produce abstention. The design implication is to reward abstention when the retrieved facts genuinely do not support an answer, and specifically to penalize confident wrong answers produced with zero or clearly irrelevant triples. As with the other shaping terms, gate the reward so that blanket abstention is not profitable: abstention should be rewarded only on examples where no correct answer is derivable from the retrieved facts.

### Should abstention trigger after a fixed number of fact calls?

A natural design is: "retrieve up to K facts, and if the model still cannot answer, say 'I don't know'." The data show that fact-count is a real signal, but also that a hard numeric cutoff does **not** improve the correctness metric.

Simulating a hard cutoff (abstain whenever an example used more than K fact calls) across the baseline runs:

| Threshold K | SFT correct sacrificed of 640 | SFT accuracy before | SFT accuracy after | 27B correct sacrificed of 536 |
|---|---:|---:|---:|---:|
| facts > 5 | 207 | 16.0% | 10.8% | 161 |
| facts > 10 | 88 | 16.0% | 13.8% | 80 |
| facts > 15 | 39 | 16.0% | 15.0% | 25 |
| facts > 20 | 14 | 16.0% | 15.7% | 9 |
| facts > 25 | 4 | 16.0% | 15.9% | 1 |

The reason accuracy cannot improve is mechanical: under the current metric a wrong answer and an abstention both score zero. Abstaining converts a wrong answer into "don't know" and changes the score by nothing, while any abstention on an example the model *would* have answered correctly is a pure loss. It is therefore **impossible to gain strict accuracy from a fact-count cutoff**; the only choices are how much to lose.

What the cutoff does buy is calibration, honesty, and compute:

- It converts confident-wrong answers into calibrated abstentions, which matters for any metric that penalizes wrong answers more than abstentions, or that scores reliability.
- It stops the runaway retrieval loops that consume the generation budget and cause the truncation failures documented earlier.

Given that, the threshold should be set to sacrifice almost no correct answers:

- **K = 20 gives 15.7% vs 16.0% for SFT and 13.2% vs 13.4% for 27B.** It removes roughly 2,670 and 2,876 wrong answers respectively while sacrificing only 14 and 9 correct ones.
- **K = 15 is a reasonable conservative choice** (15.0% SFT, 12.8% 27B) that stops loops somewhat earlier.
- **K below 10 is not recommended.** It sacrifices 80-200 correct answers and drops SFT accuracy to about 11-14%.

A fixed number is also the wrong shape of rule. The principled trigger is structural, and this project already has the mechanism for it:

- **Retrieval exhaustion** - the `<no further records>` sentinel means every object for a subject-relation pair has been returned. That is the natural cue for "no more evidence exists", far better than a magic count.
- **Repetition / no new information** - the runaway loops repeat the same relations with no new objects (visible as very high fact counts, e.g. 21+ with 0.3% accuracy). Repeating an already-answered relation for the same subject is a direct "nothing new to learn" signal.

### Prompt setting and training signal

Yes, add the behavior to the prompt and train it, but express it as a **stopping condition, not a number**. Concretely:

```text
If you have retrieved all objects for the relevant subject-relation pairs
(including any <no further records>) and the retrieved facts still do not
support an answer, respond exactly:
<answer>
["I don't know."]
</answer>
Do not repeat a subject-relation pair that has already been fully retrieved.
```

Pair this with the sentinel enabled in training (documented above) so the model can actually observe exhaustion, and with the gated reward idea: pay for abstention only when the retrieved facts genuinely do not support an answer, and only after the relevant relations are exhausted. This avoids the degenerate strategy of abstaining early to dodge hard questions, which the data show would cost real accuracy.

If a numeric guard is still wanted as a safety net, use a large value (K ≈ 20) rather than a small one, so the guard stops loops without truncating solvable multi-hop work.

### Cap-conditioned abstention reward

A concrete proposal is: put a maximum fact count in the prompt, and reward the model when it abstains after reaching that cap. The data make this more defensible than a plain fact-count cutoff, with one caveat that must be handled.

**Why the cap is a usable trigger.** Reaching the cap is not a random event; it strongly indicates the example is unsolvable for this model with this retrieval:

| Run | Examples above 15 facts | Correct among them | Examples above 20 facts | Correct among them |
|---|---:|---:|---:|---:|
| Mintaka SFT | 2,986 | 39 (1.3%) | 2,683 | 14 (0.5%) |
| Mintaka 27B | 3,144 | 25 (0.8%) | 2,885 | 9 (0.3%) |
| **Mintaka GRPO 720** | 122 | 5 (4.1%) | 106 | 1 (0.9%) |
| **2Wiki GRPO 720** | 20 | 0 (0.0%) | 7 | 0 (0.0%) |

So a cap-conditioned reward has a very low false-positive rate: it would withhold an answer on examples that were almost never going to be answered correctly anyway. If the goal is calibrating "I don't know" behavior, this is the right neighborhood.

The GRPO model that would actually receive this training has a slightly different profile and matters for the choice of K. Its fact distribution is compressed (Mintaka: 25 examples at 1-2, 22 at 3-5, 17 at 6-10, 30 at 11-20, 106 at 21+; 2Wiki: no examples above 11-20 except 7 at 21+), and its 11-20 band accuracy is 30%-the highest of any model. Consequently:

| Cap K | Mintaka GRPO accuracy | Correct sacrificed | 2Wiki GRPO accuracy | Correct sacrificed |
|---|---:|---:|---:|---:|
| facts > 10 | 19.5% (from 24.5%) | 10 | 62.0% (from 62.5%) | 1 |
| facts > 15 | 22.0% (from 24.5%) | 5 | 62.5% (unchanged) | 0 |
| facts > 20 | 24.0% (from 24.5%) | 1 | 62.5% (unchanged) | 0 |

Two GRPO-specific conclusions follow. First, **K = 20 is even more clearly the right choice for GRPO**: it sacrifices a single correct answer (0.5 accuracy points) while removing 105 confident-wrong ones. Second, **a cap of 15 costs GRPO more than it costs the baselines** (2.5 accuracy points versus 1.0-2.2), because GRPO answers 30% of 11-20 examples correctly. Capping below 20 would discard part of GRPO's strongest mid-range region. For 2Wiki the fact distribution is compact enough that any K >= 12 is effectively a no-op.

**The caveat: the shortcut.** If the reward is paid for "abstain after hitting the cap", the cheapest policy is to burn facts quickly to reach the cap and collect the reward instead of trying to answer. This is a classic reward hack and would actively destroy the 1-5 fact cases that currently carry the highest accuracy. Mitigations, in order of importance:

1. **Keep the abstention reward strictly smaller than the correctness reward**, so answering correctly always dominates reaching the cap.
2. **Grant the abstention reward once per example**, and only after genuine retrieval rather than padding.
3. **Prefer gating on evidence, not just the count.** Reward abstention when the retrieved facts do not contain the answer (e.g. none of the `answerEntity` labels appear in the retrieved triples), and use the cap only as a secondary loop-breaker. This removes the incentive to inflate the count.
4. **Enforce the cap structurally in generation rather than by prompt alone** if you want deterministic behavior: when the constrained decoder has emitted K fact calls, stop offering the fact pattern. Then the only options are answer or abstain, and the reward selects between them. That is cleaner than relying on the model to obey a number in the prompt.

**Is it worth doing?** Yes, with mitigations, because the current behavior is the opposite extreme: 0.5-2.0% abstention with persistent confident-wrong answers, and the model even answers with zero retrieved triples. The cap gives a structured moment to ask "do I actually have enough?", which is exactly the sufficiency check that is missing. It will not raise strict accuracy (a wrong answer and an abstention both score zero), but it improves calibration and stops the runaway loops. Treat it as a reliability and efficiency change, and pair it with a cost-sensitive metric that distinguishes wrong answers from abstentions so the benefit is measurable.

**Which number.**

- **K = 20 (recommended default).** For the baselines only 0.3-0.5% of above-cap examples are correct, and for the GRPO model being trained it sacrifices exactly one correct answer (0.5 accuracy points) on Mintaka and none on 2Wiki. It leaves the 11-20 fact band, which carries 15-30% accuracy (30% for GRPO), available for real work, while roughly 2,700-2,900 baseline confident-wrong answers become abstentions.
- **K = 15 (aggressive).** Safe for the baselines (0.8-1.3% of above-cap examples are correct) but costs GRPO 2.5 accuracy points because it truncates GRPO's strongest mid-range band. Use only if token cost strongly dominates.
- **K below 10 is not recommended**, because the 11-20 band contains a meaningful share of correct answers (SFT 6-10 facts is 56.9% and 11-20 is 15.4%; GRPO 11-20 is 30.0%), so a small cap truncates solvable work.
- **Per-dataset tuning is optional.** Since 2Wiki GRPO has essentially no examples beyond 11-20, a single global K = 20 is safe for both datasets and avoids two different prompt numbers.

Set the prompt guideline and the structural cap to the same value to avoid an inconsistency between what the model is told and what the decoder enforces.

### Worked example: enumeration is correct behavior, and the sentinel is what is missing

The example above (GRPO Mintaka example 78, *"Which 2020 Christopher Nolan movie does not star Matt Damon?"*, answer `Tenet`) was initially mis-framed as a redundant verification loop. That reading was wrong and is corrected here.

The question requires establishing that **no** cast member is Matt Damon, so enumerating the full cast is exactly the correct behavior, not padding:

```text
I need to find Christopher Nolan movies released in 2020 and check which ones do not star Matt Damon.

<fact> <Oppenheimer (film)> <publication date> <2023-07-21T00:00:00Z> .</fact>
   This is not a 2020 movie. Let me search for Christopher Nolan movies from 2020.

<fact> <Tenet> <description> <Wikimedia disambiguation page> .</fact>
   This is a disambiguation page. Let me look for the film specifically.

<fact> <Tenet (film)> <publication date> <2020-09-03T00:00:00Z> .</fact>
   This is a 2020 Christopher Nolan film. Now I need to check if Matt Damon stars in it.

<fact> <Tenet (film)> <cast member> <John David Washington> .</fact>
...15 cast members total...
   The cast list does not include Matt Damon.

<fact> <Tenet (film)> <director> <Christopher Nolan> .</fact>
<fact> <Tenet (film)> <genre> <Science fiction film> .</fact>
...and further confirmations...

<answer>
["Tenet"]
</answer>
```

Correct reading of this trace:

- **The cast enumeration is necessary.** A "does not star" question cannot be answered from a partial cast list, so retrieving all 15 cast members is the required behavior rather than a wasted loop. This is a case where a plain enumeration of facts, with short reasoning between groups, is the right pattern.
- **The `<no further records>` sentinel is exactly what this trace needed.** The model could not know its cast list was complete. That uncertainty is why it continued into `director`, `genre`, and `production company` lookups, hunting for confirmation. If the fact call had answered the exhausted cast relation with `<no further records>`, the model would have had a definitive completeness signal and could have stopped and answered. This is the strongest concrete argument in the logs for enabling the sentinel in training.
- **The cap is not what this example needs.** The answer became derivable as soon as the cast list was complete, so a cap at 20 would still allow a correct answer here; the fix for the extra calls is completion signalling, not a lower cap.

### Learned reasoning placement

Two observations from the logs suggest the GRPO checkpoint has already learned non-trivial structure about *where* reasoning belongs, and that this is worth preserving rather than overriding:

1. **The amount of free reasoning is already calibrated-ish.** The model reasons briefly between small fact groups and stops reasoning entirely within a tight enumeration (adjacent cast-member calls carry no prose). In the traces, prose clusters where a decision is being made and disappears where simple enumeration is sufficient. The measured sweet spot of roughly 5-15 free words per fact is consistent with what the model already does on the questions it answers well.
2. **The placement is task-sensitive and mostly correct.** Reasoning appears around entity resolution (disambiguation pages, "let me find the specific film"), around deciding what to retrieve next, and around the final composition step. Within a single relation whose objects are being enumerated for a set check, the model correctly emits facts back-to-back. This matches the per-type data, where `multihop` and `disambiguation`-heavy questions carry the most reasoning words per fact while `yes/no` and simple enumerations carry the least.

Implications for training:

- **Do not force uniform alternation.** Penalizing adjacent fact calls would fight this correct enumeration pattern. Any alternation shaping should be confined to decision points, which the model already detects.
- **Keep the free-reasoning amount as a soft preference, not a target.** A 5-15 word-per-fact band is a useful prior, but the model's own placement is reasonable when it is on the right track.
- **The missing piece is completion awareness, not reasoning quantity.** The sentinel and the sufficiency check address the actual gap, whereas adding or removing prose does not.

### Practical conclusions

1. **Track fact-call count as a first-class signal.** It predicts failure more strongly than any reasoning-style measure and identifies the retrieval-loop cases directly.
2. **Do not reward alternation directly.** It is negatively associated with success on the hardest buckets and in 2Wiki. If trained at all, gate it on easy-to-medium types and small fan-outs, and verify with a controlled comparison.
3. **Target moderate reasoning per fact (5-15 words).** Both the extremes underperform, so a length-based shaping term, if used, should be a soft preference rather than a hard constraint.
4. **Treat long interleaved reasoning as a diagnostic of being stuck.** A run that shifts mass toward high fact counts and high reasoning per fact is regressing, even if it looks more elaborate.
5. **Use N = 2 for "reason every N facts".** It is the best-performing run length across all four models and both datasets, and a controlled test is worth running.
6. **Add an explicit sufficiency check that can abstain.** Abstention is near zero today while confident-wrong answers persist, so the missing behavior is deciding when the facts are insufficient, not reasoning longer.
7. **Trigger abstention on exhaustion, not on a magic fact count.** A fact-count cutoff cannot raise strict accuracy (wrong and abstention both score zero) and only risks sacrificing correct answers. Use the `<no further records>` sentinel and repeated-relation detection as the stopping cue; keep K ≈ 20 only as a loop-breaking guard.
8. **Preserve correct enumeration and completion signalling.** Full enumeration of a relation is required for "does not / all / none" questions, and the GRPO model already places reasoning sensibly around decision points. The gap is knowing that a relation is exhausted, which the sentinel provides.

## Generation truncation / non-termination

This is the dominant failure mode:

- **Mintaka:** 119/200 examples
- **2Wiki validation:** 30/200 examples
- **Total:** 149/400 examples

All 149 cases had:

- `reached_max_tokens: true`
- `answer_complete: false`
- `prediction: null`
- generation at the configured token limit

These cases were treated as generation failures rather than parser or reasoning failures.

## Mintaka errors

### Answer extraction / normalization mismatches

Representative examples:

- **Example 9:** `July 9, 1981` vs reference `1981`.
- **Example 154:** `August 2, 1970` vs `2-Aug-70`.
- **Example 164:** `170 cm` vs `5'7"`.
- **Example 172:** `United States` vs `USA`.

These answers are often semantically equivalent but were rejected by the evaluator's normalization rules.

### Abstention / unsupported-answer behavior

Examples 22, 38, and 180 returned “I don't know” despite apparently sufficient evidence. Example 181 was classified as a retrieval failure because the retrieved evidence did not contain the required child-count fact.

### Retrieval failure / missing supporting triple

- **Example 47:** Retrieved evidence exposes four Michael Bay-directed *Transformers* films while the reference expects five.
- **Example 57:** No retrieved cast link connects Robert De Niro to a *Godfather* film.
- **Example 71:** Retrieved award facts are incomplete or inconsistent with the reference count of 11.
- **Example 156:** Only two Uma Thurman children are retrieved while the reference expects three.
- **Example 179:** Retrieved evidence identifies Michael Biehn instead of Arnold Schwarzenegger as the relevant *Terminator* actor.

### Multi-hop / compositional reasoning failure

- **Example 30:** The dates are correctly retrieved, but the model reverses the requested “before/after” relationship.
- **Example 176:** The model retrieves Liam Neeson's birth date but returns the date instead of calculating the requested age.

### Entity resolution / ambiguity failure

- **Example 155:** “Lead actress of *Emma*” is resolved to Kate Beckinsale. The retrieved evidence supports one child, while the reference expects two, making the title/lead-actor interpretation ambiguous.

### Question, dataset, or temporal inconsistency

- **Example 23:** Retrieved heights support a tie, while the reference selects Brad Pitt.
- **Example 53:** Retrieved evidence suggests four *John Wick* films involving Ian McShane, while the reference expects three.
- **Example 141:** Both *The Aviator* and *The Wolf of Wall Street* plausibly satisfy the wording, but only the latter is retrieved.
- **Example 193:** The model computes age using a 2026 date and returns 79, while the reference says 74; this appears time-dependent.

## 2Wiki validation errors

### Answer extraction / normalization mismatches

Representative examples:

- **Example 3:** `June 12, 1516` vs `12 June 1516`.
- **Example 22:** `American` vs `United States`.
- **Example 60:** `Winnipeg` vs `Winnipeg, Manitoba`.
- **Example 181:** `Aarhus` vs `Århus`.
- **Example 200:** `Immacolata` vs `Immaculata`.

Many of these outputs are arguably correct but differ from the reference in alias, date format, geographic granularity, punctuation, or expected answer scope.

### Answer-format / parser failure

- **Example 113:** The model returns both `Southfield High School` and `New York University Tisch School of the Arts`, while the reference expects only the latter institution. This is an over-answer or multi-answer extraction failure.

### Retrieval failure / missing supporting triple

- **Example 10:** Retrieved evidence conflicts with the supporting evidence about Cleopatra IV's father.
- **Example 42:** Retrieved evidence gives `Kingdom of Württemberg`, while the supporting evidence says `German`.
- **Example 81:** Retrieved evidence gives Hisham Zaman's birthplace as `Iraq`, while the supporting evidence/reference says `Kurdish`.
- **Example 104:** The correct Viktoriya Zeynep Güneş entity is not retrieved; the model sees unrelated Viktoriya entries and abstains.
- **Examples 114, 121, 148, and 150:** Supplied triples conflict with the supporting evidence or reference, including conflicts such as India vs. Geumgwan Gaya and China vs. Qatar.

### Entity resolution / ambiguity failure

- **Example 56:** `Otrag` is resolved as `Otago`, yielding New Zealand instead of the gold entity OTRAG in Germany.
- **Example 75:** `ZTR Control Systems` is resolved as the unrelated Czech company `Ztronic` instead of the Canada-linked gold entity.

### Evidence selection / grounding failure

- **Example 106:** The supplied triple explicitly states that Anthony Weiner's employer is Chuck Schumer, but the model answers with the U.S. House of Representatives based on his position.

### Multi-hop / compositional reasoning failure

- **Example 91:** The model compares death years rather than comparing the subjects' lifetimes; the supporting evidence shows that Edward Dearle lived longer.

## Interpretation

The most important issue is generation length: 149 of the 226 failed examples never reached a complete answer. Any analysis of model reasoning should therefore separate those cases from the 77 non-truncated failures.

Among non-truncated failures, the largest group is answer normalization or extraction mismatch, especially in the 2Wiki validation log. Retrieval conflicts and missing supporting triples are also common, particularly when the supplied triples disagree with the dataset's supporting evidence or reference answer.

The normalization category is evaluator-sensitive. Several model outputs are semantically reasonable but differ from the reference in formatting, aliases, granularity, or answer cardinality. Similarly, retrieval and dataset-issue classifications should be interpreted cautiously when the retrieved triples contradict the ground-truth evidence.
