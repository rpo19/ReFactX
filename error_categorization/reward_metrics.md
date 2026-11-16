# Reward and training metrics (W&B guide)

This documents the metrics emitted by `utils/run_trl.py` during GRPO/GSPO
training, what each one means, and how to read it. It is written for monitoring
runs in Weights & Biases; all metric names use `/` so W&B groups them into
collapsible sections automatically.

## How logging works

- **`train/cumulative/*`** is the mean since the start of the run. It is smooth
  but slow to react: a change at step 500 stays hidden under thousands of
  earlier examples. Do not use it for trends.
- **`train/window/*`** is the change since the previous log. On every `on_log`
  the trainer computes `current_total - previous_total` and divides by the
  number of examples in that interval. Use this for trends.
- **`train/group/*`** is computed on the same window. Group statistics are only
  accumulated when the rollout batch divides evenly by `num_generations`; if it
  does not, `train/group/groups_skipped` increments instead of silently
  mis-grouping.
- **`eval/*`** is computed by `evaluate_policy` on each validation pass (greedy
  decoding, so it is deterministic). Eval reward components are fresh per pass.
- **`train/tool/*`** mixes cumulative exact decoder counts with text-detector
  gaps. The decoder counters (`*_calls`) are ground truth; the text counters
  (`*_uses`) are the per-example signal the reward must use.

`logging_steps` (10 in the shipped config) controls the `train/*` window width.
`custom_eval_steps` (30) controls how often `eval/*` appears.

## Reward components

Emitted as `train/cumulative/reward_*`, `train/window/reward_*`, and
`eval/reward_*`.

| Metric | Meaning | How to read |
|---|---|---|
| `reward_mean` | total reward per example | Should rise as the policy learns; compare across runs only at matched steps |
| `reward_structure` | `0.6 ×` (fact block present + exactly one answer block) | Max 0.6. If below ~0.55, format is collapsing |
| `reward_json` | `0.5` when the answer parses as JSON | Max 0.5. Falling means malformed answers |
| `reward_answer` | answer credit `0..1` (normalized match = 1, else strict IoU) | The main learning signal; should track correctness |
| `reward_fact_penalty` | magnitude of the loop-breaker penalty | Should be small; rising means the policy is over-retrieving |
| `reward_tool_bonus` | gated `alpha`/`beta` bonuses | Near zero is normal (see note below) |
| `reward_fact_calls` | mean fact calls per example | Should fall or stay flat; a rise with truncation is a regression |
| `reward_correct_rate` | share of examples with a correct (strict or normalized) answer | Should track `eval/answer_accuracy` |
| `reward_normalized_matches` | share credited only via normalization | Large gap vs strict correctness means normalization carries credit |

**Why `reward_tool_bonus` looks like zero.** It is a multiplier paid only when
the question type, actual tool use, and a correct answer all coincide. With
`alpha = 0.1` on a ~2.1 base, expect ~0.2 on the rare eligible example and
~0.001–0.01 as a per-example mean. Read it with `train/tool/count_accuracy_used`
and the raw counters, not alone.

## Group-relative diagnostics (the most important new signal)

Emitted as `train/group/*`. GRPO/GSPO compute advantages *within* a group of
`num_generations` completions per prompt. If every completion in a group has the
same reward, the advantage is zero and the model learns nothing from that
prompt. `num_generations = 8` makes these numbers meaningful.

| Metric | Meaning | Red flag |
|---|---|---|
| `zero_advantage_frac` | fraction of groups with `max(reward) == min(reward)` | high (> 0.8) → almost no gradient |
| `reward_std` | mean within-group standard deviation | very small → weak signal |
| `pass_at_k` | fraction of groups with at least one correct completion | 0 on a slice → task/reward too sparse there |
| `all_wrong_frac` | fraction of groups where nothing was correct | high → too hard or reward unreachable |
| `all_correct_frac` | fraction of groups where everything was correct | high → too easy, no learning |
| `tool_diverse_frac` | fraction of groups whose completions differ in tool use | 0 → no signal on *when* to call tools |
| `groups_skipped` | batches that could not be grouped | any growth → check batch/`num_generations` divisibility |

Reading guide:

- `zero_advantage_frac` high and `all_wrong_frac` high: the reward is too sparse
  for those prompts. Options: more generations, more partial credit
  (`reward_answer` via IoU/normalization), or curriculum.
- `zero_advantage_frac` high and `all_correct_frac` high: too easy; add harder
  data or stop when eval saturates.
- `tool_diverse_frac` stuck at 0 on count questions: the tool decision is not
  varying, so `alpha` cannot teach timing. Check that `count_pattern` is enabled
  and that the count type is detected (`train/tool/count_type_rate > 0`).

## Tool behavior

| Metric | Meaning | How to read |
|---|---|---|
| `train/tool/count_uses` / `count_examples` | text-detected `<count>` occurrences / completions | matches the reward's view |
| `train/tool/count_calls` | exact decoder-side count invocations | ground truth; use for reporting |
| `train/tool/count_text_decoder_gap` | text minus decoder | persistent growth → echoed/hallucinated `<count>` text |
| `train/tool/sentinel_uses` / `sentinel_calls` | same for `<no further records>` | sentinel only fires on exhausted branches |
| `train/tool/sentinel_text_decoder_gap` | text minus decoder for the sentinel | growth → text heuristic drifting |
| `train/tool/count_type_rate` | share of examples that are counting questions | if 0, count gating is inactive |
| `train/tool/count_usage_on_count_type` | fraction of count questions that used `<count>` | vanilla 4B over-calls (~52%); SFT under-calls (<7%) |
| `train/tool/count_accuracy_used` | accuracy on count questions when the tool was used | should exceed `count_accuracy_unused` |
| `train/tool/count_accuracy_unused` | accuracy on count questions without the tool | if similar to `used`, `alpha` is not buying much |
| `train/tool/sentinel_usage_rate` | fraction of examples using the sentinel | near zero until enumeration gating is populated |

If `count_accuracy_used` is not clearly above `count_accuracy_unused`, set
`reward_alpha` to 0 (the tool is not helping under the current policy) or fix
the count-type detection first.

## Behavior and alignment (`train/window/*`)

| Metric | Meaning | Red flag |
|---|---|---|
| `abstention_rate` | share of answers equal to "I don't know." | near 0 is the current problem, but do not push blindly |
| `confident_wrong_rate` | answered, not correct, not abstained | high → guessing; the report measured 14% for GRPO |
| `normalized_match_frac` | share of correctness credited by normalization | large → inspect normalizer rules |
| `answer_vs_correct_gap` | `mean(answer_credit) - correct_rate` | large (> ~0.15) → IoU/normalization inflating credit |
| `format_saturation` | `mean(structure) / 0.6` | < 0.9 → format collapsing |

## Evaluation metrics (`eval/*`)

| Metric | Meaning |
|---|---|
| `answer_accuracy` | strict lowercased set match (the benchmark score) |
| `normalized_accuracy` | accuracy including semantic normalization (dates, quantities, aliases) |
| `valid_answer_rate` | share of completions that parsed as JSON |
| `format_accuracy` | share with a fact block and exactly one answer block |
| `mean_reward` | mean reward over the eval set |
| `answered_rate` / `abstention_rate` | share answered / share "I don't know." |
| `confident_wrong_rate` | answered but wrong and not abstained |
| `answered_without_triples_rate` | answered with zero retrieved triples (parametric shortcut) |
| `fact_calls_mean`, `fact_calls_p50/p90/p99` | retrieval volume distribution |
| `fact_calls_over_cap_frac` | share above `fact_soft_cap` (20) |
| `completion_len_mean`, `completion_len_p90` | generation length |
| `truncation_rate` | share that hit `max_completion_length` |
| `count_tool_*`, `sentinel_*` | usage counts/rates on the eval set |
| `count_tool_calls`, `sentinel_calls` | exact decoder invocations during eval |
| `tool/count_usage_on_count_type` | tool usage restricted to count questions |
| `tool/count_accuracy_used` / `_unused` | conditional accuracy with/without the count tool |
| `per_type/<type>/accuracy` and `/n` | accuracy per `complexityType` (Mintaka) or `type` (2Wiki) |
| `reward_*` | reward components on the eval set |

Read `answer_accuracy` and `normalized_accuracy` together: a widening gap means
the normalizer is doing more of the work and the strict benchmark is not moving.

## Suggested W&B dashboard

1. `eval/answer_accuracy` + `eval/normalized_accuracy` (+ per-type on a second panel)
2. `train/group/zero_advantage_frac` + `train/group/reward_std`
3. `train/window/reward_answer` + `train/window/reward_correct_rate`
4. `train/window/reward_fact_calls` + `train/window/reward_fact_penalty` + `eval/truncation_rate`
5. `train/tool/count_usage_on_count_type` + `train/tool/count_accuracy_used` vs `_unused`
6. `train/tool/count_text_decoder_gap` + `train/tool/sentinel_text_decoder_gap`
7. `train/window/answer_vs_correct_gap` + `train/window/format_saturation`
8. TRL-native stability metrics: `kl`, `completions/clipped_ratio`, `entropy`, `grad_norm`

## W&B setup notes

- Set `report_to: ["wandb"]` (the default here) and `WANDB_PROJECT`. Tag runs
  with the reward version, algorithm (`gspo`), and `num_generations` so sweeps
  are comparable.
- Because names contain `/`, W&B nests them. Use `wandb.define_metric("eval/*",
  summary="last")` and set summary to the best `eval/answer_accuracy`.
- Log distributions with `wandb.Histogram` if you need the fact-call or reward
  shape; the code reports percentiles, which are cheaper.
- Early in a run, lower `custom_eval_steps` (e.g. 10–15) to get enough eval
  points to see a trend.

## Health checklist

- `train/group/zero_advantage_frac` falling and `reward_std` non-zero.
- `train/window/reward_structure` ≈ 0.6 and `reward_json` ≈ 0.5.
- `train/window/reward_fact_calls` flat or falling; `eval/truncation_rate` falling.
- `train/tool/count_accuracy_used` > `count_accuracy_unused`.
- `train/tool/*_text_decoder_gap` small.
- `eval/normalized_accuracy - eval/answer_accuracy` not growing.
- `train/window/answer_vs_correct_gap` small.
- `kl` and `completions/clipped_ratio` stable (no spikes).

## Limitations

- Per-example reward gating uses text detection because the reward callback only
  receives completions. Decoder counters are aggregate, so they validate the
  text heuristic rather than replace it per example.
- Group diagnostics assume TRL emits each prompt's `num_generations` completions
  contiguously. Guarded by the divisibility check; watch `groups_skipped`.
- Normalized accuracy is opt-in via `use_normalized_reward`; when disabled the
  normalization metrics collapse to the strict ones.
- The sentinel bonus (`beta`) stays dormant unless the dataset is listed in
  `enumeration_datasets` (default WebQSP) and the answer is a multi-item list.
