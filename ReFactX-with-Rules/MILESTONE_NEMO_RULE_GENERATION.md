# Nemo rule generation milestone

**12 November 2026**

The Stage 5 pipeline generated and executed rules with Qwen3.5-4B on a dedicated CPU server. On the sampled pilot, each constrained arm reproduced the gold relation in **14 of 15** generations. The unconstrained arm reproduced it in **8 of 15**. The milestone establishes that constrained generation, Nemo execution, and independent tuple comparison work together on the synthetic ownership graph.

## What changed

The first server run exposed a response-boundary bug. The logits processor searched the whole chat transcript for a marker inside the user message. Qwen's role tokens followed the marker and sent the grammar into a dead state. The processor then returned the original logits. Extraction separately expected a fresh marker in the generated response. A rule with aggregates in the body also reached Nemo and reproduced a Rust panic.

The current runner gives each arm the same bare-rule output contract and starts masking at the first generated token. It records incomplete rules and mask failures explicitly. A tokenizer-only preflight verifies all five gold rules through both masks with the actual chat template. The evaluator checks the rule shape, output signature, and body predicates, requests only the candidate head from Nemo, and distinguishes rejected text, engine failures, and wrong tuple sets. Runtime rule additions preserve prior relations and commit only after successful reasoning and indexing.

The Nemo-backed QA checks also pass against the 1,110 existing reference answers. This establishes backend parity for those checks. Evaluating QA with **model-generated** rules remains a separate experiment.

## Server pilot

The runs used `Qwen/Qwen3.5-4B`, CPU, and float32. The greedy smoke run used one generation per task and arm. The sampled run used three generations per task and arm. The sampled runner defaults were seed 42, temperature 0.7, top-p 0.8, top-k 20, and a 160-token cap. Each rule was scored by exact equality of its output tuple set to the independently computed reference, with numeric values rounded to six decimal places.

| Run | Arm | Correct | Executed with wrong results | Rejected | Generation or engine errors |
|---|---|---:|---:|---:|---:|
| Greedy | `unconstrained` | 4/5 | 1 | 0 | 0 |
| Greedy | `grammar_distractors` | 5/5 | 0 | 0 | 0 |
| Greedy | `full` | 5/5 | 0 | 0 | 0 |
| Sampled | `unconstrained` | 8/15 | 2 | 5 | 0 |
| Sampled | `grammar_distractors` | 14/15 | 1 | 0 | 0 |
| Sampled | `full` | 14/15 | 1 | 0 | 0 |

| Sampled task | Unconstrained | Grammar + distractors | Full |
|---|---:|---:|---:|
| `owner_of_owner` | 1/3 | 2/3 | 2/3 |
| `owner_count` | 3/3 | 3/3 | 3/3 |
| `top_stake` | 2/3 | 3/3 | 3/3 |
| `total_owned` | 2/3 | 3/3 | 3/3 |
| `majority_owner` | 0/3 | 3/3 | 3/3 |

The greedy run took **9.5 minutes** and the sampled run took **29.1 minutes**. Sampled mean generation time was **26.29 seconds** unconstrained, **43.68 seconds** with grammar and distractors, and **42.86 seconds** with full constraints. The corresponding mean mask times were **0**, **15.38**, and **14.83 seconds**. Masking accounted for most of the additional generation time. The sampled arms averaged 28.0, 28.4, and 28.4 generated tokens, respectively.

These figures come from the two server summaries. The recorded CSV basenames are `rules_20261112_122445_3653927.csv` for greedy and `rules_20261112_123529_3657961.csv` for sampled. The per-sample CSVs are needed to verify which seeds failed, whether the two constrained arms emitted the same rules, and the exact errors behind the five unconstrained rejections. The server's package and model revision details were unavailable for this note.

## What the pilot implies

- On these five tasks, the two constrained arms both produced executable rules in 15 of 15 sampled attempts, compared with 10 of 15 unconstrained. They also returned six more correct relations. `majority_owner` accounts for three of those six and uses a supplied gold `top_stake` rule.
- The two constrained arms tied on the reported task counts. The experiment leaves the incremental benefit of the KG-only predicate catalog open. All five gold rules use variables only, so the pilot offers limited evidence about entity grounding.
- Matching the gold relation demonstrates agreement on this graph. Additional graphs with deliberate edge cases would give stronger evidence that a rule captures the intended semantics. Five task definitions and three seeds provide a pilot baseline with limited coverage.
- The corrected run also changed the prompt and scorer. Compare the arms within the corrected run when reporting the effect of constraints. The earlier failed run is useful for diagnosing the harness.

## Next experiment

1. Compare the server CSVs by task and seed. Inspect the two-hop misses and the unconstrained `majority_owner` rules before tuning the prompt.
2. Expand the task set with plausible out-of-schema predicates, tasks that require KG constants, and additional graph fixtures with ties, repeated stakes, and varied ownership chains. Pair the prompts and seeds across arms.
3. Profile the tokenizer mask before scaling the CPU run. Use the same model, dtype, and prompt when comparing generation time.
4. Evaluate held-out QA questions with fact-only ReFactX, Nemo supplied with gold rules, and Nemo supplied with generated rules. Compare answer correctness by question kind, especially counts and multi-hop questions.

See the [README](README.md) for the setup, local checks, preflight, and server commands.
