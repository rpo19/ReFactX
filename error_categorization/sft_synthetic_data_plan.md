# Plan: completing the SFT training set with synthetic samples

This plan follows the diagnostics in `sft_training_data_report.md`. It states
what the natural teacher pool already provides, what it is missing, and how to
generate, verify and mix synthetic traces that fill the gaps — with priority on
the behaviours the paper needs: the **count tool**, the **exhaustion sentinel**,
**grounding**, and **explore → detect error → backtrack → correct** traces.

All counts below are from the Qwen3.8-27B teacher runs on the four training
splits (`logs/sft_training/`, 51 319 evaluated records).

---

## 0. What the diagnostics say

| Signal | Value | Consequence |
|---|---:|---|
| Strict positives (`correct & triples>0`) | **17 176** | usable natural traces |
| Verified positives (strict + semantic equivalence) | **20 973** | +3 798 recovered by fixing evaluation |
| Truncated traces | **24 896 (48.5%)** | half the raw pool is unusable |
| Positive traces using the count tool | **257** | count behaviour barely supervised |
| Count questions correct / correct using count | 293 / **170** | the teacher often answers counts without the tool |
| Positive traces containing the sentinel | **3 348** | present, but the 4B SFT model emits none |
| Wrong but gold facts *were retrieved* | **1 510** (1 427 outside 2Wiki) | composition/selection errors we can re-generate |
| 2Wiki retrieval failures | **2 451** | evidence recall only **0.322** |
| Verification artifacts | **3 798** | mostly 2Wiki dates (3 517) |

Two conclusions drive the plan:

1. **Coverage is the bottleneck, not raw volume.** 20 973 verified positives
   exist, but they are dominated by 2Wiki (79%) and by fact-only traces; the
   behaviours the paper introduces — count, sentinel, self-correction — are far
   below the targets in §2 (the count tool, in particular, appears in only 257
   positive traces).
2. **Half the pool is lost to truncation** (72% Mintaka, 77% CWQ, 59% WebQSP).
   Generation settings are part of the data problem, not just a runtime detail.

### Mintaka per-type positives (strict), the weakest types

| Type | Positives | Type | Positives |
|---|---:|---|---:|
| comparative | 604 | multihop | 192 |
| generic | 525 | ordinal | 123 |
| yes/no | 456 | difference | 63 |
| count | 240 | superlative | 39 |
| intersection | 204 | | |

The four weakest types (`difference`, `superlative`, `ordinal`, `multihop`) are
exactly the compositional types the paper flags as the teacher's weak spot, and
they are also the most truncated. This is where synthetic data has the highest
marginal value.

---

## 1. Goals and acceptance criteria

**Goal.** Produce a balanced SFT mixture in which every behaviour the paper
claims to teach is (a) present, (b) *necessary* for the trace to succeed, and
(c) verifiably correct.

**Acceptance criteria for the mixture**

- every Mintaka question type has **≥ 500** verified positives;
- **≥ 1 500** verified count-tool traces, at least 1 000 on Mintaka count questions;
- **≥ 2 000** *sentinel-critical* traces (the answer is a complete set and the
  sentinel is what establishes completeness);
- **≥ 1 000** self-correction/backtracking traces and **≥ 1 000** verify-then-answer traces;
- **≥ 700** repaired composition traces on the 1 427 questions where the gold
  facts were retrieved but the answer was wrong;
- **≤ 50%** synthetic share, **0** truncated traces, **0** duplicates, **0**
  test/dev contamination;
- grounding rate (answer supported by retrieved facts) **≥ 95%** on a judged sample.

---

## 2. Behaviour targets (current → target)

| Behaviour | Current natural positives | Target | Primary source |
|---|---:|---:|---|
| Fact-grounded single/multi-hop | ~13 800 | keep | natural pool |
| Count tool | 257 | **1 500** | Mintaka count + synthetic counts |
| Sentinel / exhaustive enumeration | 3 348 (not *critical*) | **2 000 critical** | WebQSP/CWQ list answers + synthetic enumeration |
| Backtracking / self-correction | 0 | **1 000** | prompted + injected-error traces |
| Verify-then-answer | 0 | **1 000** | prompted verification traces |
| Composition repair | 0 (1 427 failures) | **700** | re-generation on retrieved-facts failures |
| Concise reasoning | — | — | control truncation (see §3.6) |

These are **starting targets** to be revised after the pilot acceptance rates in §6.

---

## 3. Generation strategy

### 3.1 Seeds and generator

- **Seeds**: the four training splits already evaluated, plus their KG indices
  (Wikidata for Mintaka/2Wiki, Freebase for WebQSP/CWQ).
- **Generator**: a strong model that is *not* the student. The paper already
  plans DeepSeek for the underrepresented cases; the 27B teacher with a
  behaviour-specific prompt is the fallback.
- **Gold conditioning**: generate *without* revealing the gold answer and keep
  traces by rejection sampling. Gold-conditioned generation is allowed only for
  hard count/sentinel cases and must be tagged, because it teaches the model that
  the answer is given. Never train on an answer-conditioned trace unless it also
  passes the grounding check.

### 3.2 Count behaviour

1. Use the 1 400 Mintaka count questions (only 170 correct traces use the tool),
   plus synthetic "how many" questions over high-frequency KG relations.
2. Force a `<count>` call in the prompt; the harness returns the verified leaf
   count.
3. Keep only if: the trace uses `<count>`, the returned count equals the gold
   count, and the final answer equals the count.
4. Add variants that need **filtered counts** (e.g. "how many of X were born
   before Y") so the model learns to *combine* count with facts, not just count a
   bare relation.
5. Verification is exact and cheap: compare against the index leaf count, not a
   judge.

### 3.3 Sentinel behaviour (must be *critical*)

The natural pool contains the sentinel, but the student learned nothing from it,
so presence is not enough: the trace must *depend* on knowing exhaustion.

1. Seed with enumeration questions (WebQSP/CWQ list answers; Mintaka list-like
   and superlative/ordinal questions that enumerate candidates).
2. Require the trace to enumerate until `<no further records>`, then answer the
   complete set.
3. Verify **completeness** against the KG: the predicted set must equal the set
   reachable from the subject–relation prefix (the harness already maintains
   these leaf counts).
4. Add negative-control traces where the model stops early *without* the
   sentinel and is wrong, for DPO pairing (§4).

### 3.4 Backtracking / self-correction (the "special behaviour")

This is the behaviour the paper will want to showcase: the model explores a
plausible wrong path, detects the mistake, and recovers.

1. **Prompted generation.** Instruct the generator to: propose a candidate
   entity/relation, verify it against retrieved facts, notice a contradiction
   (wrong entity type, wrong date ordering, a fact that rules the candidate out),
   state the error explicitly, and explore an alternative until consistent.
2. **Programmatic injection.** Start from a correct trace; insert a *real* KG
   fact that is plausible but leads to a wrong candidate, then a correction span
   ("that is the wrong *X* because …; try *Y*"), then resume the correct path.
   Using real facts keeps the trace realistic and the wrong branch explorable.
3. **Validation** (all required):
   - final answer is correct (strict + verified);
   - the trace contains a recognisable correction (_wrong / instead / actually /
     that's not …_) verified by regex **and** a judge;
   - the wrong branch is grounded in real retrieved facts;
   - the wrong branch is *closed* (sentinel or an explicit rejection), so the
     model learns to terminate bad paths rather than wander.
4. Mix sources: at least 300 from WebQSP/CWQ (dense graphs, many false leads),
   400 from Mintaka (compositional types), 300 from 2Wiki.
5. Keep the wrong branch as a **rejected** sample for DPO
   (`utils/select_dpo_data.py`), and the corrected continuation as the chosen
   one. This gives the same data a second use.

### 3.5 Verify-then-answer behaviour

1. Prompt: answer, then re-query to confirm the supporting fact, then emit the
   final answer.
2. Keep only when the confirmation cites a *retrieved* triple that supports the
   answer (grounding check), so the model cannot "verify" with parametric memory.
3. This directly targets the "correct but ungrounded" surface: currently only
   32.4% of Mintaka and 23.3% of 2Wiki positives have the gold answer literally
   in the retrieved triples (an upper bound on ungroundedness, since derived
   answers are expected to be absent).

### 3.6 Composition repair (WebQSP/CWQ)

1. Re-run the 1 427 questions where the gold facts *were* retrieved but the
   answer was wrong (WebQSP 459, CWQ 681, Mintaka 287).
2. Generate with the retrieved facts fixed, so the task is pure composition.
3. Keep verified-correct traces; this turns existing failures into supervision
   without new questions.

### 3.7 Conciseness and truncation

1. Measure, on a 500-question pilot, the fraction that completes at 3 072 and
   4 096 tokens; this sizes the recoverable pool before any synthetic data.
2. Add a conciseness instruction and a length penalty in generation; prefer
   traces under ~75% of the student budget.
3. For traces truncated mid-lookup, do **not** splice blindly: regenerate the
   continuation from the recorded prefix and validate coherence, or discard.

---

## 4. Verification and filtering pipeline

Run every candidate through this order; reject at the first failure.

1. **Structure**: parseable `<answer>`, no truncation, no incomplete `<fact>`
   call, exactly one answer block.
2. **Correctness**: strict evaluator **and** `verify_answer` (dataset-aware,
   fixed ambiguous two-digit years, quantities, aliases, entity folding).
3. **Tool-use contract**: ≥1 `<fact>`; `<count>` for count questions; sentinel
   for enumeration questions.
4. **Grounding**: every asserted support triple must be among the retrieved
   triples; for derived answers, a judge/NLI check that the retrieved facts
   license the answer. (Rule-based containment alone is insufficient for counts,
   ages and comparisons.)
5. **Behaviour tag check**: the claimed behaviour (count/sentinel/backtracking/
   verification) is present in the trace; for backtracking, require a correction
   span and a closed wrong branch.
6. **Deduplication**: exact + MinHash near-duplicate against the natural pool and
   across synthetic variants.
7. **Decontamination**: no overlap with the test/dev splits (n-gram + question
   embedding).
8. **Balance**: stratify by dataset, question type, answer shape and behaviour.

Reuse `utils/answer_normalization.py` (now with `verify_answer`) and
`utils/log_analysis.py`; extend `utils/filter_positive_training_samples.py` with
the behaviour tags and the schema below.

---

## 5. Example schema (one JSONL line)

```json
{
  "question": "...",
  "dataset": "mintaka",
  "kg": "wikidata",
  "gold_answer": ["..."],
  "gold_evidence": [["s", "r", "o"]],
  "tool_trace": [
    {"tool": "fact", "triple": ["s", "r", "o"]},
    {"tool": "count", "prefix": ["s", "r"], "value": 7},
    {"tool": "sentinel", "prefix": ["s", "r"]}
  ],
  "final_answer": ["..."],
  "behaviour_tags": ["count", "backtracking", "grounded"],
  "verification": {"strict": true, "normalized": true, "grounded": true, "completeness": true},
  "generator": {"model": "...", "prompt": "...", "gold_conditioned": false},
  "seed_id": "...",
  "split": "train",
  "provenance": "synthetic|natural",
  "dpo_pair": {"rejected_trace": "..."}
}
```

---

## 6. Recommended mixture

| Source | Traces | Notes |
|---|---:|---|
| Natural verified positives | 20 973 | keep, but re-verified with the fixed evaluator |
| + count synthetic | 1 500 | ≥1 000 from Mintaka count |
| + sentinel-critical synthetic | 2 000 | WebQSP/CWQ enumeration |
| + backtracking synthetic | 1 000 | 300/400/300 WebQSP/Mintaka/2Wiki |
| + verify-then-answer synthetic | 1 000 | mixed |
| + composition repair | 700 | from the 1 427 retrieved-facts failures outside 2Wiki |
| **Total** | **~27 200** | ≈23% synthetic |

If the 2Wiki dominance is undesirable, cap 2Wiki at ~50% of the natural pool
and rebalance toward Mintaka/WebQSP/CWQ, which carry the harder types.

---

## 7. Work plan

| Phase | Work | Deliverable | Acceptance |
|---|---|---|---|
| 0 | Fix the evaluator: ambiguous two-digit years; re-score the four logs | corrected scores | 2Wiki +3 517 recovered |
| 1 | Generation + verification harness; reproduce natural positives | `gen_synthetic.py` + filters | 0 false accepts on a 200-sample audit |
| 2 | Count and sentinel synthesis | tagged JSONL | targets in §2 met |
| 3 | Backtracking and verify-then-answer | tagged JSONL + DPO pairs | judge agreement ≥90% on 200 samples |
| 4 | Composition repair on retrieved-facts failures | tagged JSONL | ≥50% of 1 427 recovered |
| 5 | Mix, dedup, decontaminate, train, ablate | SFT runs | per-type + behaviour gains |

Ablations: natural-only vs +count vs +sentinel vs +backtracking vs +all; report
per-type accuracy, count accuracy, sentinel precision/recall, tool-usage
precision, truncation and abstention.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Synthetic artefacts leak into reasoning style | cap synthetic share; audit a judged sample; keep natural anchors |
| Gold-conditioned traces teach "answer is given" | prefer rejection sampling; tag and down-weight conditioned traces; require grounding |
| Backtracking traces teach rambling | require a *closed* wrong branch and a concise correction; enforce a length cap |
| Judge errors in grounding | calibrate on human labels; restrict admissible evidence to retrieved triples |
| Contamination | n-gram + embedding check against test/dev before mixing |
| Cost of a stronger generator | batch, cache, and generate only for the weak types/behaviours |
| 2Wiki retrieval failures are a harness problem, not a data problem | track separately: fix retrieval/index or increase exploration; synthetic data only helps if the facts are reachable |

---

## 9. Open decisions

1. Generator: DeepSeek (paper todo) vs Qwen3.8-27B with a behaviour prompt.
2. Gold conditioning: none (safest) vs allowed-and-tagged for count/sentinel.
3. Synthetic share: the proposed ~24% vs a stricter 15%.
4. Whether to raise `max_new_tokens` for the SFT traces (recover truncation)
   or teach conciseness (cheaper inference).
5. How aggressively to rebalance away from 2Wiki.

---

## 10. Implementation

The pipeline is implemented and documented in `error_categorization/sft_pipeline.md`:

| Step | Tool |
|---|---|
| Behaviour labels (Jev) | `utils/classify_behaviors.py` (`utils/behaviors.py`, `utils/jev_client.py`) |
| Synthetic count traces (DeepSeek) | `utils/generate_count_traces.py` |
| Synthetic backtracking traces (inject / prompted) | `utils/generate_backtracking_traces.py` |
| Behaviour-aware SFT mixture | `utils/build_sft_dataset.py` |

```bash
# 1. label the four evaluation logs (pilot with --limit 500 --mock first)
python utils/classify_behaviors.py logs/sft_training/*.out* --workers 8 \
    --output-dir logs/behaviors --report logs/behaviors/report.json

# 2. generate 1000 verified count traces
python utils/generate_count_traces.py \
    --input logs/sft_training/mintaka_train_eval_qwen38_27b.out.3.gz \
    --select teacher-wrong-or-no-count --n 1000 \
    --output logs/synthetic/count_traces.jsonl

# 3. build the mixture (all non-2Wiki positives, 2Wiki capped at 2000)
python utils/build_sft_dataset.py \
    --natural logs/behaviors/*.behaviors \
    --synthetic logs/synthetic/count_traces.jsonl \
    --output logs/sft_training_mix.jsonl \
    --two-wiki-cap 2000 --best-ratio 0.75 --min-acceptable 0.15
```

Cost (official prices): the full 51 319-record labelling is ~$12 off-peak with
DeepSeek Flash or ~$3.3 with Jev; the 7 180-record mixture is ~$1.9 / ~$0.53.
The 1 000 count traces cost ~$1.5–2.9 to generate. Start with a 500-record
pilot (<$0.15) to validate the labels and the achieved 75/25 ratio.
