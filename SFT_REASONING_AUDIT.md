# Audit of SFT_REASONING_EVAL.md

Scope: check the claims in `SFT_REASONING_EVAL.md` against the raw logs and the
training data, and note what the 30-sample numbers can and cannot support.
Datasets of interest: KQA Pro, 2Wiki, WebQSP, CWQ (GrailQA dropped from focus).

## 1. Table numbers vs raw logs

The four reasoning rows match their logs (`logs/reason30_*.out.0`):

| Model | KQA | GrailQA | WebQSP | CWQ |
|---|---:|---:|---:|---:|
| think_inside | 23.3 | 6.7 | 6.7 | 20.0 |
| think_outside | 20.0 | 20.0 | 10.0 | 26.7 |
| nothink | 20.0 | 3.3 | 13.3 | 20.0 |

The **baseline row does not come from one controlled run**:

| Column | Source in doc | Actual log | Adapter used |
|---|---|---|---|
| KQA 33.3 | `think_small_maskobject_kqa` | 33.3 | `kg_trace_maskobjects/ckpt-100`, thinking on |
| GrailQA 20.0 | `maskabl_maskobjects_grail` | 20.0 | `kg_trace_maskobjects/ckpt-100`, greedy |
| WebQSP 16.7 | `think_small_maskobject_webqsp` | 16.7 | `kg_trace_maskobjects/ckpt-100`, thinking on |
| CWQ 30.0 | `maskabl_maskobjects_cwq` | 30.0 | `kg_trace_maskobjects/ckpt-100`, greedy |

Problems this creates:

- The baseline mixes greedy and thinking-sampled runs (KQA/WebQSP thinking,
  GrailQA/CWQ greedy). Thinking-sampled GrailQA is 13.3 and CWQ is 20.0 for the
  same adapter, not 20.0 and 30.0.
- The baseline adapter is `kg_trace_maskobjects/ckpt-100`, but the reasoning
  adapters were trained from `sft_output_08b_combined/checkpoint-100`
  (`configs/sft_config_qwen35_08b_reason_*.json`, `adapter` field). So the
  comparison is **not** parent-matched. "Reasoning does not help" is therefore
  not a controlled conclusion.
- Decoding: reasoning runs use thinking + sampling (T=0.6, top_p 0.95, top_k 20).
  Thinking is only one factor; the sampled-vs-greedy difference is not isolated.

## 2. Sample size

All comparisons use 30 questions. With p≈0.2, the standard error is ≈7 points,
so differences of 3–10 points (the size of most gaps in the table) are noise.
Example: GrailQA "20.0 vs 3.3" is 6 vs 1 correct answers.

## 3. Training data defects (reasoning templates)

Source: `logs/reasoning_templates_v2/*.jsonl`, built by
`utils/build_reasoning_traces_v2.py`.

1. **Plan sentence bug (think_tools_outside):** plan text is built as
   `", then ".join(["retrieve ...", "then retrieve ..."])`, producing
   "then then retrieve". 905 / 1,869 traces contain it.
2. **Content-free plan:** the thinking block says only
   "The question asks for the answer. I'll first retrieve …". It contains no
   question-specific reasoning. The "reasoning" is a relation list, not reasoning.
3. **Answer not supported by the trace:** in all three files, about 18% of traces
   (≈280–330 each) give an answer string that does not appear in the facts or
   counts the same trace retrieves. Example (nothink, `base` question
   "duration of The Brothers Bloom"): the trace retrieves `113 minute`, then
   answers `"The Chronicles of Narnia: The Voyage of the Dawn Treader"`. The model
   is trained to emit answers it never retrieved.
4. **Teacher scratch notes leak into `full_trace`:** the `api_reasoning` field is
   the DeepSeek notes ("Need output JSON only. … Could include instance …"). The
   templated targets do not use it, but the `full_trace` fields do, and any
   reuse of those fields is contaminated.
5. **Tool-type mismatch:** `full_trace` has `Called relations(...)` /
   `Called search(...)` text from the old teacher; the template targets use
   `<fact>`/`<count>` only. Inconsistent with the tool syntax the model sees.

Defects 1 and 3 mean some of the training signal is wrong, not merely weak.

## 4. What this implies

- The reasoning experiment as run does not test "does reasoning help". It tests
  "does training on noisy, partly-wrong templated reasoning hurt", with a
  non-matched baseline and n=30.
- Reasoning SFT should be redone only after the data is fixed (filter
  answer-consistent traces, remove the "then then" bug, drop the content-free
  plan) and with a parent-matched control.

## 5. Next steps (see main report)

- 100-question evals per dataset with one adapter lineage (in progress:
  `configs/eval100_*.json`, logs in `logs/eval100/`).
- Fixed reasoning data + matched control, then evaluate on the same 100 IDs.
- Bigger lever: gold-program traces from KQA Pro (KoPL) and gold graphs from
  RoG (already used by `utils/build_path_sft.py`) give verified supervision with
  no API cost.
