"""Build stage5/rule_generation_benchmark.ipynb (run once, edit the notebook)."""
import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Stage 5 — Rule generation benchmark: constrained decoding vs baselines

Generate Datalog rules with a real LLM over the company-ownership KG and
evaluate them against the Nemo engine (stage 3 harness) + gold results.

## Ablations

| name | constraint | failure classes left |
|---|---|---|
| `unconstrained` | none | extraction, validation, engine, or result errors |
| `grammar_distractors` | rule grammar, variable binding, entity grounding, and a predicate catalog with distractors | unknown predicates, incomplete generation, engine or result errors |
| `full` | same grammar and grounding with the KG predicate catalog | incomplete generation, engine or result errors |

## Outcome of each generated rule

- `generation_error` — the constraint cannot continue or the generated rule is incomplete
- `invalid` — the output fails extraction or validation
- `engine_error` — Nemo panics or fails during execution
- `valid_wrong` — Nemo accepts and executes the rule, but results ≠ gold
- `correct` — results == gold (numeric tolerance 1e-6)

## Notes

- Nemo evaluates the candidate head and its dependencies. For a tokenizer-only
  check before loading weights, run `python stage5/run_benchmark.py --preflight`.
- `QUICK_TEST=True` uses Qwen2.5-0.5B (~1 GB) with one greedy sample per arm.
- On Apple Silicon `bfloat16` + `mps` works; if the model does not fit,
  switch `DTYPE` to `float16`/`float32` (CPU) or a smaller model.
""")

code("""import os
import sys
import time

import pandas as pd
import torch

# anchor to the repo root: cwd if launched there, one level up if launched
# from stage5/ (where this notebook lives)
ROOT = os.getcwd()
if not os.path.isdir(os.path.join(ROOT, "data")):
    ROOT = os.path.dirname(ROOT)
for p in (ROOT, os.path.join(ROOT, "stage3"), os.path.join(ROOT, "stage4"),
          os.path.join(ROOT, "stage5")):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_rules import ABLATIONS, Eval, load_model, TASKS

print("repo root:", ROOT)
print("device:", "mps" if torch.backends.mps.is_available()
      else "cuda" if torch.cuda.is_available() else "cpu")""")

md("## Model & run configuration  (edit these knobs)")

code("""# ---- model ----
MODEL_ID = "Qwen/Qwen3.5-4B"
DTYPE = "bfloat16"
DEVICE = None                       # auto: mps > cuda > cpu

# ---- benchmark ----
QUICK_TEST = True                   # True -> tiny model, 1 task, 1 sample
TASK_IDS = None                     # None -> all 5 tasks
N_SAMPLES = 3                       # samples per task x ablation
SEED = 42
MAX_NEW_TOKENS = 160

# ---- sampling ----
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

if QUICK_TEST:
    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
    TASK_IDS = ["owner_of_owner"]
    N_SAMPLES = 1
    DO_SAMPLE = False
    print("QUICK TEST:", MODEL_ID, TASK_IDS, ABLATIONS, N_SAMPLES)""")

code("""model, tok, device = load_model(MODEL_ID, DTYPE, DEVICE)
print("loaded", MODEL_ID, "on", device)
print("vocab:", tok.vocab_size)""")

md("## Run the benchmark")

code("""ev = Eval()
tasks = [t for t in TASKS if TASK_IDS is None or t["id"] in TASK_IDS]
gen_cfg = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=DO_SAMPLE,
               temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)

t0 = time.time()
results = ev.run(model, tok, tasks, ABLATIONS, N_SAMPLES, SEED, gen_cfg, verbose=True)
print("\\ntotal %.1f min" % ((time.time() - t0) / 60))
""")

md("## Results")

code("""print("=== per ablation ===")
display(ev.summary(results))

print("=== per task x ablation (%% correct) ===")
piv = results.pivot_table(index="task", columns="ablation", values="outcome",
                          aggfunc=lambda s: 100 * (s == "correct").mean()
                          ).round(1)
display(piv)

print("=== example generation, validation and engine failures ===")
fails = results[results.outcome.isin(["generation_error", "invalid", "engine_error"])]
for _, r in fails.head(8).iterrows():
    print("[%s|%s] %s" % (r.task, r.ablation, r.error))
""")

code("""os.makedirs(os.path.join(ROOT, "stage5", "results"), exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")
out = os.path.join(ROOT, "stage5", "results", f"rules_{stamp}.csv")
results.to_csv(out, index=False)
print("saved", out)""")

md("""`mask_steps` shows where the decoder applied the mask. The distractor arm
tests the effect of expanding the predicate catalog under the same constraint.
Correctness means the executed relation matches the gold tuples on this KG.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "rule_generation_benchmark.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out)
