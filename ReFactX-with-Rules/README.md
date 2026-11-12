# ReFactX with Rules

This extension uses constrained decoding to generate executable Nemo Datalog rules for a synthetic company-ownership knowledge graph. It builds on [ReFactX](https://arxiv.org/abs/2508.16983), which constrains fact retrieval to entries in a prefix-tree index. Here the model writes a rule, Nemo derives the requested relation, and the benchmark compares the resulting tuples against an independent reference.

The [November 2026 milestone](MILESTONE_NEMO_RULE_GENERATION.md) records the first successful Qwen3.5-4B server run, the observed accuracy and latency, and the questions for the next experiment.

## What works

- A seeded graph of 100,000 entities and 273,594 base triples. The stored predicates are `OWN(owner, company, stake_percent)`, `ROLE(person, company, role)`, and `REACHABLE(entity, other_entity)`.
- A Python reference reasoner and a Nemo backend with the same lookup API. Both derive `QUALIFIED_HOLDINGS`, `INFLUENCE`, `CONTROL`, `ULTIMATE_CONTROL`, `CONNECTED`, and `INDIRECT_ROLE`. The lookup API exposes symmetric `REACHABLE` and inverse property names such as `is_own`.
- A synthetic QA suite of 1,110 questions across six kinds. The QA checks exercise the reasoner API. Stage 5 separately evaluates generated relations by executing rules and comparing tuples.
- Runtime registration of binary Nemo rules through `Reasoner.add_rule()`. Successful additions stay available for later queries and rule dependencies. A failed addition preserves the prior program and indexes.
- A headless benchmark and notebook for five tasks, with tokenizer preflight, per-sample diagnostics, and three decoding arms.

Nemo materializes the graph and builds indexes for API lookups. `CONTROL` uses a data-sized sequence of nonrecursive aggregates for the ownership fixpoint. The [Vadalog design notes](note.md) and [Nemo integration notes](working_plan/nemo_design_and_integration.md) describe the earlier design work.

## Setup

Run the commands in this README from `ReFactX-with-Rules/` unless a command explicitly uses a different directory. Use Python 3.12 and a platform with a [Nemo Python wheel](https://pypi.org/project/nemo-python/0.10.1/).

```bash
cd ReFactX-with-Rules
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

mkdir -p data
python kg_gen.py
```

The generator writes `data/entities.csv` and `data/triples.csv`, the two inputs required by Stage 5. The command should report **100,000 entities** and **273,594 triples**. The Faker version is pinned to reproduce the pilot dataset. For other models, tokenizers, and test dependencies, install the benchmark requirements in the same environment.

```bash
python -m pip install -r requirements-benchmark.txt
```

For a CPU-only Linux environment, install a compatible [CPU build of PyTorch](https://pytorch.org/get-started/locally/) before installing the benchmark requirements. Stage 5 uses the Hugging Face cache for model and tokenizer files. A full run requires enough RAM or accelerator memory for the selected model and dtype. The repository contains the generators and benchmark code. Generated datasets, exported gold CSVs, and per-run results stay local.

## Verify the reasoner

With the graph generated, run the Stage 1 parity check and the Stage 3 harness. Neither loads model weights.

```bash
python stage1/smoke_test.py
python stage3/rule_harness.py
```

Stage 1 should end with `ALL RELATIONS MATCH`. Stage 3 should report all five gold rules as `correct`. It also runs deliberately wrong rules and probes where Nemo rejects malformed or unstratified programs.

To regenerate the QA fixtures and exercise both backends, run:

```bash
python reasoner.py
python qa_gen.py
python add_groundtruth.py
python test.py
python stage2/run_with_nemo.py test.py
```

`reasoner.py` writes the base-fact index and a larger dump containing derived facts. The Stage 2 runner injects the Nemo backend before loading the same QA checks. Use `python api.py` for the Python backend walkthrough or `python stage2/run_with_nemo.py api.py` for the Nemo-backed walkthrough.

The focused regression tests cover constraint acceptance, token masking, malformed model output, engine failures, and rollback after failed registration. The tokenizer test downloads GPT-2 tokenizer files. Set `REFACTX_TEST_TOKENIZER` to a local tokenizer directory for offline testing.

```bash
python -m pytest -q stage4/test_constraint.py stage5/test_benchmark.py
```

## Run the rule-generation benchmark

### Tokenizer-only preflight

```bash
python stage5/run_benchmark.py --preflight --model Qwen/Qwen3.5-4B
```

Preflight scores the five gold rules in Nemo and walks their token sequences through both constrained arms using the real chat template. It requires the tokenizer files. The expected final line is `preflight passed`.

### Greedy and sampled runs

```bash
# One greedy generation per task and arm
python -u stage5/run_benchmark.py --model Qwen/Qwen3.5-4B \
  --device cpu --dtype float32 --n-samples 1 --no-sample --check-masks

# Three sampled generations per task and arm
python -u stage5/run_benchmark.py --model Qwen/Qwen3.5-4B \
  --device cpu --dtype float32 --n-samples 3
```

Use `--quick-test --device cpu --dtype float32` for the smaller Qwen2.5-0.5B smoke run. Use `--tasks owner_of_owner,majority_owner` or `--ablations full` to narrow a run. Use `--seed` to set the first seed and `--out-dir` to set the destination for the CSV. The [Stage 5 notebook](stage5/rule_generation_benchmark.ipynb) uses the same evaluator. Open it with the benchmark environment as the Jupyter kernel.

### Tasks and decoding arms

| Task | Requested relation |
|---|---|
| `owner_of_owner` | Two-hop ownership join |
| `owner_count` | Distinct owners per company |
| `top_stake` | Maximum stake per company |
| `total_owned` | Sum of each owner's stakes, preserving equal stakes in different companies |
| `majority_owner` | All holders of the maximum stake, including ties, using a supplied gold `top_stake` rule |

All three arms receive the same prompt and output contract. `unconstrained` samples freely. `grammar_distractors` applies the rule grammar, variable-binding checks, and entity grounding with extra plausible predicate names. `full` uses the same constraint and allows only predicates in the task's KG catalog. The legacy CLI name `syntax` maps to `grammar_distractors`.

The evaluator checks one bare rule with the expected head and arity, validates its body predicates, then runs Nemo with the candidate head as the sole requested output. It compares the result set against gold tuples computed from `triples.csv` using pandas. Numeric values are rounded to six decimal places for comparison.

| Outcome | Meaning |
|---|---|
| `generation_error` | A constrained generation hit a dead prefix, stopped before completion, or missed a mask step |
| `invalid` | The response failed extraction or validation |
| `engine_error` | Nemo panicked or failed during execution |
| `valid_wrong` | The rule executed and returned missing or extra tuples |
| `correct` | The executed result matched the reference tuple set |

The runner flushes each sample to `stage5/results/rules_*.csv`. Inspect `raw_text`, `error`, `failure_stage`, `missing`, and `extra` to diagnose failures. `gen_s` includes mask time. `mask_s` and `nemo_s` report mask and engine timings, and `wall_s` covers the whole sample. Pair samples by `task` and `seed` when comparing arms.

## Run on a remote server

Generate the two input CSVs locally before packaging. The archive includes the Stage 5 sources, dependency manifests, and the two CSVs. Set the SSH destination explicitly. The port defaults to 22.

```bash
export REFACTX_SERVER="your-user@your-host"
export REFACTX_PORT=22
bash stage5/transfer.sh --upload
ssh -p "$REFACTX_PORT" "$REFACTX_SERVER"
```

On the server, set up a Python environment with the benchmark dependencies above. The runner defaults to `~/refactx_env/bin/python`, Qwen3.5-4B, CPU, and float32. Set `REFACTX_PYTHON`, `REFACTX_MODEL`, `REFACTX_DEVICE`, or `REFACTX_DTYPE` to override those values.

```bash
mkdir -p ~/refactx
tar -xzf ~/rules_srv.tgz -C ~/refactx
bash ~/refactx/stage5/rerun_server.sh smoke
```

`smoke` runs the tokenizer preflight and a greedy pass. It exits with an error if a constrained generation fails to complete. After it passes, run the 45-sample pilot configuration.

```bash
bash ~/refactx/stage5/rerun_server.sh benchmark
```

The CSVs are written to `~/refactx/stage5/results/`. Run `transfer.sh` without `--upload` to package locally and print the transfer commands.

## Repository map

| Path | Purpose |
|---|---|
| [`kg_gen.py`](kg_gen.py), [`qa_gen.py`](qa_gen.py), [`add_groundtruth.py`](add_groundtruth.py) | Build the synthetic KG, question set, and reference answers |
| [`reasoner.py`](reasoner.py), [`nemo_reasoner.py`](nemo_reasoner.py), [`api.py`](api.py) | Python reference, Nemo adapter, and lookup API |
| [`stage1/`](stage1), [`stage2/`](stage2) | Backend parity and API checks |
| [`stage3/`](stage3) | Task definitions, gold computation, and Nemo validation and execution |
| [`stage4/`](stage4) | Character grammar, entity trie, tokenizer mask, and tests |
| [`stage5/`](stage5) | CLI and notebook benchmark, preflight, and server scripts |
| [`working_plan/`](working_plan) | Historical design and integration notes |

The earlier [`<kg>` lookup notebook](try_refactx_knowledge_graph.ipynb) uses a separate [ReFactX `feat/kg` checkout](https://github.com/rpo19/ReFactX/tree/feat/kg) with `KnowledgeGraphGeneration` and its prompt files. To run it, install that checkout's model dependencies in a separate environment, set `REFACTX_KG_CHECKOUT` to the checkout's absolute path, generate the QA fixtures, and open the notebook from this directory. The Nemo Stage 1–5 commands use the files in this directory.
