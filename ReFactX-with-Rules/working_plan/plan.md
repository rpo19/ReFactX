# Working Plan — ReFactX-with-Rules → Constrained Rule Generation

> Historical design notes. See the [current milestone](../MILESTONE_NEMO_RULE_GENERATION.md) and [testing guide](../README.md).

> Working analysis: project context, limitations, and the plan to adapt
> constrained generation of Datalog rules to the open-source Nemo engine
> (testing ground for Vadalog, license-gated).
>
> Created from the analysis of the repo at
> `~/Documents/ReFactX/ReFactX-with-Rules` (Banca d'Italia collaboration).

---

## 1. Project context

### 1.1 ReFactX (ISWC 2025, Pozzi et al.)

- LLMs access external knowledge **without retrievers**: ~800M Wikidata triples are
  verbalized into textual facts, tokenized, and indexed in a **prefix tree**.
- During decoding, a HuggingFace `LogitsProcessor` masks every token that is not a
  legal continuation of an existing fact (triggered by a `Fact:` / `<kg>` marker).
- Guarantee: **the model physically cannot generate a fact that is not in the index**.

### 1.2 ReFactX-with-Rules (this repo)

Extends the same trick from static fact lookups to a **rule-backed KG with a reasoner**.

- **KG**: synthetic company-ownership — 100k entities (70% companies, 30% persons),
  273,594 base triples in 3 stored relations: `OWN(s,o,%)`, `ROLE(s,o,role)`,
  `REACHABLE(s,o)`.
- **Derived relations** (rules, materialized by the mock / derived on demand by Vadalog):
  `QUALIFIED_HOLDINGS`, `INFLUENCE`, `CONTROL`, `ULTIMATE_CONTROL`, `CONNECTED`,
  `INDIRECT_ROLE`. Example: `CONTROL(X,Y)` when summed OWN stakes along chains
  from X into Y exceed 50%; `ULTIMATE_CONTROL(X,Y)` when nobody controls X.
- **Three-level interaction protocol** (LLM ↔ reasoner):
  1. *Entity index* (prefix tree): only existing entity names can be generated.
  2. *`properties(entity)`*: relations that have answers for that entity
     (`own` binds entity as subject, `is_own` as object — **no inverse triples
     stored**, reverse direction is a query).
  3. *`get` / `count` / `ask`*: execute the rule and return results.
- The LLM **never reproduces the database** — it interrogates the reasoner and gets
  derived knowledge in real time; `add_rule` allows extending the KG at runtime.
- **Benchmark**: 6 question kinds (`ask`, `count`, `list`, `more`, `and_not`, `top`)
  × 9 relations = 1,110 questions with reasoner-computed ground truth;
  `utils/eval_kg.py` runs the whole benchmark under constrained `<kg>` generation
  (Qwen3.5-4B).

### 1.3 Current status

- [x] Constrained generation of **facts** from the synthetic KG works (mocked
      Vadalog API; see `try_refactx_knowledge_graph.ipynb`).
- [ ] **Rule generation** (the model writes Vadalog rules, e.g. to add a missing
      relation or compute aggregations like `max`, `count`) — vibecoded but
      untestable: Vadalog requires a license.

---

## 2. Limitations → basis for extending the work

### L1. The constraint covers *names*, not *programs*

The current semantic constraint only restricts entity/relation **names**:

- **Arity and direction**: nothing forces the LLM to pick `control` vs `is_control`
  to match the query's binding pattern (it's a string convention).
- **Composition**: each question kind maps to a fixed composition in `api.py`
  (`CALL` dict); nothing constrains the LLM to any valid composition — it can
  still produce nonsense like `ask(x,p,y) > 3`.
- **Types/values**: rules like `OWN(X,Y,P), P >= 10` rely on numeric domains the
  constraint knows nothing about.

→ *Extension*: a **grammar-level constrainer** (typed grammar of the rule/query
language) layered on the name-level index. This is the syntax/semantics split.

### L2. Closed-world, static index

- Prefix tree built once from **stored triples only** (`reasoner.facts()` yields
  only `BASE` relations); derived knowledge is invisible to the tree.
- Composable facts ("X controls Y and Z controls Y") have no constrained form.
- `add_rule` / KG edits don't update the index at decode time.

→ *Extension*: incremental/dynamic indexes, or lazy on-demand candidate
generation (query the reasoner per position instead of pre-indexing).

### L3. The three-level protocol is a fixed straitjacket

- No multi-entity / quantified queries ("who controls more than 3 companies…").
- Round-trip latency: each level is an LLM step + reasoner call; no planner,
  no caching, no retry on error.
- The model can still issue *ill-formed calls* (`get(e, "nonsense")` raises
  `KeyError`): names are constrained, call well-formedness is not.

→ *Extension*: agent loop with a **constrained action grammar** (typed plan DSL)
+ execution feedback.

### L4. Rule generation is entirely uncharted

- **Safety/termination**: arbitrary LLM-written rules can be non-terminating or
  intractable. Vadalog has wardedness; Nemo has stratification. No checker exists
  in the repo.
- **Semantic legality**: existing predicates, correct arities, consistent
  variables, aggregate semantics — needs a typed grammar + validator.
- **No evaluation**: no pipeline for generated rules, no metrics for
  correct/safe/useful.

→ *Extension*: rule synthesis with constrained decoding + validator +
test-execution; new benchmark.

### L5. The mock hides the hard part

`reasoner.py` materializes everything eagerly in Python. Real Vadalog differences:

- On-demand derivation + **streaming** results (benchmark caps `list` answers at 10).
- Aggregations over recursive rules (CONTROL fixpoint) — engine semantics differ.
- Vadalog is closed-source: cannot test, benchmark, or introspect it.

→ *Extension*: define the abstract reasoner API formally; implement adapters for
Nemo and Vadalog; study which engine properties matter (latency, streaming,
aggregation semantics, introspection cost of `properties(e)`).

### L6. Evaluation is narrow

- Template questions only (6 syntactic kinds); no compositional/free-form questions.
- Synthetic KG; no noise, no missing data, no schema heterogeneity.
- End-answer metrics only: no failure attribution (entity? property? call?),
  no RAG baseline, no cost/latency reporting.

→ *Extension*: confusion matrix over the three levels; harder compositional
benchmark; ablations per constraint level.

### L7. Engineering fragility

- Tokenizer-specific trees (full re-index per model).
- Beam-search patch uses global mutable state (not thread-safe — `generate.py`
  warns about this).
- Per-position candidate collection is a new hot path in the decode loop.
- `Fact:` trigger needs prompt engineering; nothing guarantees the model enters it.

→ *Extension*: model-independent indexes or lazy per-position candidates;
thread-safe multi-beam; trigger-free schemes.

### L8. Generalizing to Python is a different beast

- KG world is **closed**; Python is **open**: names created at runtime (variables,
  imports, lambdas), libraries change, huge candidate sets per position.
- Executing LLM-generated Python = code-execution security problem.
- Honest scope: *positional name constraints* (after `import`, after `df.`) in
  typed fragments, not full-program generation.

---

## 3. Decision: adapt to Nemo as the open testing ground

### 3.1 Why Nemo (evidence gathered)

- **Feature parity with Vadalog on what we need**: Datalog with existential rules
  (TGDs), stratified negation, aggregates (`#count`, `#min`, `#max`, `#sum`),
  datatypes. Successor of VLog — same Datalog± family as Vadalog.
  - `distinctAges(#count(?AGE)) :- person(_, _, ?AGE, _).`
  - `calculated(?x, !v) :- data(?y, ?x) .`  (`!v` = existential head variable)
- **Testable today, on this Mac, no license**:
  - CLI `nmo` v0.10.1, prebuilt `aarch64-apple-darwin` binary.
  - Python bindings `nemo-python 0.10.1` on PyPI:
    ```python
    from nmo_python import load_string, NemoEngine
    engine = NemoEngine(load_string(rules))
    engine.reason()
    list(engine.result("calculated"))
    ```
- **Free validator for rule generation**: `load_string` rejects bad syntax,
  unknown predicates, wrong arities, unbound head variables, non-stratified
  negation. Parse/exec errors = feedback signal + evaluation metric.

### 3.2 Architecture transfer

`reasoner.py` already hides the engine behind an abstract interface
(`properties(e)`, `get/count/ask`, `facts()`). The adaptation is a **backend swap**:

- Dump `data/triples.csv` as Nemo facts, load program, `reason()` once.
- `properties(e)` = which relations have tuples touching `e`.
- `get/count/ask` = lookups over `engine.result(...)`.
- Keep `test.py` unchanged as cross-check: if Nemo's derived relations disagree
  with the Python mock, it's an **aggregate-semantics finding** (Nemo counts
  distinct tuples; the mock counts multisets) — a paper paragraph, not a bug.

For constrained rule generation, the ReFactX mechanism generalizes:

```
<rule>  grandparent(?X,?Z) :- parent(?X,?Y), parent(?Y,?Z) .
        └── grammar automaton ──┘ └─── semantic candidates ───┘
```

Constraint at each decode step = **syntax-next-tokens ∩ semantic candidates**
(predicates/arities from KG schema, entity names from prefix tree).
Ablations: unconstrained vs syntax-only vs syntax+semantics.

### 3.3 Caveats

1. **Syntax is engine-specific** (Nemo: `?var`, `!existential`, `#count`;
   Vadalog: uppercase vars, Prolog-style, different aggregate syntax and
   annotations). → Keep the constraint declarative (grammar module + schema
   provider); Vadalog becomes a second grammar module, not a rewrite.
2. **Validity doesn't fully transfer**: Nemo-valid ≠ Vadalog-warded-safe and
   vice versa. Honest scope: "constrain generation to valid Datalog± rules over
   KG predicates"; wardedness/streaming = the Vadalog paper's delta.
3. **Nemo is in-memory, batch**: no streaming, full result tables. Fine for the
   three-level API; streaming stays Vadalog-only.
4. **`nemo-python` is experimental**: pin the version; fallback = `nmo` CLI with
   CSV in/out via subprocess (arguably cleaner for an adapter anyway).

---

## 4. Roadmap

### Stage 1 — Get Nemo running with our KG
- [x] Install `nmo` CLI + `nemo-python` (pin 0.10.1).
- [x] Load 273k triples as Nemo facts (CSV import; header rows must be skipped).
- [x] Reproduce the 7 derived relations in Nemo syntax.
      ⚠️ Nemo does NOT support recursive aggregation → CONTROL unrolled into
      20 layers of non-recursive `#sum`; exact match with the mock.
- [x] `stage1/smoke_test.py`: all 7 relations match the mock exactly
      (verified tuple-by-tuple). Reason time ~35s vs ~4s mock.

### Stage 2 — Backend swap
- [x] Write `nemo_reasoner.py` implementing the same interface as `reasoner.py`.
- [x] Run `api.py`, `qa_gen.py`, `add_groundtruth.py`, `test.py` unchanged.
- [x] Reconcile mismatches; document aggregate-semantics differences.
      Results: test.py → all checks passed (exit 0); qa.csv and
      qa_answers.csv byte-identical to the mock's; api.py output identical.
      Swap mechanism: `stage2/run_with_nemo.py` injects nemo_reasoner into
      `sys.modules` as `reasoner` — consumer scripts untouched.
      Bug fixed along the way: `stored` must snapshot REACHABLE BEFORE the
      symmetric closure (mock stores asymmetric, serves symmetric).

### Stage 3 — Rule-generation benchmark
- [x] Define *missing-relation* tasks with expected results (hand/Python):
  - `owner_of_owner(?X,?Z) :- OWN(?X,?Y,?P), OWN(?Y,?Z,?Q) .` (2-hop join)
  - `owner_count` via `#count`, `top_stake` via `#max`,
    `total_owned` via `#sum`, `majority_owner` (on top of `top_stake`)
- [x] `<rule>` tag in the prompt; execute generated rules in Nemo; validate.
- [x] `stage3/rule_harness.py` — three-tier outcomes (invalid / valid_wrong /
      correct), near-miss catalog, negative-rule catalog.
- [x] `nemo_reasoner.add_rule` — runtime rule registration; new relation
      joins properties()/get/count/ask; verified == gold (166,954 tuples).

**Key findings (validation model):**
1. Nemo validation is TWO-STAGE: `load_string` rejects syntax / arity /
   unsafe-head-variable / no-positive-literal errors; stratification is only
   checked when the engine REASONS ("rules of the program are not
   stratified" at execution).
2. Nemo is OPEN-WORLD: rules referencing unknown predicates are accepted
   (empty relation) — closed-world "predicate must exist" checking is OUR
   semantic layer (Stage 4). Headline finding: engine validation ≠ semantic
   validation.
3. Loose aggregate typing: `#max` over strings is accepted.
4. Dead-code elimination: a predicate not reachable from @output/@export is
   silently NOT computed (`result()` yields 0 rows) — candidate heads must
   be declared with @output.
5. Near-miss quantification: `#sum(?p)` (distinct values) is wrong on 2,382
   tuples; requiring `?p > 50` on the first hop drops 96,157 of 166,954.

### Stage 4 — Constraint layer
- [x] Grammar automaton for the Nemo rule dialect (restricted safe fragment first).
- [x] Semantic candidate provider: schema (predicates+arities) + data (constants
      actually appearing — like the entity prefix tree).
- [x] Constrained decoder: pre-built "rule prefix tree" **or** on-the-fly
      candidate computation at name positions (the latter is the more
      generalizable / publishable).

> ✅ Stage 4 completed: `stage4/rule_constraint.py` (on-the-fly
> character-level automaton over the safe fragment: head/atoms/guards/
> aggregates, variable binding, arity, closed-world predicates, entity
> trie, numbers), `stage4/logits_processor.py` (HF LogitsProcessor with
> prefix lookahead over the tokenizer vocab), `stage4/test_constraint.py`:
>  - 5/5 gold rules accepted; 7/7 invalid rules dead/stuck at the right char
>  - SOUNDNESS: 176/176 constraint-enumerated rules accepted by Nemo
>  - tokenizer end-to-end: gold rule reproduced token-by-token through the
>    mask (gpt2, 37 tokens); bad continuations masked

**Key findings:**
1. The soundness test exposed two Nemo semantic rules the grammar alone
   misses: distinct-variable aggregates (`#max(?p,?p)` rejected) and
   aggregation over a group-by variable (head var in the agg is rejected).
2. Completion search needs canonicalization of entity/number positions in
   the memo (continuation after ANY entity is identical) — otherwise the
   100k-entity trie makes one lookahead ~50s; now <10ms.
3. Determinism: branch order must be sorted (set iteration is hash-random
   per process) or performance varies wildly.
4. Mask computation is ~5s/step at token-heavy positions on 50k vocab in
   pure Python — fine for validation, but the LLM benchmark will need the
   C-level fast path (as ReFactX did with its postgres trie).
5. Configurable semantic providers = the generality claim: schema dict +
   entity list are inputs; Vadalog later = new grammar module + same
   provider interface.

### Stage 5 — Evaluate + ablate

- [x] Verify the model's chat response boundary and test each constrained arm with the target tokenizer.
- [x] Report extraction, validation, engine failures, and tuple-set correctness in the Stage 5 runner.
- [x] Run the Qwen3.5-4B pilot on the dedicated CPU server.
- [ ] Evaluate held-out QA with model-generated rules registered in the backend.
- [ ] Repeat the workflow with a Vadalog backend when available.

The three arms are unconstrained, grammar with distractor predicates, and
full constraints. The two constrained arms each scored 14/15 in the sampled
pilot, compared with 8/15 unconstrained. See the
[November 2026 milestone](../MILESTONE_NEMO_RULE_GENERATION.md) for the results
and the [README](../README.md) for the current commands.

---

## 5. Open decisions

1. **Semantic candidates for rules**: schema introspection only (predicates +
   arities) or also data-level candidates (constants in columns)? → Proposal:
   both; it is precisely the syntax/semantics distinction.
2. **Constraint implementation**: pre-built rule prefix tree vs on-the-fly
   per-position candidates. → Proposal: on-the-fly (generalizes to Vadalog/Python).
3. **Cross-engine benchmark spec**: define the 6 question kinds + rule tasks in a
   dialect-agnostic way so Nemo and Vadalog backends can be compared directly.
4. **Where to keep the abstract reasoner API**: `reasoner.py` (mock) +
   `nemo_reasoner.py` + later `vadalog_reasoner.py` behind one interface
   consumed by `api.py` unchanged.

---

## 6. References

- ReFactX paper: https://arxiv.org/abs/2508.16983 (ISWC 2025,
  doi 10.1007/978-3-032-09527-5_16)
- Nemo: https://iccl.inf.tu-dresden.de/web/Nemo/en
  - GitHub: https://github.com/knowsys/nemo
  - Docs: https://knowsys.github.io/nemo-doc/
  - Paper: Ivliev et al., "Nemo: Your Friendly and Versatile Rule Reasoning
    Toolkit", KR 2024, p743-754.
  - Python bindings: `nemo-python` (PyPI, module `nmo_python`), experimental.
