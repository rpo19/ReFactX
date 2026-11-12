# Nemo Design Analysis & Integration Plan

> Historical design notes. See the [current milestone](../MILESTONE_NEMO_RULE_GENERATION.md) and [testing guide](../README.md).

> How to build a Nemo-backed version of the ReFactX-with-Rules reasoner,
> based on studying the cloned repo at `~/Documents/Nemo` (main branch,
> v0.10.2-dev, workspace edition 2024).

---

## 1. Nemo's architecture (what we learned from the code)

### 1.1 Workspace crates

| Crate | Role |
|---|---|
| `nemo/` | Core: parser (AST), rule model, pipeline (parse→transform→validate), execution engine (chase), io (import/export) |
| `nemo-python/` | PyO3 bindings → Python module `nmo_python` (marked experimental) |
| `nemo-cli/` | The `nmo` CLI client |
| `nemo-physical/` | Physical layer (columnar storage) |
| `nemo-language-server/`, `nemo-wasm/`, `xtask/` | LSP, browser build, dev tools |

### 1.2 Core Rust API (`nemo/src/api.rs`)

```rust
load_program_handle(input, label) -> Result<ProgramHandle, ProgramReport>
load_program(input, label)       -> Result<Program, ProgramReport>   // materialized
validate(input, label)           -> ProgramReport                     // errors + warnings
load_string(input)               -> Engine (async)                    // parse + feature check
reason(&mut engine)              -> ()                                 // execute the chase
output_predicates(engine)        -> Vec<Tag>
```

- **Validation is a first-class API**: `ProgramReport` collects warnings + errors
  with source positions (ariadne-based rich rendering). This is our **rule
  validator** — the feedback channel for LLM-generated rules.
- `load_program_handle` keeps the pipeline (parse → transformation →
  validation) intact so the same program can drive parsing, execution **and
  tracing** (origin resolution back to source).

### 1.3 Python bindings (`nemo-python/src/lib.rs`, module `nmo_python`)

| Symbol | What it does |
|---|---|
| `load_string(rules)` / `load_file(path)` | Parse + validate → `NemoProgram`; on error raises `NemoError` with the **full formatted report** (syntax + validation errors) |
| `NemoProgram.output_predicates()` | `Vec<String>` of exported predicate names → **schema introspection** |
| `NemoProgram.edb_predicates()` | `HashSet<String>` of imported (EDB) predicates |
| `NemoEngine(program)` | Built from a `NemoProgram` (creates its own tokio runtime) |
| `engine.reason()` | Execute the chase (blocking) |
| `engine.result(predicate)` | Iterable `NemoResults` (Python iteration, one tuple per row) → **streaming results** |
| `engine.trace(fact_string)` | `NemoTrace` tree: rule applications + variable assignments + subtraces → **explainable reasoning traces** (matches note.md's "explainable trace" goal) |
| `engine.write_result(predicate, output_manager)` | Export via `NemoOutputManager(path, overwrite, gzip)` |
| `engine.timing()` | `NemoTiming` tree with process/system/thread times |
| `NemoLiteral` | RDF literal support |

**Gap spotted**: Rust has `engine.predicate_arity(&tag)`, but it is **not
exposed in the Python bindings**. Workarounds (see §4.3).

### 1.4 Language syntax (`nemo/src/syntax.rs`)

Every token of the language is a constant in one module — a machine-readable
vocabulary, ideal for the grammar-constraint layer:

- Statements: directives (`@import`, `@export`, `@base`, `@prefix`,
  `@parameter`, `@declare`, `@output`), facts, rules; delimiter `.`
- Rule: `HEAD :- BODY .`; attributes `#[...]`; atom negation `~`
- Aggregates: `#count`, `#min`, `#max`, `#sum` — one aggregate per rule,
  distinct-value semantics
- Variables: universal `?X`, existential `!X` (head-only, TGDs), global `$X`
- Operators: `> >= < <= = != + - * /`
- CSV import (verified in docs):
  `@import table :- csv { resource="path.csv", format=(int,string,any) } .`

The parser AST mirrors this in `nemo/src/parser/ast/` (program, statement,
rule, expression, term, tag, directive, guard...).

---

## 2. How this maps onto ReFactX-with-Rules

The current repo hides the engine behind an abstract interface
(`reasoner.py`): `properties(e)`, `get/count/ask`, `facts()`. A Nemo-backed
adapter must implement the **same interface** so that `api.py`, `qa_gen.py`,
`add_groundtruth.py`, `test.py` run unchanged.

| ReFactX-with-Rules | Nemo equivalent |
|---|---|
| `reasoner.py` (Python mock) | `nemo_reasoner.py` wrapping `nmo_python` |
| `load(triples.csv)` | `@import` CSV directives or fact strings in the program |
| derived relations (7 rules) | rules in Nemo syntax, computed by `engine.reason()` |
| `properties(e)` | lookups over `engine.result(p)` for each predicate |
| `get/count/ask` | same lookups (cached in-memory indexes after one `reason()`) |
| `facts()` (verbalized triples for the prefix tree) | stored relations only, as before |
| `add_rule` | append rule to program string → reload (`load_string` is cheap for 274k facts? see §5) |
| — | `engine.trace()` → the explainable-trace story from note.md |

---

## 3. The rule-generation constraint layer

Goal: `<rule>…</rule>` generation constrained by **syntax ∩ semantics**.

### 3.1 Syntax layer

Do **not** try to constrain the full Nemo language initially. Define a **safe
rule fragment** whose grammar we control:

```
<rule>        := <head> ":-" <body> "."
<head>        := <predicate> "(" <terms> ")"        // 1 head atom
<body>        := <atom> ("," <atom>)*               // positive atoms + guards
<atom>        := <predicate> "(" <terms> ")"
<guard>       := <term> <op> <term>                 // comparisons
<aggregate>   := "#" (count|min|max|sum) "(" ... ")" // one per rule, head position
<term>        := <var> | <const>
<var>         := "?" <name>
```

The full token vocabulary is already centralized in `syntax.rs` — port it as
the constrainer's terminal symbols. Later: extend the fragment (negation `~`,
existentials `!X`, arithmetic expressions) and, ultimately, derive a grammar
from the parser AST itself.

### 3.2 Semantic layer (per-position candidates)

- **Predicates + arities**: from `output_predicates()` / `edb_predicates()` +
  arity info (§4.3). Only existing predicates can be generated; arity
  constrains the term list length.
- **Constants**: the existing entity prefix tree (level 1) — constants must be
  entities from the KG.
- **Variables**: constrained structurally (fresh vs already-bound in the
  rule), not by content.

### 3.3 Validation & feedback loop

1. LLM generates `<rule>…</rule>` under the constraint.
2. `load_string(program + rule)` → success or `NemoError` with the full
   formatted report (rich, positioned errors).
3. If valid: `engine.reason()`, then compare `engine.result(new_predicate)`
   against expected gold results (hand/computed).
4. Metrics: % syntactically valid, % executing, % logically correct.
   Ablations: unconstrained vs syntax-only vs syntax+semantics.

---

## 4. Design decisions

### 4.1 Which artifact: wheel vs build from source

- **Start**: `pip install nemo-python==0.10.1` (PyPI wheel).
- **Fallback**: build from the cloned repo with `maturin develop` in
  `nemo-python/` (needs Rust; repo has `rust-toolchain.toml`). Only needed if
  we add bindings (e.g. `predicate_arity`) or hit wheel bugs.

### 4.2 How to load the KG

- **Option A (facts in the program string)**: 274k facts + 7 rules in one
  `load_string` call. Simplest; parse time may be non-trivial but one-time.
- **Option B (CSV import)**: `@import own :- csv { resource="data/own.csv",
  format=(string,string,string,double) } .` — cleaner, and CSV is already the
  format `kg_gen.py` produces. Likely preferred.
- Measure both; pick per speed.

### 4.3 Arity introspection (gap)

Rust exposes `predicate_arity`; Python doesn't. Options:
1. Derive arity from the first row of `engine.result(p)` after `reason()`
   (zero-result predicates are irrelevant for constraint anyway).
2. Patch the cloned bindings (add a `predicate_arity` method, build with
   maturin) — keeps the PyPI wheel untouched only if we vendor the patch;
   acceptable for research, and a possible upstream PR.

### 4.4 `add_rule` efficiency

Re-loading the whole program per added rule may be too slow at scale. For the
benchmark, batch rules per question or reload once per question set. Longer
term: keep one engine alive and use Nemo's program-update API if available
(check `ProgramHandle`/engine mutation API in Rust).

---

## 5. Risks / experiments needed (Stage 1) — RESULTS

> ✅ Stage 1 completed: all 7 derived relations match the mock **exactly**
> (QUALIFIED_HOLDINGS 151,600; INFLUENCE 171,943; CONNECTED 760,110;
> CONTROL 160,096; ULTIMATE_CONTROL 62,763; INDIRECT_ROLE 45,326;
> symmetric REACHABLE 173,376). See `stage1/smoke_test.py`.
> Timing on this Mac: load 0.1–0.2s, reason 34.7s (vs 4.2s mock).

1. **Recursive aggregation is NOT supported** (v0.10.1): a CONTROL rule with
   `#sum` over a recursive relation is rejected as "rules of the program are
   not stratified". **Solution**: bounded unrolling — the mock's fixpoint
   converges in ≤ 16 rounds on this KG (measured), so unroll into L=20 layers
   of non-recursive `#sum` (`acc_n`, `direct_sum_n`, `new_n`). Exact match.
   Documented as an expressivity delta vs Vadalog (monotonic aggregation).
2. **Aggregate distinct-value semantics pitfall**: `#sum(?p)` sums distinct
   *values* — two different owners holding the same stake would undercount.
   Fix: `#sum(?p, ?z)` sums over distinct (value, owner) tuples → true sum.
3. **CSV import reads the header row as data** (`('subject','object','value')`
   appeared as a fact). Fix: write CSVs with `header=False`.
4. **Self-loop guard**: transitive closure derives `(x,x)` via symmetry;
   the mock computes component-minus-self. Fix: `?x != ?y` guard on CONNECTED.
5. **Bindings work fine** (arm64 wheel, cp312): `@import` CSV, quoted string
   constants (entities with spaces/commas), case-sensitive predicates
   (`OWN` ≠ `own`), guards, stratified negation `~`, `@output` for schema
   introspection, `engine.result()` iteration. `NemoError` messages are
   clean and human-readable ("predicate `ROLE` used with arity 2",
   "rules of the program are not stratified") — ideal validator feedback
   for the LLM loop.

---

## 6. Staged coding plan

### Stage 1 — Smoke-test Nemo with our KG (nemo_reasoner.py skeleton)
- [x] `pip install nemo-python==0.10.1` (or maturin build from `~/Documents/Nemo/nemo-python`)
- [x] Load `data/triples.csv` (facts or CSV import) + the 7 rules in Nemo syntax
- [x] `reason()` + spot-check results vs `reasoner.py` mock
- [x] Resolve CONTROL expression in Nemo dialect; document any semantic delta

### Stage 2 — Adapter behind the existing interface
- [x] `nemo_reasoner.py` with `properties/get/count/ask/facts` + in-memory
      indexes after one `reason()`
- [x] Run `api.py`, `qa_gen.py`, `add_groundtruth.py`, `test.py` unchanged
- [x] Reconcile all mismatches; note aggregate-semantics differences

> ✅ Stage 2 completed. `test.py`: all checks passed. `qa.csv` and
> `qa_answers.csv` byte-identical to the mock run; `api.py` walkthrough
> identical. Swap mechanism: `stage2/run_with_nemo.py` pre-registers
> `nemo_reasoner` in `sys.modules` as `reasoner`, so the repo scripts are
> never modified. The adapter materializes Nemo results into the mock's
> data structures (`out`/`inv`/`stored`/`pct`) and reuses the mock's
> methods verbatim. Build time ~35–45s (reason 21–35s + Python
> materialization of ~2M result rows).

### Stage 3 — Rule-generation benchmark
- [x] Task set (expected results precomputed):
      `owner_of_owner` (2-hop join), `owner_count` (#count), `top_stake`
      (#max), `total_owned` (#sum), `majority_owner` (built on top_stake)
- [x] `<rule>` tag in the prompt; validation loop via `load_string` +
      `NemoError` reports; execution check vs gold
- [x] `stage3/rule_harness.py` — three-tier outcomes (invalid /
      valid_wrong / correct), near-miss + negative-rule catalogs
- [x] `nemo_reasoner.add_rule` — runtime rule registration; new relation
      joins properties()/get/count/ask; verified == gold (166,954 tuples)

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
- [x] Port the safe-fragment grammar (tokens from `syntax.rs` semantics)
- [x] Semantic candidate provider: schema (predicates+arities) + data (entity index)
- [x] Constrained decoder: pre-built rule prefix tree first; on-the-fly
      per-position candidates second (the generalizable one)

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
