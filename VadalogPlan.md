# `<vadalog>{code}</vadalog>` — implementation plan

Goal: let the model emit Vadalog rules inside a new `<vadalog> … </vadalog>`
command, have them executed by the Vadalog engine, and feed the results back
into the answer — while **constraining generation** so the emitted code is
(a) syntactically valid Vadalog and (b) grounded in existing entities,
properties and variables.

Run with: `branch feat/vadalog`.

> **Status:** Phase 0 ✅ done (`refactx/vadalog.py`, `VadalogEngine` stub,
> registered in `utils/eval_kg.py` behind `--vadalog`, trivial prompt
> `prompts/prompt_vadalog_nothink.yaml`, tests `tests/test_vadalog.py`).
> The `vadalog` engine is not installed here, so execution degrades to a
> structured error until Phase 4.

---

## 0. Context / existing building blocks

| Piece | Where | Why it matters |
|---|---|---|
| Pattern registration | `refactx.generate.ConstrainedLogitsProcessor.add_pattern` | how `<vadalog>` is wired like `<kg>` |
| Sliding-window trigger | `PatternConstrainedState` (`_decode_recent` / `update`) | pattern detection |
| Constrained generation base | `PatternConstrainedGeneration.constrain` | where token masks are written |
| Collect-until-delimiter example | `CountBranchesGeneration` | template for collecting a block |
| KB trie + per-phase vocab | `DictIndex`, `KnowledgeGraphGeneration._make_index` | entity/relation lexicons |
| **Grammar constraining** | `refactx/xgrammar_matcher.py` (`feat/xgrammar-matcher`), xgrammar (`Grammar.from_ebnf`, `compile_grammar`, `compile_structural_tag`, `GrammarMatcher.fill_next_token_bitmask`, `apply_token_bitmask_inplace`) | Challenge 1 |
| Alt backends | `feature/outlines-backend` (FSM/regex) | fallback if xgrammar is not desired |

Design principle: **grammar = structure, lexicons = vocabulary**, intersected at
decode time. Never bake the (huge, shifting) entity/relation/variable sets into
the EBNF.

---

## 1. Phase 0 — plumbing (no real constraints yet)  ✅

- [x] Create `refactx/vadalog.py` with `VadalogGeneration(PatternConstrainedGeneration)`.
  - [x] `eot='</vadalog>'`; collect tokens between `<vadalog>` and `</vadalog>`.
  - [x] `constrain()` only enforces the closing tag (allows everything else),
        stores the decoded code on the state (`state.vadalog_blocks`), and
        force-closes when `max_code_tokens` is exceeded.
- [x] Register the pattern: `add_pattern('<vadalog>', VadalogGeneration, execute=True)`
      in `utils/eval_kg.py` behind a `--vadalog` flag (notebook wiring deferred to Phase 5).
- [x] Execute the collected code with a `VadalogEngine` stub (no-op / structured
      error when the engine is missing); echo/store raw results.
- [x] Add a trivial prompt: `prompts/prompt_vadalog_nothink.yaml`.
- [x] Tests: `tests/test_vadalog.py` (unconstrained block, model close, forced
      close, execute toggles, engine failure/unavailable).

**Accepted:** the class activates from the state machine, collects a block,
hands it to a stub executor and stores the result; existing tests still pass
(98 tests total).

**Files:** `refactx/vadalog.py`, `refactx/__init__.py`, `refactx/generate.py`
(`state.vadalog_blocks`), `utils/eval_kg.py`, `prompts/prompt_vadalog_nothink.yaml`,
`tests/test_vadalog.py`.

---

## 2. Phase 1 — grammar-constrained generation (Challenge 1)

- [ ] Write a Vadalog **EBNF** for a conservative subset (`refactx/grammars/vadalog.ebnf`):
  - [ ] rule: `head :- body .`
  - [ ] atom: `pred ( term { , term } )`
  - [ ] term: variable | string | number | arithmetic expr (optional)
  - [ ] body: atoms joined by `,` and `;`, negation `~`, comparisons
  - [ ] optional whitespace between tokens (Vadalog is whitespace-insensitive)
  - [ ] explicitly **out of scope** in v1: recursion, aggregates, `@` annotations
- [ ] Compile with xgrammar:
  - [ ] `TokenizerInfo.from_huggingface(tokenizer)` → `GrammarCompiler`
  - [ ] `compiler.compile_grammar(ebnf)` (or `compile_structural_tag` wrapping the
        `<vadalog>` tags with the content grammar)
  - [ ] cache the `CompiledGrammar` (compile once)
- [ ] Wire the matcher into `VadalogGeneration.constrain`:
  - [ ] `matcher.accept_token(last)` → `fill_next_token_bitmask(bitmask, 0)` →
        `apply_token_bitmask_inplace(scores, bitmask)`
  - [ ] handle `is_terminated()` / tag close
- [ ] Offline grammar tests (`tests/test_vadalog_grammar.py`) using
      `GrammarMatcher.accept_string` + `is_completed()`: accept valid rules,
      reject malformed ones — **no model needed**.

**Acceptance:** model output inside `<vadalog>` always parses as Vadalog (subset);
grammar tests pass; malformed inputs are rejected by the matcher.

**Risks:** tokenizer whitespace/quote handling (mitigate with token healing and
explicit optional-space in the EBNF); grammar growth is iterative.

---

## 3. Phase 2 — entity & relation lexicons (Challenge 2a)

Bring the vocabulary under control (same idea as `KnowledgeGraphGeneration`).

- [ ] **Entity lexicon**: `DictIndex` built from `api.entity_index()` (reuse the
      eval's index construction).
- [ ] **Predicate/relation lexicon**: `DictIndex` of schema predicates
      (`api.property_strings` universe), split by position:
  - [ ] *body* atoms (EDB) → existing predicates only;
  - [ ] *head* atom (IDB) → allow a known target set (e.g. `control`,
        `ultimate_control`) and forbid collisions with EDB names.
- [ ] Track the **expected slot** while parsing (entity | relation | number |
      variable | none) — a lightweight incremental parser parallel to the matcher
      (or derive it from the grammar stack).
- [ ] In `constrain`, intersect: `allowed = grammar_bitmask ∩ lexicon_tokens(slot)`.
- [ ] Normalization: one canonical quoting/escaping for entities, shared by the
      lexicon builder and (later) the executor.

**Acceptance:** generated atoms only reference existing predicates; entity
arguments only reference entities present in the index; invalid names never
reach the model's output.

---

## 4. Phase 3 — variables & safety (Challenge 2b)

The part a plain grammar/regex cannot express — needs decoder state.

- [ ] Maintain a **symbol table** (`VadalogState`): variables introduced so far.
- [ ] Vadalog is head-first, so:
  - [ ] head variables: introduce freely from a small closed alphabet
        (`X, Y, Z, …`);
  - [ ] body variables: constrain to the already-introduced set (rebuild the
        variable lexicon each time a name is added).
- [ ] Enforce **range restriction / safety**: every head variable must appear in
      the body. Enforce live (body vars ⊆ head vars) and/or validate + reject
      after generation.
- [ ] Policy decision (see Open questions): allow body-only variables or not.

**Acceptance:** every emitted rule is safe (head vars ⊆ body vars); repeated
variable names within a rule are consistent.

---

## 4b. Phase 3b — argument sorts / predicate signatures ("type-constrained")

Extend grounding with **types**: treat each predicate as a typed function and
require its arguments to inhabit the expected sort.  This is the idea from
*Type-Constrained Code Generation with Language Models* (Mündler et al., PLDI
2025, arXiv 2504.09246) adapted to a relational language — their "search over
inhabitable types" becomes "is there a term of this sort for this position?".

- [ ] `PredicateSignature(name, arity, arg_sorts)` + a `SignatureCatalog`.
- [ ] **Discover** signatures from what we already have:
  - [ ] base sorts from the entity `type` column
        (`data/entities.csv`: `company` / `person`);
  - [ ] per-relation **domain/range** observed over the entity set
        (`get_relations` / `get_objects`, or the reasoner);
  - [ ] declared signatures for derived/IDB predicates
        (`control`, `ultimate_control`, …).
- [ ] **Sort inference / unification**: each variable gets the intersection of
      the domains of the positions it occurs in; require a non-empty,
      consistent sort (a variable cannot be both `company` and `person`).
- [ ] Constrain **constant** arguments to entities of the required sort
      (lexicon filtered by sort).
- [ ] **Arity** check: atoms must match the predicate arity (grammar + signature).
- [ ] **Inhabitable-term / reachability** check: at a position requiring sort
      `S`, ensure some variable-of-sort-`S` or constant-of-sort-`S` can inhabit
      it; otherwise mask (dead end) — the direct analogue of their type
      reachability.
- [ ] **Head predicate policy**: head must be a declared IDB (or a freshly
      declared predicate) with a signature; forbid reusing an EDB name as head.

**Sources for the signatures (semantic web).** We can bootstrap the catalog
from standards rather than hand-writing it:

| Source | Maps to |
|---|---|
| entity `type` column | base sorts (`company`, `person`) |
| OWL/RDFS `rdfs:domain` / `rdfs:range` | per-argument sorts of a predicate |
| `owl:ObjectProperty` / `owl:DatatypeProperty` | object vs literal argument |
| Freebase / Neo4j relation types | domain/range per relation |
| **SHACL** `sh:path` + `sh:class` | allowed object class (range) |
| SHACL `sh:datatype` / `sh:nodeKind` | literal type / IRI vs literal |
| SHACL `sh:minCount` / `sh:maxCount` | cardinality (catalog sanity) |
| SHACL `sh:in` | enumerations (allowed constants) |
| SHACL `sh:or` / `sh:and` | disjunction/conjunction of allowed sorts |
| Vadalog data types (`int`, `string`, …) | literal sorts |

**Acceptance:** every argument is sort-consistent, every variable has a coherent
sort, no position is uninhabitable, arities match the catalog.

---

## 5. Phase 4 — execution & result injection

- [ ] `VadalogEngine` adapter: parse the block, run it through the `vadalog`
      package / `vada` CLI, return derived facts.
  - [ ] detect/execute derived predicates (`control`, `ultimate_control`, …).
- [ ] Store results on the state and expose them to the model like
      `get_objects` does for `<kg>` (so following reasoning / lookups can use them).
- [ ] Error handling: engine errors → surface a compact message, do not crash the
      run; optionally trigger a retry.
- [ ] Determinism/timeout guards on the engine call.

**Acceptance:** a `<vadalog>` block runs, results are visible in the transcript and
reusable; engine failures degrade gracefully.

---

## 6. Phase 5 — integration

- [ ] Prompt: document the command, the available predicates/entities, and the
      safety requirement (few-shot examples).
- [ ] `utils/eval_kg.py`: register `<vadalog>` (optionally behind a flag
      `--vadalog`), log the blocks and engine results per record.
- [ ] `utils/logs_viewer.py` / `utils/log_analysis.py`: show/aggregate `<vadalog>`
      usage (analogous to the `<kg>`/sentinel metrics).
- [ ] Config `configs/*.json`: new options.

**Acceptance:** end-to-end run on the rules benchmark produces and executes
`<vadalog>` blocks; logs and viewer expose them.

---

## 7. Phase 6 — tests & docs

- [ ] Grammar unit tests (Phase 1) — offline.
- [ ] Lexicon unit tests: entity/relation/variable masking — offline (`DictIndex`).
- [ ] `VadalogGeneration.constrain` tests mirroring `tests/test_knowledge_graph_generation.py`
      (hand-fed sequences → expected masks).
- [ ] Engine adapter tests with tiny fixed rules.
- [ ] Docs: this file + a short section in `Readme.md`.

---

## 8. Phase 7 — advanced (later)

- [ ] Extend the grammar: aggregates (`#sum`, `#count`), arithmetic, recursion
      guards, `@` annotations.
- [ ] Header/head predicates from a discoverable IDB catalog.
- [ ] Cache/compile grammar + lexicons once per process; measure per-token cost.
- [ ] Fallback grammar backend (`feature/outlines-backend`) behind a flag.

---

## Open questions / decisions

1. **Representation of entity constants**: quoted strings (`"Gremese e figli"`) vs
   atoms — pick one canonical form used by grammar, lexicon and engine.
2. **Head predicate policy**: fixed target set vs free-but-distinct names.
3. **Body-only variables**: allowed, or restrict to head-introduced names only?
4. **Grammar backend**: commit to xgrammar, or keep `outlines` as a fallback?
5. **Failure mode on engine error**: hard-stop the `<vadalog>` block vs retry.
6. **Scope of v1 grammar**: which Vadalog features are in/out (recursion,
   aggregates, disjunction).
7. **Sort granularity**: base sorts only (`company`/`person`) vs finer classes
   (subclasses, Freebase types); how strict (hard constraint vs soft prior)?
8. **Signature source**: data-derived domain/range vs OWL (`rdfs:domain/range`)
   vs **SHACL shapes** vs hand-declared — and how to merge/override them.
9. **Subsumption**: is `company` a subtype of `legal entity`? Needed for
   subclass-aware sort unification; do we model a small class hierarchy?

---

## Related work / inspiration

- **Type-Constrained Code Generation with Language Models** — Mündler, He, Wang,
  Sen, Song, Vechev (PLDI 2025, arXiv 2504.09246).
  Repo: `github.com/eth-sri/type-constrained-code-generation`.
  *Same paradigm* (constrained decoding with an incremental, context-carrying
  parser; "every prefix completable to a valid program") but for a *typed GPL
  (TypeScript)*; the central problem there is **type reachability / inhabitable
  types**, which maps here to the Phase 3b "inhabitable-term" check.  Their
  `ExistingIdentifierParserState` (identifiers restricted to in-scope ones) is
  the blueprint for our **variable symbol table** + entity/property lexicons.
  They hand-roll the automaton per construct because a generic CFG can't carry
  semantics; we keep a generic grammar engine (xgrammar) for syntax and add a
  **semantic side-channel** for grounding/sorts.
- **Semantic web schema sources**: RDFS/OWL (`rdfs:domain`, `rdfs:range`,
  `owl:ObjectProperty`/`DatatypeProperty`), Freebase schema, and **SHACL**
  shapes — see the mapping table in Phase 3b.

---

## Risks

- Tokenizer quirks around whitespace / quotes / newlines (xgrammar healing helps).
- Keeping the parallel parser's "expected slot" in sync with the grammar stack.
- Grammar↔lexicon mask intersection must be correct and cheap per token.
- Engine variability / non-termination on unsafe rules (mitigate with validation).
- Signature discovery noise: observed domain/range can be incomplete/over-broad;
  sort unification must degrade gracefully (widen, don't dead-end).
