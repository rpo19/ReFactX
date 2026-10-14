# `<vadalog>{code}</vadalog>` — implementation plan

Goal: let the model emit Vadalog rules inside a new `<vadalog> … </vadalog>`
command, have them executed by the Vadalog engine, and feed the results back
into the answer — while **constraining generation** so the emitted code is
(a) syntactically valid Vadalog and (b) grounded in existing entities,
properties and variables.

Run with: `branch feat/vadalog`.

> **Status:** Phase 0 ✅, Phase 1 ✅, Phase 2 ✅.  `refactx/vadalog.py`
> (`VadalogGeneration`, `VadalogEngine` stub), `refactx/vadalog_grammar.py`
> + `refactx/grammars/vadalog.ebnf` (xgrammar grammar),
> `refactx/vadalog_lexicon.py` (vocabulary grounding), registered in
> `utils/eval_kg.py` behind `--vadalog`, prompt
> `prompts/prompt_vadalog_nothink.yaml`, tests `tests/test_vadalog*.py`.
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

## 2. Phase 1 — grammar-constrained generation (Challenge 1)  ✅

- [x] Vadalog **EBNF** for a conservative subset (`refactx/grammars/vadalog.ebnf`):
      `rule`, `atom`, `terms`, `term`, `body`, `literal`, `variable`,
      `identifier`, `string`, `number`, `ws` (negation `~`, disjunction `;`,
      comparisons and aggregates are out of scope for v1).
- [x] Compile with xgrammar (`refactx/vadalog_grammar.py`):
      `TokenizerInfo.from_huggingface` → `GrammarCompiler.compile_grammar`,
      compiled once and cached per (tokenizer, tags).
- [x] The `root` rule wraps the content with the tags
      (`root ::= "<vadalog>" ws program ws "</vadalog>"`), assembled in code;
      `new_matcher()` consumes the opening tag so the matcher is positioned at
      the start of the content.
- [x] Wired into `VadalogGeneration.constrain`: feed new tokens with
      `accept_token`, then `fill_next_token_bitmask` →
      `apply_token_bitmask_inplace` to set the mask; the block finishes when
      `is_completed()` (closing tag accepted).  `grammar=False` disables it;
      compilation failures fall back to unconstrained.
- [x] Offline grammar tests (`tests/test_vadalog_grammar.py`) via
      `accept_string` + `is_completed` and token-level `accept_token` +
      bitmask — **no model needed**.

**Accepted:** valid programs (single/multi-rule, strings, negation) are accepted
at both string and token level; malformed ones are rejected; the generation
class drives a valid program to completion.  105 tests pass overall.

**Notes / decisions:** the closing tag is part of the grammar (cleaner than a
separate structural tag, which expects a JSON-schema content).  `is_completed`
— not `is_terminated` — marks the end.  `max_code_tokens` forced-close bypasses
the grammar as a safety net.

**Risks:** tokenizer whitespace/quote handling (xgrammar token healing + explicit
`ws`); grammar growth is iterative.

---

## 3. Phase 2 — entity & relation lexicons (Challenge 2a)  ✅

**Approach chosen: encode the vocabulary into the grammar as alternations**
(`refactx/vadalog_lexicon.py`), rather than a separate token mask intersected at
run time.  xgrammar then handles tokenization, so this is token-boundary safe and
needs no runtime slot tracker.  Cost is grammar size → relations/variables are
cheap, entities are capped (see below).

- [x] **Body predicates** (EDB) → `body_pred ::= "own" | "is_own" | ...`.
- [x] **Head predicates** (IDB) → `head_pred ::= "control" | ...`; defaults to the
      body set; forbids using an EDB name as a fresh head-fact predicate.
- [x] **Variables** → a small alphabet (`"X" | "Y" | "Z"`).
- [x] **Entity constants** → `string ::= "\"" ("Acme Corp" | "Globex") "\""`,
      optional and bounded by `MAX_ENTITIES` (a few thousand); oversized sets
      raise, so the caller passes a question-relevant subset.
- [x] `collect_relations(api, sample=...)` builds the predicate set from
      `api.property_strings`.
- [x] Wired via `VadalogGeneration(lexicon={...})`; unset terminals fall back to
      the generic grammar (Phase 1).
- [x] Tests: `tests/test_vadalog_lexicon.py` (render, collect, accept/reject a
      grounded grammar, and a token-level generation test).

**Deviation from the original plan:** we don't use a `DictIndex` + slot tracker +
mask intersection.  Rationale: grammar-level alternation is simpler, exact w.r.t.
tokenization, and sufficient for the small predicate/variable sets; the
intended `DictIndex`/tracker route is the fallback for **unbounded** entity sets
(100k+), which we cap instead.  If full-KG entity grounding is needed, add a
Phase 2b using a char-trie ∧ vocab-token-trie mask (token-boundary safe).

**Acceptance:** generated atoms only reference grounded predicates; entity
arguments only reference the entity set; variables only the alphabet; invalid
names are rejected by the grammar.  ✅ (112 tests pass overall).

**Files:** `refactx/vadalog_lexicon.py`, `refactx/vadalog.py`, `tests/test_vadalog_lexicon.py`.

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
