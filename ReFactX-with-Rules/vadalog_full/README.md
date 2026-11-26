# Full mode with Vadalog: `<kg>` + `<rule>`

This ports the Nemo work (Stages 1–4) to the open-source Vadalog engine,
[vadalog-portable](https://github.com/joint-kg-labs/vadalog-portable). It adds the **full mode**: the
model generates freely and can switch into either constrained mode by writing a trigger.

| Trigger | Mode | The model writes | Constrained to |
|---|---|---|---|
| `<kg>` | KG | `<Subject> <relation> <Object> .` | existing entities; relations from `properties(subject)`, which covers **stored**, **derived by the Vadalog program** (unmaterialized), and **rules written earlier in the answer**; existing objects |
| `<rule>` | RULE | one Vadalog rule `head(X, ..) :- body.` | Vadalog syntax, and the semantics listed below |

After a `<rule>`, Vadalog evaluates the rule and registers its head as a new relation. The processor then forces
` </rule>` and a `<result> N facts: ... </result>` block, and free generation resumes. The new relation becomes
available to later `<kg>` and `<rule>` blocks. All of this happens inside one `model.generate()` call.

## What RULE mode guarantees

`refactx/rules/vadalog_constraint.py` is a character-level automaton (subset construction) over one rule:

- **predicates**: only predicates in the current schema, with their arity. Vadalog is open world: it would
  accept an unknown predicate as an empty relation.
- **types**: each argument position is typed (`entity`, `number`, `role`). A variable keeps its first type,
  so a stake cannot be joined with a company. Constants must be existing values: entity names come from a
  100k-name trie, roles from a trie, numbers are integers. `>`/`<` and `sum/min/max/avg` apply only to numbers.
- **safety**: every head variable is bound in the body. Otherwise Vadalog would invent a marked null. Negated
  atoms, comparisons and aggregate arguments use only bound variables. The aggregate result is a head variable.
- **head**: a fresh lower-case name that is not an existing predicate and not `is_*`. Because of that, a generated
  rule is never recursive and the program stays stratified.

The fragment covers atoms, `not atom`, comparisons, and one aggregate (`count/sum/min/max/avg`). Variables are
single upper-case letters, and there are at most 4 body literals and 3 head arguments (configurable).

`refactx/rules/token_mask.py` turns the automaton into a token mask. The vocabulary is sorted, and the mask walks
the automaton and the sorted vocabulary together, so a refused prefix prunes every token that starts with it. On
Qwen3.5 (250k tokens) a step takes milliseconds inside the body and up to ~2 s at the first token, where any fresh
relation name is allowed. That is about 3–4 s of masking per rule.

## Files

| Path | What |
|---|---|
| `refactx/rules/vadalog_constraint.py` | the rule automaton (syntax × semantics) |
| `refactx/rules/token_mask.py` | characters → allowed tokens |
| `refactx/rules/vadalog_kg.py` | `VadalogKG`: base relations + Vadalog program; `properties/get/count/ask`, `schema`, `evaluate/add_rule`, `export` |
| `refactx/rules/generation.py` | `RuleGeneration`, the `<rule>` pattern |
| `refactx/rules/full_mode.py` | `build_full_mode_processor`, `FullModeSession`, `Trace` (HTML rendering) |
| `prompts/prompt_full_mode_vadalog.yaml` | the two-tool prompt (`{SCHEMA}` is filled from the KG) |
| `ReFactX-with-Rules/vadalog_reasoner.py` | the company KG in Vadalog; drop-in `Reasoner` with the API of `reasoner.py` |
| `ReFactX-with-Rules/vadalog_full/setup_vadalog.sh` | install → generate data → reason → export → `vadalog check` |
| `ReFactX-with-Rules/vadalog_full/parity_check.py` | Vadalog vs the Python reference |
| `tests/test_vadalog_rules.py` | constraint, KG, token mask, `RuleGeneration` |
| `../try_refactx_full_mode_vadalog.ipynb` (parent repo) | the interactive notebook |

## Setup

```bash
pip install -e ".[torch,vadalog]"                          # from the ReFactX checkout
bash ReFactX-with-Rules/vadalog_full/setup_vadalog.sh /path/to/data
python ReFactX-with-Rules/vadalog_full/parity_check.py /path/to/data
python -m pytest -q tests/test_vadalog_rules.py
```

vadalog-portable is pure Python with no server: `VadalogKG` runs the engine in-process. The first build runs
the chase over the 273,594 stored facts (~25 s; it derives 1.5M facts) and caches the result in
`DATA/vadalog_cache.pkl`. The setup script also exports `DATA/vadalog/company_kg.vada` plus CSVs, which can be run
with `vadalog run` or opened in the web editor with `vadalog ui --root DATA/vadalog`.

## Usage

```python
import vadalog_reasoner as vr, refactx
from refactx.rules.full_mode import FullModeSession, fill_schema

kg = vr.build_kg("data/triples.csv")
prompt = fill_schema(refactx.load_prompt("prompts/prompt_full_mode_vadalog.yaml"), vr.DESCRIPTIONS.values())
session = FullModeSession(model, processor, kg, prompt)
trace = session.ask("How many companies does Puglisi-Bartoli Group control?")
trace            # in Jupyter: coloured FREE / KG / RULE / result blocks + a table
print(trace.summary())
```

For a custom setup, use `build_full_mode_processor(tokenizer, kg)`, or add the patterns yourself:
`processor.add_pattern('<rule>', RuleGeneration, kg=kg)`.

## Verified

- Parity with `reasoner.py`: all six derived relations match tuple by tuple; `properties`/`get` match on 2,000
  entities; all 1,110 reference QA answers match. Recursive `CONTROL` is native in Vadalog (`msum`), with no
  unrolling.
- The token mask accepts the gold tokenisation of every test rule with the Qwen3.5 tokenizer and the
  100k-entity trie.
- Qwen3.5-9B in fp16 on two T4s, greedy: the model writes rules with `count`, `sum`, `max` + follow-up, and
  `not`, and answers from `<result>`; it answers derived-fact questions with `<kg>`.
  Qwen3.5-4B handles single rules but tends to loop on `<kg>` for yes/no questions with a false claim.

## Limits and next steps

- Derived relations are computed once at build time: vadalog-portable is a bottom-up chase, so "on demand"
  means "by the reasoner, not stored in the index". Rules written during an answer are evaluated on demand,
  using only the relations they reference (~0.3–3 s).
- No recursive generated rules and no multi-rule blocks. Several `<rule>` blocks can be chained instead,
  because each one sees the previous heads.
- Variables are single letters and numbers are integers (stakes are integer percentages).
