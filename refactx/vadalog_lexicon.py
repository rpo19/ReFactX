"""Lexicon grounding for ``<vadalog>`` decoding (Phase 2 / Challenge 2a).

Instead of intersecting a separate token mask, we **bake the allowed vocabulary
into the grammar** as alternations.  xgrammar then handles tokenization, so this
is token-boundary safe and needs no extra runtime machinery.

Groundable vocabularies (all optional; unset ⇒ generic grammar terminal):

* **body predicates** — the existing relations (EDB); a rule body may only use
  these;
* **head predicates** — the derived / declared relations (IDB); a rule head may
  only use these (and may not collide with the body ones);
* **variables** — a small variable alphabet;
* **entity constants** — the entities a string argument may name.

Cost is grammar size, so entity sets must stay modest: compiling an alternation
of ~10k names is a couple of seconds, 100k is too much.  For a full KG, pass the
subset that is relevant to the current question (or ground entities in a later
phase via a dynamic token-trie).
"""
from __future__ import annotations

MAX_ENTITIES = 4000


def _escape(text):
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _alt(values):
    """A GBNF alternation ``("a" | "b" | ...)`` or ``None`` when empty."""
    if not values:
        return None
    values = [v for v in dict.fromkeys(values) if v != ""]
    if not values:
        return None
    return "(" + " | ".join(f'"{_escape(v)}"' for v in values) + ")"


def render_content_ebnf(body_relations=None, head_relations=None,
                        variables=None, entities=None, max_entities=MAX_ENTITIES):
    """The ``program`` grammar, with lexicon terminals substituted in."""
    body = _alt(body_relations)
    head = _alt(head_relations)
    var = _alt(variables)
    ent = _alt(entities)

    if ent is not None and entities and len(set(entities)) > max_entities:
        raise ValueError(
            f"entity lexicon has {len(set(entities))} names (> max_entities="
            f"{max_entities}); pass a smaller, question-relevant subset")

    body_pred = body if body is not None else "[a-z] [a-zA-Z0-9_]*"
    head_pred = head if head is not None else body_pred
    variable = var if var is not None else "[A-Z] [A-Za-z0-9_]*"
    string = f'"\\"" {ent} "\\""' if ent is not None else '"\\"" ([^"\\\\] | "\\\\" [^\\n])* "\\""'

    return f"""program ::= rule (ws rule)*
rule ::= head_atom ws ":-" ws body ws "." | head_atom ws "."
head_atom ::= {head_pred} ws "(" ws terms? ws ")"
body ::= literal (ws "," ws literal)*
literal ::= "~" ws body_atom | body_atom
body_atom ::= {body_pred} ws "(" ws terms? ws ")"
terms ::= term (ws "," ws term)*
term ::= {variable} | string | number
variable ::= {variable}
string ::= {string}
number ::= "-"? [0-9]+ ("." [0-9]+)?
ws ::= [ \\t\\n\\r]*
"""


def build_grounded_ebnf(open_text, close_text, **lexicon):
    """Assemble ``root`` (with tags) around the lexicon-grounded content."""
    return f'root ::= "{open_text}" ws program ws "{close_text}"\n' \
        + render_content_ebnf(**lexicon)


def collect_relations(api, entities=None, sample=None):
    """Union of the relations offered by ``api.property_strings``.

    ``sample`` limits how many entities are queried (the schema is usually
    already covered by a small sample), keeping this cheap on large KGs.
    """
    entities = entities if entities is not None else api.entity_index()
    if sample is not None:
        entities = entities[:sample]
    relations = set()
    for entity in entities:
        relations.update(api.property_strings(entity))
    return sorted(relations)
