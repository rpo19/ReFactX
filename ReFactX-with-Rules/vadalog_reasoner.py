"""Vadalog-backed reasoner for the synthetic company-ownership KG.

Same interface as ``reasoner.py`` / ``nemo_reasoner.py`` (``Reasoner(triples)``
with ``properties/get/count/ask/facts/add_rule``), so ``api.py``,
``test.py`` and the stage-2 runner work unchanged, plus the hooks of the
full ``<kg>`` + ``<rule>`` mode (``kg``: a :class:`refactx.rules.VadalogKG`).

Unlike Nemo, Vadalog supports recursion through aggregation with monotonic
aggregates, so CONTROL is the textbook rule -- no unrolling::

    ctl(X, X) :- own(X, Y, P).
    ctl(X, Y) :- ctl(X, Z), own(Z, Y, P), S = msum(P, <Z>), S > 50.

Vadalog variables are upper case, so predicates are lower case
(``own`` = OWN); the property strings are the same as the mock's
(``own``, ``is_own``, ``control``, ``is_control`` ...).

Usage:  python3 vadalog_reasoner.py [path/to/triples.csv]
"""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))           # the ReFactX checkout

from refactx.rules.vadalog_kg import VadalogKG  # noqa: E402

BASE = ["OWN", "ROLE", "REACHABLE"]
SYMMETRIC = {"REACHABLE", "CONNECTED"}
RULES = ["QUALIFIED_HOLDINGS", "INFLUENCE", "CONTROL", "ULTIMATE_CONTROL",
         "CONNECTED", "INDIRECT_ROLE"]
DATA_DIR = os.path.join(HERE, "data")

PROGRAM = """\
% ---- company-ownership KG: the rules of reasoner.py, in Vadalog ----------
qualified_holdings(X, Y) :- own(X, Y, P), P >= 10.
influence(X, Y) :- own(X, Y, P), P >= 20.
influence(X, Y) :- role(X, Y, R).

% REACHABLE is symmetric; CONNECTED is its transitive closure
reach(X, Y) :- reachable(X, Y).
reach(X, Y) :- reachable(Y, X).
connected(X, Y) :- reach(X, Y), X != Y.
connected(X, Y) :- connected(X, Z), reach(Z, Y), X != Y.

% X controls Y when the shares of Y held by X and by the companies X
% controls sum to more than 50% (recursion through a monotonic sum)
ctl(X, X) :- own(X, Y, P).
ctl(X, Y) :- ctl(X, Z), own(Z, Y, P), S = msum(P, <Z>), S > 50.
control(X, Y) :- ctl(X, Y), X != Y.

is_controlled(Y) :- control(X, Y).
ultimate_control(X, Y) :- control(X, Y), not is_controlled(X).
indirect_role(X, Y) :- role(X, Z, R), control(Z, Y).
"""

E, N, ROLE_T = "entity", "number", "role"
SCHEMA = {
    "own": (E, E, N), "role": (E, E, ROLE_T), "reachable": (E, E),
    "qualified_holdings": (E, E), "influence": (E, E), "control": (E, E),
    "ultimate_control": (E, E), "connected": (E, E), "indirect_role": (E, E),
}
DERIVED = ["qualified_holdings", "influence", "control", "ultimate_control",
           "connected", "indirect_role", "reach"]

# What each predicate means -- used in prompts and in the notebook.
DESCRIPTIONS = {
    "own": "own(Owner, Company, Percent): Owner owns Percent% of Company (stored)",
    "role": "role(Person, Company, Role): Person has a role (ceo, chairman, board member, director, general manager, auditor) in Company (stored)",
    "reachable": "reachable(A, B): A is directly connected to B, in both directions (stored)",
    "qualified_holdings": "qualified_holdings(X, Y): X has qualified holdings in Y, i.e. owns >= 10% of it (rule)",
    "influence": "influence(X, Y): X has significant influence over Y: owns >= 20% of it or has a role in it (rule)",
    "control": "control(X, Y): X controls Y, directly or through the companies it controls (recursive rule)",
    "ultimate_control": "ultimate_control(X, Y): X is the ultimate controller of Y: controls it and nobody controls X (rule)",
    "connected": "connected(X, Y): X is connected to Y through a chain of direct connections (recursive rule)",
    "indirect_role": "indirect_role(X, Y): X has a role in an entity that controls Y (rule)",
}


def base_relations(triples):
    own = triples[triples.predicate == "OWN"]
    role = triples[triples.predicate == "ROLE"]
    reach = triples[triples.predicate == "REACHABLE"]
    return {
        "own": list(zip(own.subject, own.object, own.value.astype(float))),
        "role": list(zip(role.subject, role.object, role.value)),
        "reachable": list(zip(reach.subject, reach.object)),
    }


def build_kg(triples_path=None, entities_path=None, cache=None, verbose=True):
    """The company KG as a :class:`VadalogKG` (cached next to the CSVs)."""
    triples_path = triples_path or os.path.join(DATA_DIR, "triples.csv")
    data_dir = os.path.dirname(os.path.abspath(triples_path))
    entities_path = entities_path or os.path.join(data_dir, "entities.csv")
    triples = pd.read_csv(triples_path, keep_default_na=False, low_memory=False)
    entities = pd.read_csv(entities_path, keep_default_na=False).entity
    base = base_relations(triples)
    if cache is None:
        cache = os.path.join(data_dir, "vadalog_cache.pkl")
    return VadalogKG(
        PROGRAM, base, SCHEMA, DERIVED, entities,
        constants={ROLE_T: sorted({r for _, _, r in base["role"]})},
        symmetric={"reachable", "connected"}, views={"reachable": "reach"},
        cache=cache, verbose=verbose)


class Reasoner:
    """Drop-in replacement of ``reasoner.Reasoner`` backed by Vadalog."""

    def __init__(self, triples=None, kg=None, **kw):
        if kg is None:
            kg = build_kg(**kw)
        self.kg = kg
        self.stored = {p: {} for p in BASE}
        for p in BASE:
            for row in kg.base[p.lower()]:
                self.stored[p].setdefault(row[0], set()).add(row[1])

    def properties(self, entity):
        return set(self.kg.properties(entity))

    def get(self, entity, prop):
        return self.kg.get(entity, prop)

    def count(self, entity, prop):
        return self.kg.count(entity, prop)

    def ask(self, subject, prop, obj):
        return self.kg.ask(subject, prop, obj)

    def add_rule(self, rule, head=None):
        res = self.kg.add_rule(rule)
        if not res.ok:
            raise ValueError(res.error)
        if head is not None and head != res.head:
            raise ValueError("expected head %s, got %s" % (head, res.head))
        return res.head

    def facts(self, entity=None, prop=None):
        for p in ([prop] if prop else BASE):
            for s in ([entity] if entity else sorted(self.stored[p])):
                for o in sorted(self.stored[p].get(s, ())):
                    yield "<%s><%s><%s>." % (s, p, o)


def load(path=None):
    return Reasoner(kg=build_kg(path))


if __name__ == "__main__":
    t = time.time()
    kg = build_kg(sys.argv[1] if len(sys.argv) > 1 else None)
    print("ready in %.1fs" % (time.time() - t))
    for p in SCHEMA:
        print("%22s %8d facts  %s" % (p, len(kg.rows(kg.views.get(p, p))),
                                      "stored" if p in kg.base else "derived"))
