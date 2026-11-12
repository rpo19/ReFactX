"""Minimal reasoner simulating the Vadalog API used by ReFactX-with-rules.

Base relations come from the KG; the following rules are derived (materialized
once here, evaluated on demand by Vadalog in the real system):

  QUALIFIED_HOLDINGS(X,Y) :- OWN(X,Y,P), P >= 10.
  INFLUENCE(X,Y)          :- OWN(X,Y,P), P >= 20.
  INFLUENCE(X,Y)          :- ROLE(X,Y,_).
  CONTROL(X,Y)            :- sum(P) over {OWN(Z,Y,P) | Z=X or CONTROL(X,Z)} > 50.
  ULTIMATE_CONTROL(X,Y)   :- CONTROL(X,Y), not CONTROL(_,X).
  REACHABLE(X,Y)          :- REACHABLE(Y,X).
  CONNECTED(X,Y)          :- REACHABLE(X,Y).
  CONNECTED(X,Y)          :- CONNECTED(X,Z), REACHABLE(Z,Y).
  INDIRECT_ROLE(X,Y)      :- ROLE(X,Z), CONTROL(Z,Y).

There are no inverse relations and no inverse triples: the reverse direction is
a query, i.e. the entity is bound in the object instead of the subject position.
`properties(e)` returns the property strings that answer for `e`, the direction encoded
in the name: `own` asks OWN(e, ?) and `is_own` asks OWN(?, e).
"""
from collections import defaultdict
import pandas as pd

QUALIFIED_MIN, INFLUENCE_MIN, CONTROL_MIN = 10, 20, 50
BASE = ["OWN", "ROLE", "REACHABLE"]
SYMMETRIC = {"REACHABLE", "CONNECTED"}          # no `is_` form: both directions agree
RULES = ["QUALIFIED_HOLDINGS", "INFLUENCE", "CONTROL", "ULTIMATE_CONTROL",
         "CONNECTED", "INDIRECT_ROLE"]


class Reasoner:
    def __init__(self, triples):
        self.pct = {}                                              # (owner, target) -> %
        self.out = {p: defaultdict(set) for p in BASE + RULES}     # property -> subj -> objs
        for s, p, o, v in triples.itertuples(index=False):
            self.out[p][s].add(o)
            if p == "OWN":
                self.pct[(s, o)] = float(v)
        self.stored = {p: self.out[p] for p in BASE}   # the triples, before any rule
        self._derive()
        self.inv = {p: defaultdict(set) for p in self.out}         # object-bound queries
        for p, d in self.out.items():
            for s, objs in d.items():
                for o in objs:
                    self.inv[p][o].add(s)

    # ------------------------------------------------------------------ rules
    def _derive(self):
        own, role, ctrl = self.out["OWN"], self.out["ROLE"], self.out["CONTROL"]

        for (x, y), p in self.pct.items():
            if p >= QUALIFIED_MIN:
                self.out["QUALIFIED_HOLDINGS"][x].add(y)
            if p >= INFLUENCE_MIN:
                self.out["INFLUENCE"][x].add(y)
        for x, ys in role.items():
            self.out["INFLUENCE"][x] |= ys

        for x in list(own):                     # per-candidate fixpoint of the >50% rule
            acc = {x}
            while True:
                tot = defaultdict(float)
                for z in acc:
                    for y in own.get(z, ()):
                        tot[y] += self.pct[(z, y)]
                new = {y for y, p in tot.items() if p > CONTROL_MIN} - acc
                if not new:
                    break
                acc |= new
            if len(acc) > 1:
                ctrl[x] = acc - {x}

        controlled = {y for ys in ctrl.values() for y in ys}
        for x, ys in ctrl.items():
            if x not in controlled:
                self.out["ULTIMATE_CONTROL"][x] = set(ys)

        for x, zs in role.items():
            ys = {y for z in zs for y in ctrl.get(z, ())}
            if ys:
                self.out["INDIRECT_ROLE"][x] = ys

        reach = defaultdict(set)                # REACHABLE is symmetric
        for x, ys in self.out["REACHABLE"].items():
            for y in ys:
                reach[x].add(y)
                reach[y].add(x)
        self.out["REACHABLE"] = reach
        seen = set()
        for start in reach:                     # CONNECTED = connected component
            if start in seen:
                continue
            comp, stack = set(), [start]
            while stack:
                v = stack.pop()
                if v not in comp:
                    comp.add(v)
                    stack.extend(reach[v] - comp)
            seen |= comp
            for v in comp:
                self.out["CONNECTED"][v] = comp - {v}

    # -------------------------------------------------------------------- API
    def properties(self, entity):
        """The property strings `get`/`count`/`ask` accept for `entity`: the relation
        name binds it as subject (`own` -> OWN(entity, ?)) and the `is_` prefix binds it
        as object (`is_own` -> OWN(?, entity)).  Symmetric relations have no `is_` form."""
        names = set()
        for p in self.out:
            if self.out[p].get(entity):
                names.add(p.lower())
            if self.inv[p].get(entity) and p not in SYMMETRIC:
                names.add("is_" + p.lower())
        return names

    def _resolve(self, prop):
        """'is_own' -> ('OWN', True), 'own'/'OWN' -> ('OWN', False)."""
        inverse = prop.lower().startswith("is_")
        p = (prop[3:] if inverse else prop).upper()
        if p not in self.out:
            raise KeyError("unknown property %r" % prop)
        return p, inverse

    def get(self, entity, prop):
        """Answers for one of the property strings returned by `properties`."""
        p, inverse = self._resolve(prop)
        return sorted((self.inv if inverse else self.out)[p].get(entity, ()))

    def count(self, entity, prop):
        p, inverse = self._resolve(prop)
        return len((self.inv if inverse else self.out)[p].get(entity, ()))

    def ask(self, subject, prop, obj):
        p, inverse = self._resolve(prop)
        if inverse:
            subject, obj = obj, subject
        return obj in self.out[p].get(subject, ())

    def facts(self, entity=None, prop=None):
        """Verbalized triples '<subject><PROPERTY><object>.' for the prefix-tree index.

        Exactly the stored triples: no inverse facts, and not even the symmetric
        closure of REACHABLE -- the reasoner answers those directions by rule."""
        for p in ([prop] if prop else BASE):
            for s in ([entity] if entity else sorted(self.stored[p])):
                for o in sorted(self.stored[p].get(s, ())):
                    yield "<%s><%s><%s>." % (s, p, o)


def load(path="data/triples.csv"):
    return Reasoner(pd.read_csv(path, keep_default_na=False, low_memory=False))


if __name__ == "__main__":
    import time
    t = time.time()
    R = load()
    print("built in %.1fs" % (time.time() - t))
    for p in BASE + RULES:
        print("%22s %8d facts %s" % (p, sum(len(v) for v in R.out[p].values()),
                                     "derived" if p in RULES else "base"))
    n = 0
    with open("data/facts.txt", "w") as f:
        for n, fact in enumerate(R.facts(), 1):
            f.write(fact + "\n")
    print("%d verbalized facts ->  data/facts.txt" % n)

    ## verbalize all extended reasoning triples
    with open("data/all_facts_and_derived.txt", "w") as f:
        for p in BASE + RULES:
            for s, objs in R.out[p].items():
                for o in objs:
                    f.write("<%s><%s><%s>.\n" % (s, p, o))