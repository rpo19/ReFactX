"""Nemo-backed reasoner implementing the exact interface of reasoner.py.

Same class shape (`Reasoner(triples)`), same attributes (`pct`, `out`, `stored`,
`inv`), same methods (`properties`, `get`, `count`, `ask`, `facts`), same
constants (BASE / SYMMETRIC / RULES) -- so api.py, qa_gen.py,
add_groundtruth.py and test.py run unchanged (see stage2/run_with_nemo.py).

Internally the 7 derived relations are computed by the Nemo rule engine
(nemo-python 0.10.1) instead of the Python mock:

  - OWN / ROLE / REACHABLE are loaded via @import CSV directives.
  - QUALIFIED_HOLDINGS, INFLUENCE, CONNECTED, ULTIMATE_CONTROL,
    INDIRECT_ROLE are ordinary Nemo rules (guards, stratified negation).
  - CONTROL needs recursive aggregation, which Nemo does NOT support
    ("rules of the program are not stratified").  Its fixpoint converges in
    a bounded number of rounds on this KG, so it is UNROLLED into `depth`
    layers of non-recursive #sum -- an exact emulation (verified
    tuple-by-tuple against the mock in stage1/smoke_test.py).

The unroll depth is measured automatically from the data: a small Python
pre-pass counts the fixpoint rounds of the (mock-equivalent) CONTROL rule;
Nemo then recomputes the relation itself.
"""
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from threading import Lock

import pandas as pd
from nmo_python import NemoEngine, load_string

from nemo_rule import body_has_aggregate, rule_parts

QUALIFIED_MIN, INFLUENCE_MIN, CONTROL_MIN = 10, 20, 50
BASE = ["OWN", "ROLE", "REACHABLE"]
SYMMETRIC = {"REACHABLE", "CONNECTED"}          # no `is_` form: both directions agree
RULES = ["QUALIFIED_HOLDINGS", "INFLUENCE", "CONTROL", "ULTIMATE_CONTROL",
         "CONNECTED", "INDIRECT_ROLE"]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

CSV_IMPORTS = """
@import OWN :- csv {{ resource="{data}/own.csv", format=(string,string,double) }}.
@import ROLE :- csv {{ resource="{data}/role.csv", format=(string,string,string) }}.
@import REACHABLE :- csv {{ resource="{data}/reachable.csv", format=(string,string) }}.
"""

BASE_RULES = """
QUALIFIED_HOLDINGS(?x, ?y) :- OWN(?x, ?y, ?p), ?p >= 10 .
INFLUENCE(?x, ?y) :- OWN(?x, ?y, ?p), ?p >= 20 .
INFLUENCE(?x, ?y) :- ROLE(?x, ?y, ?r) .

reach(?x, ?y) :- REACHABLE(?x, ?y) .
reach(?x, ?y) :- REACHABLE(?y, ?x) .
CONNECTED(?x, ?y) :- reach(?x, ?y), ?x != ?y .
CONNECTED(?x, ?y) :- CONNECTED(?x, ?z), reach(?z, ?y), ?x != ?y .

self(?x) :- OWN(?x, ?y, ?p) .
"""

OUTPUTS = """
@output OWN .
@output ROLE .
@output REACHABLE .
@output QUALIFIED_HOLDINGS .
@output INFLUENCE .
@output CONNECTED .
@output CONTROL .
@output ULTIMATE_CONTROL .
@output INDIRECT_ROLE .
"""


def control_unroll_rules(depth):
    """Unrolled CONTROL fixpoint (exact emulation of the mock, see module doc)."""
    rules = ["acc_0(?x, ?x) :- self(?x) ."]
    for n in range(1, depth + 1):
        prev, cur = f"acc_{n-1}", f"acc_{n}"
        rules.append(f"direct_sum_{n}(?x, ?y, #sum(?p, ?z)) :- {prev}(?x, ?z), OWN(?z, ?y, ?p) .")
        rules.append(f"new_{n}(?x, ?y) :- direct_sum_{n}(?x, ?y, ?s), ?s > 50, ?x != ?y .")
        rules.append(f"{cur}(?x, ?z) :- {prev}(?x, ?z) .")
        rules.append(f"{cur}(?x, ?y) :- new_{n}(?x, ?y) .")
        rules.append(f"CONTROL(?x, ?y) :- new_{n}(?x, ?y) .")
    rules.append("is_controlled(?x) :- CONTROL(?z, ?x) .")
    rules.append("ULTIMATE_CONTROL(?x, ?y) :- CONTROL(?x, ?y), ~is_controlled(?x) .")
    rules.append("INDIRECT_ROLE(?x, ?y) :- ROLE(?x, ?z, ?r), CONTROL(?z, ?y) .")
    return "\n".join(rules) + "\n"


def build_program(depth, data_dir, *, include_outputs=True):
    imports = CSV_IMPORTS.format(data=data_dir)
    outputs = OUTPUTS if include_outputs else ""
    return imports + BASE_RULES + control_unroll_rules(depth) + outputs


def control_rounds(triples):
    """Rounds needed by the CONTROL fixpoint on this data (mock-equivalent
    pre-pass; used only to size the unrolling -- CONTROL itself is computed
    by Nemo)."""
    own, pct = defaultdict(set), {}
    for s, p, o, v in triples.itertuples(index=False):
        if p == "OWN":
            own[s].add(o)
            pct[(s, o)] = float(v)
    max_rounds = 0
    for x in list(own):
        acc, rounds = {x}, 0
        while True:
            rounds += 1
            tot = defaultdict(float)
            for z in acc:
                for y in own.get(z, ()):
                    tot[y] += pct[(z, y)]
            new = {y for y, s in tot.items() if s > CONTROL_MIN} - acc
            if not new:
                break
            acc |= new
        max_rounds = max(max_rounds, rounds)
    return max_rounds


def _unquote(v):
    if isinstance(v, str) and len(v) >= 2 and v[0] == v[-1] == '"':
        return v[1:-1]
    return v


@dataclass(frozen=True)
class _State:
    program: str
    engine: NemoEngine
    heads: tuple
    reason_time: float
    out: dict
    pct: dict
    stored: dict
    inv: dict


class Reasoner:
    def __init__(self, triples, unroll_margin=3):
        self._data = TemporaryDirectory(prefix="refactx-reasoner-")
        self._update_lock = Lock()
        self._build_csvs(triples, self._data.name)
        depth = control_rounds(triples) + unroll_margin
        program = build_program(depth, self._data.name)

        t0 = time.perf_counter()
        engine = NemoEngine(load_string(program))
        self._load_time = time.perf_counter() - t0
        self._state = self._reason(program, engine, (), 0.0)

    @property
    def engine(self):
        return self._state.engine

    @property
    def out(self):
        return self._state.out

    @property
    def pct(self):
        return self._state.pct

    @property
    def stored(self):
        return self._state.stored

    @property
    def inv(self):
        return self._state.inv

    # ------------------------------------------------------------- data load
    @staticmethod
    def _build_csvs(triples, data_dir):
        own = triples[triples.predicate == "OWN"]
        own[["subject", "object", "value"]].assign(
            value=own.value.astype(float)
        ).to_csv(f"{data_dir}/own.csv", index=False, header=False)
        triples[triples.predicate == "ROLE"][["subject", "object", "value"]].to_csv(
            f"{data_dir}/role.csv", index=False, header=False)
        triples[triples.predicate == "REACHABLE"][["subject", "object"]].to_csv(
            f"{data_dir}/reachable.csv", index=False, header=False)

    # ------------------------------------------------ materialize Nemo results
    @staticmethod
    def _reason(program, engine, heads, previous_reason_time):
        """Reason and build the complete API state before publishing it."""
        t0 = time.perf_counter()
        engine.reason()
        reason_time = previous_reason_time + time.perf_counter() - t0

        def rows(pred):
            return (tuple(_unquote(v) for v in row) for row in engine.result(pred))

        out = {p: defaultdict(set) for p in BASE + RULES}
        pct = {}
        for s, o, v in rows("OWN"):
            out["OWN"][s].add(o)
            pct[(s, o)] = float(v)
        for s, o, _ in rows("ROLE"):
            out["ROLE"][s].add(o)
        for s, o in rows("REACHABLE"):
            out["REACHABLE"][s].add(o)
        stored = {p: defaultdict(set, {k: set(v) for k, v in out[p].items()})
                  for p in BASE}  # snapshot before the symmetric closure
        for s, o in rows("reach"):
            out["REACHABLE"][s].add(o)
        for pred in (*RULES, *heads):
            relation = out.setdefault(pred.upper(), defaultdict(set))
            for s, o in rows(pred):
                relation[s].add(o)

        inv = {p: defaultdict(set) for p in out}
        for p, d in out.items():
            for s, objs in d.items():
                for o in objs:
                    inv[p][o].add(s)
        return _State(program, engine, heads, reason_time, out, pct, stored, inv)

    # ------------------------------------------------------------- reasoner API
    def add_rule(self, rule, head=None):
        """Register a derived relation at runtime: append the rule to the
        program, re-reason with Nemo, and expose the new head predicate
        through properties()/get/count/ask (2-ary projection)."""
        predicate, arity, body = rule_parts(rule)
        if head is not None and head != predicate:
            raise ValueError(f"expected head {head}, got {predicate}")
        if arity != 2:
            raise ValueError("registered rules must define a binary relation")
        if any(body_has_aggregate(literal) for literal in body):
            raise ValueError("aggregates must appear in the head")
        head = predicate
        with self._update_lock:
            current = self._state
            heads = current.heads
            program = current.program + "\n" + rule + "\n"
            if head not in (*BASE, *RULES, *heads):
                if head.upper() in current.out or head.lower().startswith("is_"):
                    raise ValueError(f"head conflicts with an existing property name: {head}")
                program += f"@output {head} .\n"
                heads += (head,)
            engine = NemoEngine(load_string(program))
            self._state = self._reason(program, engine, heads, current.reason_time)
        return head

    def properties(self, entity):
        """The property strings `get`/`count`/`ask` accept for `entity`."""
        state = self._state
        names = set()
        for p in state.out:
            if state.out[p].get(entity):
                names.add(p.lower())
            if state.inv[p].get(entity) and p not in SYMMETRIC:
                names.add("is_" + p.lower())
        return names

    @staticmethod
    def _resolve(prop, out):
        """'is_own' -> ('OWN', True), 'own'/'OWN' -> ('OWN', False)."""
        inverse = prop.lower().startswith("is_")
        p = (prop[3:] if inverse else prop).upper()
        if p not in out:
            raise KeyError("unknown property %r" % prop)
        return p, inverse

    def get(self, entity, prop):
        """Answers for one of the property strings returned by `properties`."""
        state = self._state
        p, inverse = self._resolve(prop, state.out)
        return sorted((state.inv if inverse else state.out)[p].get(entity, ()))

    def count(self, entity, prop):
        state = self._state
        p, inverse = self._resolve(prop, state.out)
        return len((state.inv if inverse else state.out)[p].get(entity, ()))

    def ask(self, subject, prop, obj):
        out = self.out
        p, inverse = self._resolve(prop, out)
        if inverse:
            subject, obj = obj, subject
        return obj in out[p].get(subject, ())

    def facts(self, entity=None, prop=None):
        """Verbalized triples '<subject><PROPERTY><object>.' for the prefix-tree
        index. Exactly the stored triples: no inverse facts, and not even the
        symmetric closure of REACHABLE."""
        stored = self.stored
        for p in ([prop] if prop else BASE):
            for s in ([entity] if entity else sorted(stored[p])):
                for o in sorted(stored[p].get(s, ())):
                    yield "<%s><%s><%s>." % (s, p, o)


def load(path="data/triples.csv"):
    return Reasoner(pd.read_csv(path, keep_default_na=False, low_memory=False))


if __name__ == "__main__":
    t = time.time()
    R = load()
    print("built in %.1fs (load %.1fs, reason %.1fs)" %
          (time.time() - t, R._load_time, R._state.reason_time))
    for p in BASE + RULES:
        print("%22s %8d facts %s" % (p, sum(len(v) for v in R.out[p].values()),
                                     "derived" if p in RULES else "base"))
    n = 0
    with open(f"{DATA_DIR}/facts.txt", "w") as f:
        for n, fact in enumerate(R.facts(), 1):
            f.write(fact + "\n")
    print("%d verbalized facts ->  data/facts.txt" % n)

    with open(f"{DATA_DIR}/all_facts_and_derived.txt", "w") as f:
        for p in BASE + RULES:
            for s, objs in R.out[p].items():
                for o in objs:
                    f.write("<%s><%s><%s>.\n" % (s, p, o))
