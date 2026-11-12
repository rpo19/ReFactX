"""Stage 1 smoke test: load the synthetic KG into Nemo, derive the 7 relations,
and compare them exactly against the Python mock (reasoner.py).

CONTROL strategy: Nemo (v0.10.1) does NOT support recursive aggregation
("rules of the program are not stratified"). The mock's CONTROL is a
per-candidate fixpoint of summed direct stakes > 50 that converges in <= 16
rounds on this KG (measured). We therefore UNROLL the fixpoint into L=20
layers, each layer using a NON-recursive #sum aggregate over the previous
layer's accumulated set -- an exact emulation for this KG.

  acc_0(x,x) :- self(x).
  direct_sum_n(x, y, #sum(p, z)) :- acc_{n-1}(x, z), own(z, y, p).
  new_n(x, y) :- direct_sum_n(x, y, s), s > 50, x != y.
  acc_n(x, z) :- acc_{n-1}(x, z).  acc_n(x, y) :- new_n(x, y).
  control(x, y) :- new_n(x, y).   (for each n)
"""
import os
import sys
import time
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nmo_python import NemoEngine, load_string  # noqa: E402

from reasoner import load as load_mock  # noqa: E402

DATA = "data"
L = 20  # unroll depth (mock converges in <= 16 rounds on this KG)

CSV_IMPORTS = """
@import OWN :- csv { resource="data/own.csv", format=(string,string,double) }.
@import ROLE :- csv { resource="data/role.csv", format=(string,string,string) }.
@import REACHABLE :- csv { resource="data/reachable.csv", format=(string,string) }.
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


def control_rules(L):
    """Unrolled CONTROL fixpoint emulating the mock exactly."""
    rules = []
    rules.append("acc_0(?x, ?x) :- self(?x) .")
    for n in range(1, L + 1):
        prev = f"acc_{n-1}"
        cur = f"acc_{n}"
        rules.append(
            f"direct_sum_{n}(?x, ?y, #sum(?p, ?z)) :- {prev}(?x, ?z), OWN(?z, ?y, ?p) ."
        )
        rules.append(
            f"new_{n}(?x, ?y) :- direct_sum_{n}(?x, ?y, ?s), ?s > 50, ?x != ?y ."
        )
        rules.append(f"{cur}(?x, ?z) :- {prev}(?x, ?z) .")
        rules.append(f"{cur}(?x, ?y) :- new_{n}(?x, ?y) .")
        rules.append(f"CONTROL(?x, ?y) :- new_{n}(?x, ?y) .")
    rules.append("is_controlled(?x) :- CONTROL(?z, ?x) .")
    rules.append("ULTIMATE_CONTROL(?x, ?y) :- CONTROL(?x, ?y), ~is_controlled(?x) .")
    rules.append("INDIRECT_ROLE(?x, ?y) :- ROLE(?x, ?z, ?r), CONTROL(?z, ?y) .")
    return "\n".join(rules) + "\n"


def make_csvs():
    """Split triples.csv into per-relation CSVs for the @import directives."""
    triples = pd.read_csv(
        f"{DATA}/triples.csv", keep_default_na=False, low_memory=False
    )
    for pred, cols, types in [
        ("own", ["subject", "object", "value"], {"value": float}),
        ("role", ["subject", "object", "value"], {}),
        ("reachable", ["subject", "object"], {}),
    ]:
        sub = triples[triples.predicate == pred.upper()][cols]
        for c, t in types.items():
            sub[c] = sub[c].astype(t)
        sub.to_csv(f"{DATA}/{pred}.csv", index=False, header=False)
    return triples


def result_sets(engine, predicates):
    """{predicate: set of tuples} with quotes stripped from string values."""
    out = {}
    for p in predicates:
        s = set()
        for row in engine.result(p):
            vals = []
            for v in row:
                if isinstance(v, str) and len(v) >= 2 and v[0] == v[-1] == '"':
                    v = v[1:-1]
                vals.append(v)
            s.add(tuple(vals))
        out[p] = s
    return out


def mock_sets(R, predicates):
    out = {}
    for p in predicates:
        s = set()
        d = R.out[p]
        for subj, objs in d.items():
            for o in objs:
                s.add((subj, o))
        out[p] = s
    return out


def main():
    print("1. generating per-relation CSVs ...")
    make_csvs()

    program = CSV_IMPORTS + BASE_RULES + control_rules(L) + OUTPUTS
    print(f"2. program: {program.count(chr(10))} lines, "
          f"unrolled CONTROL x{L}")

    t0 = time.time()
    engine = NemoEngine(load_string(program))
    print(f"   load (parse+validate): {time.time() - t0:.1f}s")
    t0 = time.time()
    engine.reason()
    print(f"   reason: {time.time() - t0:.1f}s")

    print("3. loading mock ...")
    R = load_mock()

    print("4. comparing derived relations (nemo vs mock) ...")
    all_ok = True
    for p in [
        "QUALIFIED_HOLDINGS", "INFLUENCE", "CONNECTED",
        "CONTROL", "ULTIMATE_CONTROL", "INDIRECT_ROLE",
    ]:
        nemo_set = result_sets(engine, [p])[p]
        mock_set = mock_sets(R, [p])[p]
        missing = mock_set - nemo_set
        extra = nemo_set - mock_set
        status = "OK" if not missing and not extra else "MISMATCH"
        if status != "OK":
            all_ok = False
        print(f"   {p:22s} mock={len(mock_set):8d} nemo={len(nemo_set):8d} "
              f"missing={len(missing)} extra={len(extra)}  {status}")
        if missing:
            print("     missing examples:", sorted(missing)[:3])
        if extra:
            print("     extra examples:", sorted(extra)[:3])

    # symmetric REACHABLE parity (mock serves symmetric closure as REACHABLE)
    reach_nemo = result_sets(engine, ["reach"])["reach"]
    reach_mock = mock_sets(R, ["REACHABLE"])["REACHABLE"]
    diff = reach_mock ^ reach_nemo
    all_ok = all_ok and not diff
    print(f"   {'reach (symmetric)':22s} mock={len(reach_mock):8d} "
          f"nemo={len(reach_nemo):8d} diff={len(diff)}  "
          f"{'OK' if not diff else 'MISMATCH'}")

    print("\nRESULT:", "ALL RELATIONS MATCH" if all_ok else "MISMATCHES FOUND")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
