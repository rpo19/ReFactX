"""Rule-generation validation harness (Stage 3).

For a candidate Datalog rule the harness reports:

  invalid      -- Nemo rejects the program (parse/validation error); the
                  formatted NemoError message is the LLM feedback signal
  engine_error -- Nemo panics or fails during execution
  valid_wrong  -- the rule loads and executes, but its results differ from
                  the gold relation (missing/extra tuples)
  correct      -- results match gold exactly (numeric tolerance 1e-6)

Usage:  python3 stage3/rule_harness.py
"""
import os
import re
import sys
import time
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nmo_python import NemoEngine, load_string  # noqa: E402
from nemo_reasoner import (DATA_DIR, Reasoner, build_program,  # noqa: E402
                           control_rounds, _unquote)
from tasks import TASKS, compute_gold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# bad rules: which stage rejects them, and with what message.
# (Nemo validation is TWO-STAGE: load_string checks syntax/arity/safety;
#  stratification is only checked when the engine executes.  Nemo is also
#  OPEN-WORLD: undefined predicates are accepted as empty relations -- the
#  closed-world semantic check is OUR layer, Stage 4.)
NEGATIVE = [
    ("syntax error",
     "owner_of_owner(?x, ?z) :- "),
    ("unknown predicate (open world gap)",
     "foo(?x, ?y) :- bar(?x, ?y) ."),
    ("unbound head variable",
     "owner_of_owner(?x, ?z) :- OWN(?x, ?y, ?p) ."),
    ("wrong arity usage",
     "foo(?x) :- OWN(?x, ?y) ."),
    ("aggregate over non-numeric (loose typing)",
     "bad(?y, #max(?r)) :- ROLE(?s, ?y, ?r) ."),
    ("rule without positive literals",
     "is_controlled(?x) :- ~is_controlled(?x) ."),
    ("negation cycle (non-stratified)",
     "bad(?x) :- OWN(?x, ?y, ?p), ~bad(?y) ."),
    ("recursive aggregation",
     "ctl(?x, ?y) :- ctl(?x, ?z), OWN(?z, ?y, ?p), ?p > 50 .\n"
     "ctl(?x, ?y) :- OWN(?x, ?y, ?p), ?p > 50 .\n"
     "ds(?x, ?y, #sum(?p, ?z)) :- ctl(?x, ?z), OWN(?z, ?y, ?p) .\n"
     "ctl2(?x, ?y) :- ds(?x, ?y, ?s), ?s > 50 .\n"
     "ctl(?x, ?y) :- ctl2(?x, ?y) ."),
]

# plausible but logically wrong rules (valid Nemo, wrong results)
NEAR_MISS = [
    ("owner_of_owner", "too strong (p > 50 on the first hop)",
     "owner_of_owner(?x, ?z) :- OWN(?x, ?y, ?p), OWN(?y, ?z, ?q), ?p > 50 ."),
    ("owner_count", "counts distinct percentages, not owners",
     "owner_count(?y, #count(?p)) :- OWN(?x, ?y, ?p) ."),
    ("top_stake", "min instead of max",
     "top_stake(?y, #min(?p)) :- OWN(?x, ?y, ?p) ."),
    ("total_owned", "distinct values only (equal stakes collapse)",
     "total_owned(?x, #sum(?p)) :- OWN(?x, ?y, ?p) ."),
]


def _norm(v):
    """Normalize a value for set comparison: numerics round to 6 decimals
    (our stakes are integer percentages, so rounding is exact), strings stay."""
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return v


def _rows(engine, pred):
    return {tuple(_norm(_unquote(v)) for v in row)
            for row in engine.result(pred)}


def _is_panic(error):
    return (type(error).__module__, type(error).__name__) == (
        "pyo3_runtime", "PanicException")


class RuleBench:
    def __init__(self, triples=None):
        if triples is None:
            triples = pd.read_csv(f"{DATA_DIR}/triples.csv", keep_default_na=False,
                                  low_memory=False)
        self._data = TemporaryDirectory(prefix="refactx-bench-")
        Reasoner._build_csvs(triples, self._data.name)
        self.depth = control_rounds(triples) + 3
        self.base = build_program(self.depth, self._data.name, include_outputs=False)

    def program(self, rule, context=(), head=None):
        out = "\n@output %s ." % head if head else ""
        return self.base + "\n".join(context) + "\n" + rule + "\n" + out

    def stages(self, rule, head=None):
        """Which stage rejects a rule: (stage, message) with stage in
        {'validation', 'execution', None(=accepted)}."""
        result = self.check(rule, head)
        return result.get("failure_stage"), result.get("error")

    def check(self, rule, head, context=(), gold=None):
        """Full pipeline: validate, execute, compare against gold.

        Note: Nemo computes only predicates reachable from @output / @export
        directives (dead-code elimination), so the candidate's head must be
        declared with @output or `result()` silently yields nothing."""
        t0 = time.perf_counter()
        stage = "validation"
        try:
            program = load_string(self.program(rule, context, head))
            stage = "execution"
            engine = NemoEngine(program)
            engine.reason()
            got = _rows(engine, head) if head else set()
        except BaseException as error:
            panic = _is_panic(error)
            if not panic and not isinstance(error, Exception):
                raise
            return {"outcome": "engine_error" if panic or stage == "execution" else "invalid",
                    "failure_stage": stage, "error": str(error),
                    "nemo_s": time.perf_counter() - t0}
        result = {"nemo_s": time.perf_counter() - t0, "rows": len(got)}
        if gold is None:
            return dict(result, outcome="valid")
        gold = {tuple(_norm(v) for v in r) for r in gold}
        missing = gold - got
        extra = got - gold
        if not missing and not extra:
            return dict(result, outcome="correct", missing=0, extra=0)
        return dict(result, outcome="valid_wrong", missing=len(missing), extra=len(extra))


def main():
    t0 = time.time()
    bench = RuleBench()
    print("base program ready (depth %d) in %.1fs\n" %
          (bench.depth, time.time() - t0))

    print("=" * 78)
    print("1. gold rules -> must be CORRECT")
    print("=" * 78)
    golds = compute_gold(write=False)
    for t in TASKS:
        r = bench.check(t["gold_rule"], t["head"], t["context"],
                        golds[t["id"]])
        print("   %-16s %s %s" % (t["id"], r["outcome"],
                                  r.get("error", "")[:80]))
        assert r["outcome"] == "correct", t["id"]

    print("\n" + "=" * 78)
    print("2. near-miss rules -> valid Nemo but WRONG results")
    print("=" * 78)
    for task_id, why, rule in NEAR_MISS:
        t = next(x for x in TASKS if x["id"] == task_id)
        r = bench.check(rule, t["head"], t["context"], golds[task_id])
        print("   %-16s %-40s %-11s missing=%s extra=%s" %
              (task_id, why, r["outcome"], r.get("missing", "-"),
               r.get("extra", "-")))

    print("\n" + "=" * 78)
    print("3. bad rules -> which stage rejects them, and how")
    print("=" * 78)
    for why, rule in NEGATIVE:
        head = re.search(r"^\s*([\w]+)\s*\(", rule)
        head = head.group(1) if head else None
        stage, err = bench.stages(rule, head)
        if stage is None:
            print("   %-44s ACCEPTED (semantic-layer gap: Nemo does not check this)" % why)
        else:
            print("   %-44s REJECTED at %s" % (why, stage))
            print("      %s" % (err or "").replace("\n", "\n      ")[:250])

    print("\n" + "=" * 78)
    print("4. task catalog (LLM prompt feed)")
    print("=" * 78)
    for t in TASKS:
        print("   [%s] %s" % (t["id"], t["instruction"]))
        print("        gold: %s" % t["gold_rule"])


if __name__ == "__main__":
    main()
