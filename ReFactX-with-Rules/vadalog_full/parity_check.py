"""Check the Vadalog reasoner against the Python reference (reasoner.py).

    python3 vadalog_full/parity_check.py [path/to/data]

1. every derived relation, tuple by tuple;
2. properties(e) / get(e, p) for a sample of entities, both directions;
3. the 1,110 reference QA answers (data/qa_answers.csv).
"""
import os
import random
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import reasoner as mock  # noqa: E402
import vadalog_reasoner as vr  # noqa: E402


def main(data):
    t = time.time()
    triples = pd.read_csv(os.path.join(data, "triples.csv"), keep_default_na=False, low_memory=False)
    ref = mock.Reasoner(triples)
    print("reference built in %.1fs" % (time.time() - t))
    R = vr.Reasoner(kg=vr.build_kg(os.path.join(data, "triples.csv")))
    kg, ok = R.kg, True

    print("\n1. derived relations")
    for p in vr.RULES:
        a = {(s, o) for s, os_ in ref.out[p].items() for o in os_}
        b = {(r[0], r[1]) for r in kg.rows(p.lower())}
        same = a == b
        ok &= same
        print("   %-20s %8d  %s" % (p, len(b), "match" if same else "DIFF %d/%d" % (len(a - b), len(b - a))))

    print("\n2. properties / get on 2000 random entities")
    ents = kg.entities()
    random.Random(0).shuffle(ents)
    bad = 0
    for e in ents[:2000]:
        if ref.properties(e) != R.properties(e):
            bad += 1
            continue
        for p in ref.properties(e):
            bad += ref.get(e, p) != R.get(e, p)
    ok &= bad == 0
    print("   mismatches:", bad)

    print("\n3. reference QA answers")
    qa = pd.read_csv(os.path.join(data, "qa_answers.csv"), keep_default_na=False)
    from add_groundtruth import answer  # the same dispatch that built the references
    wrong = 0
    for _, q in qa.iterrows():
        wrong += answer(R, q) != str(q["answer"])
    ok &= wrong == 0
    print("   %d/%d answers match" % (len(qa) - wrong, len(qa)))
    print("\nALL CHECKS PASS" if ok else "\nMISMATCHES FOUND")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1] if len(sys.argv) > 1 else vr.DATA_DIR) else 1)
