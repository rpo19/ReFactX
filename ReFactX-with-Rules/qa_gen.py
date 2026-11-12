"""Template questions over the KG (answers are added by add_groundtruth.py).

Six question kinds, each one a stress test for a different capability:
  ask       yes/no on a single fact                "Is it true that X controls Y?"
  count     counting over derived facts            "How many entities control Y?"
  list      enumeration (<=10 answers)             "Which entities control Y?"
  more      comparative on counts                  "...controls more entities than Z?"
  and_not   comparative on two facts               "...controls Y while Z does not?"
  top       superlative over three candidates      "Among X, Y and Z, which one ...?"
"""
import random
import pandas as pd
from reasoner import load

MAX_ANSWERS = 10                # keep list/count answers enumerable
N_PER_COMBO = 30

PHRASE = {  # property -> (verb phrase, plural verb phrase for object-bound questions)
    "OWN": ("owns", "own"),
    "ROLE": ("has a role in", "have a role in"),
    "REACHABLE": ("is directly connected to", "are directly connected to"),
    "CONNECTED": ("is connected to", "are connected to"),
    "QUALIFIED_HOLDINGS": ("has qualified holdings in", "have qualified holdings in"),
    "INFLUENCE": ("has significant influence over", "have significant influence over"),
    "CONTROL": ("controls", "control"),
    "ULTIMATE_CONTROL": ("is the ultimate controller of", "are ultimate controllers of"),
    "INDIRECT_ROLE": ("has a role in an entity that controls",
                      "have a role in an entity that controls"),
}
PLAN = {
    "ask": list(PHRASE),
    "count": [p for p in PHRASE if p != "INDIRECT_ROLE"],
    "list": [p for p in PHRASE if p != "INDIRECT_ROLE"],
    "more": ["OWN", "ROLE", "INFLUENCE", "CONTROL", "CONNECTED"],
    "and_not": ["OWN", "ROLE", "QUALIFIED_HOLDINGS", "CONTROL"],
    "top": ["OWN", "ROLE", "CONTROL"],
}
ANSWER_TYPE = {"ask": "yes_no", "count": "count", "list": "list", "more": "yes_no",
               "and_not": "yes_no", "top": "entity"}


def phrase(kind, prop, x, y, z):
    v, vp = PHRASE[prop]
    return {
        "ask": "Is it true that %s %s %s?" % (x, v, y),
        "count": "How many entities %s %s?" % (vp, y),
        "list": "Which entities %s %s?" % (vp, y),
        "more": "Is it true that %s %s more entities than %s?" % (x, v, z),
        "and_not": "Is it true that %s %s %s while %s does not?" % (x, v, y, z),
        "top": "Among %s, %s and %s, which one %s the most entities?" % (x, y, z, v),
    }[kind]


def _draw(R, kind, prop, subs, objs, rnd, want):
    """Sample the (x, y, z) entities of one question whose answer is `want`
    (yes/no kinds only), or None if the draw failed and has to be retried.

    Return: (x, y, z) where x is subject, y is object, z is another entity (for comparative questions)
    """
    x = rnd.choice(subs)
    if kind == "ask":
        if want:
            return (x, rnd.choice(R.get(x, prop)), "")
        y = rnd.choice(objs)                            # verified: the fact must not hold
        return None if R.ask(x, prop, y) else (x, y, "")


    ## not clear
    if kind in ("count", "list"):
        return ("", rnd.choice(objs), "")

    if kind == "more":
        z = rnd.choice(subs)
        nx, nz = R.count(x, prop), R.count(z, prop)
        if nx == nz:                                    # verified: counts must differ
            return None
        elif want:                                        # x must have more than z
            return (x, "", z) if nx > nz else (z, "", x)
        else:                                           # x must have less than z
            return (x, "", z) if nx < nz else (z, "", x)

    if kind == "and_not":
        y = rnd.choice(R.get(x, prop))
        if want:                                        # z unrelated to y -> "yes"
            z = rnd.choice(subs)
            z = None if R.ask(z, prop, y) else z
        else:                                           # z also related to y -> "no"
            others = [e for e in R.get(y, "is_" + prop) if e != x]
            z = rnd.choice(others) if others else None
        return (x, y, z) if z and z != x else None

    if kind == "top":
        c = [x, rnd.choice(subs), rnd.choice(subs)]
        if len(set(c)) < 3:  # verified: must be three distinct entities
            return None
        n = [R.count(e, prop) for e in c]
        if len(set(n)) < 3:  # verified: counts must differ
            return None
        return tuple(c) #if want else None

        #sorted_c = [e for _, e in sorted(zip(n, c), reverse=True)]
        #return tuple(sorted_c)


def make_qa(R, n_per_combo=N_PER_COMBO, seed=0):
    rnd = random.Random(seed)
    rows, seen, pools = [], set(), {}

    def pool(prop, inverse):
        """
        Entities with at least one result for `prop` in that query direction
            (sorted, for reproducibility; object-bound pools are capped at MAX_ANSWERS).

        """
        if (prop, inverse) not in pools:
            side = R.inv[prop] if inverse else R.out[prop]
            pools[prop, inverse] = sorted(e for e, res in side.items()
                                          if res and (len(res) <= MAX_ANSWERS or not inverse))
        return pools[prop, inverse]

    for kind, props in PLAN.items():
        for prop in props:
            subs = pool(prop, False)                 # entities bound as subject
            objs = pool(prop, True)                  # entities bound as object
            got = 0
            for _ in range(n_per_combo * 50):
                if got >= n_per_combo:
                    break
                d = _draw(R, kind, prop, subs, objs, rnd, got % 2 == 0)
                if d is None:
                    continue
                q = phrase(kind, prop, *d)
                if q in seen:
                    continue
                seen.add(q)
                got += 1
                rows.append({"qid": len(rows), "kind": kind, "property": prop if kind not in ["count", "list"] else "is_" + prop,
                             "answer_type": ANSWER_TYPE[kind], "question": q,
                             "x": d[0], "y": d[1], "z": d[2]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    qa = make_qa(load())
    qa.to_csv("data/qa.csv", index=False)
    print(len(qa), "questions")
    print(qa.groupby(["kind", "property"]).size().to_string())
