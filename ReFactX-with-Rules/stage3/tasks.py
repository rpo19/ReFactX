"""Rule-generation benchmark tasks: missing relations the LLM must write.

Each task defines a relation that is NOT in the base program, a
natural-language instruction (what the LLM is asked for), the gold rule
(what a correct answer looks like), and gold results computed independently
from the base data (pandas), so correctness is never circular.

Task set (all over the company-ownership KG):
  owner_of_owner  2-hop join (the grandparent analog)
  owner_count     #count aggregate
  top_stake       #max aggregate
  total_owned     #sum aggregate
  majority_owner  #max + join, built on top of top_stake (session/context rule)
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
GOLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold")

# schema the LLM is told about (Stage 4 prompt feed)
SCHEMA = {
    "OWN": 3, "ROLE": 3, "REACHABLE": 2,
    "QUALIFIED_HOLDINGS": 2, "INFLUENCE": 2, "CONNECTED": 2,
    "CONTROL": 2, "ULTIMATE_CONTROL": 2, "INDIRECT_ROLE": 2,
}

TASKS = [
    {
        "id": "owner_of_owner",
        "instruction": ("Write a Datalog rule defining the relation "
                        "owner_of_owner(x, z): x directly owns some company "
                        "y that directly owns z."),
        "head": "owner_of_owner",
        "gold_rule": "owner_of_owner(?x, ?z) :- OWN(?x, ?y, ?p), OWN(?y, ?z, ?q) .",
        "context": [],          # previously accepted rules (session rules)
    },
    {
        "id": "owner_count",
        "instruction": ("Write a Datalog rule defining the relation "
                        "owner_count(y, n): n is the number of distinct "
                        "entities that own company y. Use the #count aggregate."),
        "head": "owner_count",
        "gold_rule": "owner_count(?y, #count(?x)) :- OWN(?x, ?y, ?p) .",
        "context": [],
    },
    {
        "id": "top_stake",
        "instruction": ("Write a Datalog rule defining the relation "
                        "top_stake(y, m): m is the largest ownership "
                        "percentage held in company y. Use the #max aggregate."),
        "head": "top_stake",
        "gold_rule": "top_stake(?y, #max(?p)) :- OWN(?x, ?y, ?p) .",
        "context": [],
    },
    {
        "id": "total_owned",
        "instruction": ("Write a Datalog rule defining the relation "
                        "total_owned(x, s): s is the sum of all the "
                        "ownership percentages that x holds. Use the #sum "
                        "aggregate over the target company as well as the "
                        "percentage, so that equal percentages held in "
                        "different companies are all added."),
        "head": "total_owned",
        "gold_rule": "total_owned(?x, #sum(?p, ?y)) :- OWN(?x, ?y, ?p) .",
        "context": [],
    },
    {
        "id": "majority_owner",
        "instruction": ("Write a Datalog rule defining the relation "
                        "majority_owner(y, x): x holds the largest ownership "
                        "stake in company y, including all tied holders. "
                        "There is no minimum stake threshold. You may use "
                        "the previously defined relation top_stake(y, m)."),
        "head": "majority_owner",
        "gold_rule": ("majority_owner(?y, ?x) :- top_stake(?y, ?m), "
                      "OWN(?x, ?y, ?m) ."),
        "context": ["top_stake(?y, #max(?p)) :- OWN(?x, ?y, ?p) ."],
    },
]


def _own(t=None):
    if t is None:
        t = pd.read_csv(f"{DATA}/triples.csv", keep_default_na=False,
                        low_memory=False)
    own = t[t.predicate == "OWN"][["subject", "object", "value"]].copy()
    own["value"] = own.value.astype(float)
    return own


def compute_gold(triples=None, *, write=True):
    """Compute independent gold results, optionally exporting stage3/gold/."""
    own = _own(triples)

    oo = own.merge(own, left_on="object", right_on="subject")
    golds = {
        "owner_of_owner": set(zip(oo.subject_x, oo.object_y)),
        "owner_count": {(y, n) for y, n in
                        own.groupby("object").subject.nunique().items()},
        "top_stake": {(y, m) for y, m in
                      own.groupby("object").value.max().items()},
        "total_owned": {(x, s) for x, s in
                        own.groupby("subject").value.sum().items()},
    }
    top = own.groupby("object").value.max()
    maj = own[own.value == own.object.map(top)]
    golds["majority_owner"] = set(zip(maj.object, maj.subject))

    if write:
        os.makedirs(GOLD_DIR, exist_ok=True)
        for task_id, rows in golds.items():
            pd.DataFrame(sorted(rows), columns=["a", "b"]).to_csv(
                f"{GOLD_DIR}/{task_id}.csv", index=False)
    return golds


def load_gold():
    """Gold results for every task (computed on the fly if not on disk)."""
    if not os.path.isdir(GOLD_DIR):
        compute_gold()
    golds = {}
    for f in os.listdir(GOLD_DIR):
        if f.endswith(".csv"):
            d = pd.read_csv(os.path.join(GOLD_DIR, f), keep_default_na=False)
            golds[f[:-4]] = set(zip(d.a, d.b))
    return golds


if __name__ == "__main__":
    golds = compute_gold()
    for k, v in golds.items():
        print("%-16s %8d gold tuples" % (k, len(v)))
