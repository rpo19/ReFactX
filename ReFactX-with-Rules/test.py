"""Sanity tests and a guided tour of the KG and the QA set:  python3 test.py

Part 1 re-derives every rule from the base triples and re-answers every question,
independently of reasoner.py / add_groundtruth.py.  Part 2 just prints examples.
"""
import random
import sys
from collections import defaultdict
import pandas as pd
import reasoner as rz
import qa_gen
from add_groundtruth import answer

E = pd.read_csv("data/entities.csv", keep_default_na=False)
T = pd.read_csv("data/triples.csv", keep_default_na=False, low_memory=False)
QA = pd.read_csv("data/qa_answers.csv", keep_default_na=False)
R = rz.Reasoner(T)
TYPE = dict(zip(E.entity, E.type))
ALL = sorted({s for p in R.out for s in R.out[p]})
rnd = random.Random(0)
fails = []


def check(name, cond):
    print("%-5s %s" % ("ok" if cond else "FAIL", name))
    if not cond:
        fails.append(name)


def pairs(prop, limit=None):
    p = [(x, y) for x, ys in R.out[prop].items() for y in ys]
    return rnd.sample(p, limit) if limit else p


def reverse(prop):
    """The reverse direction of `prop`, recomputed from scratch."""
    rev = defaultdict(set)
    for s, objs in R.out[prop].items():
        for o in objs:
            rev[o].add(s)
    return rev


print("=" * 78, "\n1. KG\n" + "=" * 78)
check("100k entities, unique names", len(E) == 100_000 == E.entity.nunique())
check("names contain no '<' or '>' (they delimit the verbalized facts)",
      not E.entity.str.contains("[<>]").any())
check("only the base relations are materialized", set(T.predicate) == set(rz.BASE))
check("every subject/object is a known entity",
      set(T.subject) <= set(TYPE) and set(T.object) <= set(TYPE))
check("no self-loops", not (T.subject == T.object).any())
check("OWN and ROLE always point to a company",
      all(TYPE[o] == "company" for o in T.object[T.predicate != "REACHABLE"]))
check("only persons hold a ROLE", all(TYPE[s] == "person" for s in T.subject[T.predicate == "ROLE"]))
own = T[T.predicate == "OWN"]
check("no cap table exceeds 100%", own.value.astype(float).groupby(own.object).sum().max() <= 100)
check("no two entities each hold a majority of the other",
      not any(p > 50 and R.pct.get((b, a), 0) > 50 for (a, b), p in R.pct.items()))

print("\n" + "=" * 78, "\n2. rules (recomputed from the base triples)\n" + "=" * 78)
role = {(x, y) for x, ys in R.stored["ROLE"].items() for y in ys}
check("QUALIFIED_HOLDINGS == OWN with a stake >= 10%",
      set(pairs("QUALIFIED_HOLDINGS")) == {k for k, p in R.pct.items() if p >= 10})
check("INFLUENCE == OWN with a stake >= 20%, or ROLE",
      set(pairs("INFLUENCE")) == {k for k, p in R.pct.items() if p >= 20} | role)
own_in = defaultdict(list)
for (o, t), p in R.pct.items():
    own_in[t].append((o, p))
check("CONTROL: stakes held by X and by what X controls always sum to > 50%",
      all(sum(p for z, p in own_in[y] if z == x or R.ask(x, "CONTROL", z)) > 50
          for x, y in pairs("CONTROL", 300)))
check("CONTROL: a direct majority stake always yields control",
      all(R.ask(x, "CONTROL", y) for x, y in rnd.sample([k for k, p in R.pct.items() if p > 50], 300)))
check("CONTROL is transitive",
      all(R.ask(x, "CONTROL", z) for x, y in pairs("CONTROL", 300) for z in R.get(y, "CONTROL")))
ctrl = set(pairs("CONTROL"))
check("no two entities control each other, and nobody controls itself",
      not any((y, x) in ctrl or x == y for x, y in ctrl))
check("ULTIMATE_CONTROL(X,Y) => CONTROL(X,Y) and nobody controls X",
      all((x, y) in ctrl and R.count(x, "is_control") == 0 for x, y in pairs("ULTIMATE_CONTROL", 300)))
check("INDIRECT_ROLE == a ROLE in an entity that controls the target",
      set(pairs("INDIRECT_ROLE")) == {(x, y) for x, zs in R.stored["ROLE"].items()
                                      for z in zs for y in R.get(z, "CONTROL")})
adj = defaultdict(set)
for s, ys in R.stored["REACHABLE"].items():
    for y in ys:
        adj[s].add(y)
        adj[y].add(s)


def component(e):
    seen, pending = set(), [e]
    while pending:
        v = pending.pop()
        if v not in seen:
            seen.add(v)
            pending += adj[v]
    return seen - {e}


check("REACHABLE is symmetric even though it is stored one way only",
      all(R.ask(y, "REACHABLE", x) for x, y in pairs("REACHABLE", 300)))
check("CONNECTED == the REACHABLE component of the entity",
      all(set(R.get(e, "CONNECTED")) == component(e) for e in rnd.sample(sorted(adj), 200)))

print("\n" + "=" * 78, "\n3. property names and query directions\n" + "=" * 78)
sample = rnd.sample(ALL, 300)
check("every name from properties() is accepted and has at least one result",
      all(R.count(e, n) > 0 for e in sample for n in R.properties(e)))
check("properties() advertises no 'is_' form for a symmetric relation",
      not any(n[3:].upper() in rz.SYMMETRIC for e in sample for n in R.properties(e)
              if n.startswith("is_")))
check("properties() lists a relation whenever the entity has a result for it",
      all(R.properties(e) == {n for p in R.out for n in (p.lower(), "is_" + p.lower())
                              if R.count(e, n) and not (n.startswith("is_") and p in rz.SYMMETRIC)}
          for e in sample))
asym = [p for p in R.out if p not in rz.SYMMETRIC]
check("'is_p' returns exactly the entities X with p(X, entity)",
      all(R.get(e, "is_" + p.lower()) == sorted(rev[e])
          for p in asym for rev in [reverse(p)] for e in rnd.sample(sorted(rev), 100)))
check("ask(X, p, Y) and ask(Y, 'is_p', X) are the same fact",
      all(R.ask(x, p.lower(), y) == R.ask(y, "is_" + p.lower(), x)
          for p in asym for x, y in pairs(p, 100)))
check("property names are case-insensitive",
      all(R.get(e, "own") == R.get(e, "OWN") for e in rnd.sample(sorted(R.out["OWN"]), 100)))
try:
    R.get(sample[0], "owned_by")
    rejected = False
except KeyError:
    rejected = True
check("an unknown property name is rejected", rejected)

print("\n" + "=" * 78, "\n4. fact index\n" + "=" * 78)
facts = open("data/facts.txt").read().splitlines()
triples = set(zip(T.subject, T.predicate, T.object))
check("facts.txt has one fact per stored triple, nothing else", len(facts) == len(T))
check("facts.txt holds no derived relation and no inverse",
      {f.split("><")[1] for f in facts} == set(rz.BASE))
check("every fact in facts.txt is a stored triple",
      all(tuple(f[1:-2].split("><")) in triples for f in rnd.sample(facts, 300)))
n_all, preds = 0, set()
for line in open("data/all_facts_and_derived.txt"):
    n_all += 1
    preds.add(line.split("><")[1])
check("all_facts_and_derived.txt holds every base and derived fact",
      n_all == sum(sum(len(v) for v in R.out[p].values()) for p in rz.BASE + rz.RULES))
check("all_facts_and_derived.txt covers all base and derived relations",
      preds == set(rz.BASE + rz.RULES))

print("\n" + "=" * 78, "\n5. QA\n" + "=" * 78)
check("every ground-truth answer is reproducible",
      [answer(R, r) for _, r in QA.iterrows()] == list(QA.answer))
check("the six question kinds are all present", set(QA.kind) == set(qa_gen.PLAN))
check("every entity in a question exists", all(e in TYPE for c in "xyz" for e in QA[c] if e))
check("no duplicate question", QA.question.nunique() == len(QA))
check("list answers hold 1 to 10 entities", QA.n_answers.between(1, 10).all())
check("count answers are positive", all(int(a) > 0 for a in QA.answer[QA.answer_type == "count"]))
yn = QA[QA.answer_type == "yes_no"]
check("every yes/no kind x relation is exactly half yes, half no",
      yn.groupby(["kind", "property"]).answer.apply(lambda s: (s == "yes").sum() * 2 == len(s)).all())
check("a 'top' answer is the strict maximum of its three candidates",
      all(R.count(r.answer, r["property"]) >
          max(R.count(e, r["property"]) for e in (r.x, r.y, r.z) if e != r.answer)
          for _, r in QA[QA.kind == "top"].iterrows()))
check("a 'more' question compares two entities with different counts",
      all(R.count(r.x, r["property"]) != R.count(r.z, r["property"])
          for _, r in QA[QA.kind == "more"].iterrows()))

print("\n%d checks failed" % len(fails) if fails else "\nall checks passed")

print("\n" + "=" * 78, "\nthe KG at a glance\n" + "=" * 78)
print(len(E), "entities", dict(E.type.value_counts()), "|", len(T), "base triples")
for p in rz.BASE:
    r = T[T.predicate == p].iloc[0]
    print("   %-10s %s  ->  %s   value=%r" % (p, r.subject, r.object, r.value))
print("derived by the reasoner (not in facts.txt):")
for p in rz.RULES:
    print("   %-20s %8d facts" % (p, sum(len(v) for v in R.out[p].values())))

print("\n" + "=" * 78, "\nquerying the reasoner\n" + "=" * 78)
x = max(R.out["ULTIMATE_CONTROL"], key=lambda e: len(R.out["ULTIMATE_CONTROL"][e]))
print("entity:", x, "\nproperties():")
for n in sorted(R.properties(x)):
    print("   %-24s %6d results" % (n, R.count(x, n)))
y = R.get(x, "control")[0]
print("   get(x, 'control')[:2]   ->", R.get(x, "control")[:2])
print("   get(x, 'is_own')        ->", R.get(x, "is_own"), "(the entities that own x)")
print("   ask(x, 'control', y)    ->", R.ask(x, "control", y), " y = %r" % y[:40])
print("   ask(y, 'is_control', x) ->", R.ask(y, "is_control", x), "(the same fact, reversed)")

print("\n" + "=" * 78, "\none question per kind\n" + "=" * 78)
for _, r in QA.drop_duplicates("kind").iterrows():
    print("[%-8s %-18s] %s\n   -> %s" % (r.kind, r["property"], r.question, r.answer[:110]))

sys.exit(1 if fails else 0)
