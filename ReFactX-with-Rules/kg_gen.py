"""Synthetic company-ownership KG: entities + base triples.

Only three relations are materialized here -- OWN (with a % stake), ROLE and
REACHABLE.  Everything else (CONTROL, ULTIMATE_CONTROL, ...) is derived by the
reasoner at query time, see reasoner.py.
"""
import random
import pandas as pd
from faker import Faker

N_ENTITIES = 100_000
COMPANY_FRAC = 0.70
GROUP_SIZE = 10                 # companies per corporate group
CLUSTER_SIZE = (3, 12)          # size range of a REACHABLE cluster
ROLES = ["ceo", "chairman", "board member", "director", "general manager", "auditor"]


def make_entities(n=N_ENTITIES, seed=0):
    """`n` uniquely-named entities: 70% companies, 30% persons."""
    Faker.seed(seed)
    fake = Faker("it_IT")
    n_comp = int(n * COMPANY_FRAC)
    used, rows = set(), []
    for gen, k, kind in [(fake.company, n_comp, "company"), (fake.name, n - n_comp, "person")]:
        while k:
            name = gen().strip()
            if name not in used:
                used.add(name)
                rows.append((name, kind))
                k -= 1
    return pd.DataFrame(rows, columns=["entity", "type"])


def make_triples(entities, seed=0):
    """Base triples: ownership groups, person stakes/roles, REACHABLE clusters."""
    rnd = random.Random(seed)
    companies = entities.entity[entities.type == "company"].tolist()
    persons = entities.entity[entities.type == "person"].tolist()
    triples, held, stake = [], {}, {}

    def own(owner, target, pct):
        """Add OWN if the cap table of `target` stays <= 100% and A and B do not
        each hold a majority of the other, which cannot exist."""
        if owner == target or (owner, target) in stake:
            return
        if held.get(target, 0) + pct > 100 or (pct > 50 and stake.get((target, owner), 0) > 50):
            return
        stake[(owner, target)] = pct
        held[target] = held.get(target, 0) + pct
        triples.append((owner, "OWN", target, pct))

    # a corporate group is a random ownership tree: majority stakes give direct
    # control, pairs of minority stakes give joint (derived) control
    groups = [companies[i:i + GROUP_SIZE] for i in range(0, len(companies), GROUP_SIZE)]
    for gi, g in enumerate(groups):
        for i in range(1, len(g)):
            if rnd.random() < 0.6:
                own(g[rnd.randrange(i)], g[i], rnd.randint(51, 95))
            else:
                own(g[rnd.randrange(i)], g[i], rnd.randint(20, 45))
                own(g[rnd.randrange(i)], g[i], rnd.randint(20, 45))
        r = rnd.random()                      # who sits on top of the group
        if gi and r < 0.30:                   # another group -> longer control chains
            own(rnd.choice(groups[rnd.randrange(gi)]), g[0], rnd.randint(51, 90))
        elif r < 0.75:                        # a person -> person as ultimate controller
            own(rnd.choice(persons), g[0], rnd.randint(51, 90))

    ## sum to 1 all the ownership stakes of each company, to avoid "ghost" shares
    for c in companies:
        tot = held.get(c, 0)
        if tot < 100:
            own(rnd.choice(persons), c, 100 - tot)

    # persons: minority stakes and roles
    for p in persons:
        for _ in range(rnd.randint(0, 2)):
            own(p, rnd.choice(companies), rnd.randint(5, 30))
        for _ in range(rnd.choices([0, 1, 2, 3], weights=[35, 35, 20, 10])[0]):
            triples.append((p, "ROLE", rnd.choice(companies), rnd.choice(ROLES)))

    # REACHABLE: small symmetric clusters (e.g. shared address/contacts), so that
    # transitive closure stays small and enumerable
    pool = companies + persons
    rnd.shuffle(pool)
    i = 0
    while i < len(pool):
        cl = pool[i:i + rnd.randint(*CLUSTER_SIZE)]
        i += len(cl)
        for j in range(1, len(cl)):
            triples.append((cl[rnd.randrange(j)], "REACHABLE", cl[j], ""))

    return pd.DataFrame(triples, columns=["subject", "predicate", "object", "value"])


if __name__ == "__main__":
    ents = make_entities()
    trip = make_triples(ents)
    ents.to_csv("data/entities.csv", index=False)
    trip.to_csv("data/triples.csv", index=False)
    print(len(ents), "entities,", len(trip), "triples")
    print(trip.predicate.value_counts().to_string())
