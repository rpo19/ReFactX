"""Smoke test for the Neo4j connector (no model needed).

Run:  python scripts/test_neo4j_kg.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from refactx.neo4j_kg import Neo4jKnowledgeGraph

t0 = time.time()
kg = Neo4jKnowledgeGraph()
print(f"connected in {time.time() - t0:.2f}s  database={kg.database!r}")

for seed in ["Barack Obama", "Paris"]:
    t0 = time.time()
    rels = kg.property_strings(seed)
    print(f"\n=== {seed}: {len(rels)} relations ({time.time() - t0:.2f}s) ===")
    print(rels[:15])

    if rels:
        rel = rels[0]
        t0 = time.time()
        objs = kg.get(seed, rel)
        print(f"  get({seed!r}, {rel!r}) -> {len(objs)} objects ({time.time() - t0:.2f}s): {objs[:5]}")
        if objs:
            print(f"  ask -> {kg.ask(seed, rel, objs[0])}")

t0 = time.time()
names = kg.entity_index(limit=200, seeds=["Barack Obama"], hops=1)
print(f"\n=== entity_index(limit=200, seeds=['Barack Obama'], hops=1): "
      f"{len(names)} names ({time.time() - t0:.2f}s) ===")
print(names[:10])

kg.close()
print("\nOK")
