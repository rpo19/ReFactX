#!/usr/bin/env python3
"""Load ents_freebase.pickle and write entity names as '<name>' one per line, gzipped."""
import pickle, gzip, time

PICKLE = "/workspace/data/ents_freebase.pickle"
OUT = "/workspace/data/entity_names.txt.gz"

t0 = time.time()
print(f"Loading {PICKLE} ...")
with open(PICKLE, "rb") as f:
    ents = pickle.load(f)
print(f"  {len(ents):,} entities ({time.time()-t0:.0f}s)")

print(f"Writing {OUT} ...")
t0 = time.time()
total_lines = 0
with gzip.open(OUT, "wt", encoding="utf-8") as out:
    for i, (mid, (label, alt_labels, _)) in enumerate(ents.items()):
        if label:
            out.write(f" {label}\n")
            total_lines += 1
        for alt in alt_labels:
            if alt:
                out.write(f" {alt}\n")
                total_lines += 1
        if i > 0 and i % 5_000_000 == 0:
            print(f"  {i:,}/{len(ents):,} entities processed, {total_lines:,} lines ({time.time()-t0:.0f}s)")
print(f"  done: {len(ents):,} entities, {total_lines:,} lines ({time.time()-t0:.0f}s)")
