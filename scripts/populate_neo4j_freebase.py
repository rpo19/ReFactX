#!/usr/bin/env python3
"""
Populate a new Neo4j database from Freebase data using neo4j-admin bulk import.
Follows the model from neo4j.md: sanitized predicate names as relationship types.

Steps:
  1. Stop Neo4j
  2. Create nodes CSV from ents_freebase.pickle
  3. Filter fb_en.txt for entity-to-entity triples, write rels CSV with sanitized types
  4. neo4j-admin database import full (bulk load into a new database)
  5. Start Neo4j
"""
import os, sys, time, subprocess, pickle, csv, gc, re

NEO4J_HOME = "/workspace/notebooks/neo4j"
DB_NAME = "freebase-v2"
PICKLE_PATH = "/workspace/data/ents_freebase.pickle"
FB_EN = "/workspace/data/FastRDFStore-data/data/fb_en.txt"
NODES_CSV = "/workspace/data/_import_nodes.csv"
RELS_CSV = "/workspace/data/_import_rels.csv"
BIN = lambda *a: os.path.join(NEO4J_HOME, "bin", *a)

_CYPHER_KEYWORDS = {
    "ALL","AND","AS","ASC","BY","CREATE","DELETE","DESC","DETACH","DO",
    "DROP","ELSE","END","EXISTS","FOR","FROM","IN","IS","LIMIT","MATCH",
    "MERGE","NOT","NULL","OPTIONAL","ORDER","REMOVE","RETURN","SET",
    "SKIP","THEN","UNION","UNWIND","WHEN","WHERE","WITH","XOR","YIELD",
}


def fmt(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    return f"{secs/60:.1f}m"


def sanitize_rel_type(predicate):
    t = predicate.lstrip("/")
    t = t.replace("/", "_").replace(".", "_")
    t = t.upper()
    if re.match(r"^[0-9]", t) or t in _CYPHER_KEYWORDS:
        t = "FB_" + t
    if len(t) > 400:
        t = t[:400]
    return t


def run_cmd(cmd, timeout=7200):
    print(f"  $ {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for out in [r.stdout, r.stderr]:
        for l in out.strip().split("\n"):
            if l.strip():
                print(f"    {l}")
    if r.returncode != 0:
        raise RuntimeError(f"Exited {r.returncode}")
    return r


def stop_neo4j():
    print("Stopping Neo4j...")
    run_cmd([BIN("neo4j"), "stop"], timeout=120)
    time.sleep(5)


def start_neo4j():
    print("Starting Neo4j...")
    run_cmd([BIN("neo4j"), "start"], timeout=120)
    import neo4j as neo4j_driver
    for i in range(60):
        time.sleep(2)
        try:
            d = neo4j_driver.GraphDatabase.driver(
                "bolt://localhost:7687", auth=("neo4j", "password"))
            d.verify_connectivity()
            d.close()
            print("  Neo4j is ready")
            return
        except Exception:
            pass
    print("  WARNING: Neo4j may not be ready")


def create_nodes_csv():
    t0 = time.time()
    print(f"Loading pickle {PICKLE_PATH}...")
    with open(PICKLE_PATH, "rb") as f:
        ents = pickle.load(f)
    print(f"  loaded {len(ents):,} entities ({fmt(time.time() - t0)})")

    print(f"Writing nodes CSV to {NODES_CSV}...")
    t0 = time.time()
    with open(NODES_CSV, "w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id:ID(Entity-ID)", "name", "description", ":LABEL"])
        for i, (mid, (label, _, desc)) in enumerate(ents.items()):
            w.writerow([mid, (label or "")[:200], (desc or "")[:500], "Entity"])
            if i > 0 and i % 5000000 == 0:
                print(f"  wrote {i:,}/{len(ents):,} ({fmt(time.time() - t0)})")
    print(f"  wrote {len(ents):,} nodes ({fmt(time.time() - t0)})")
    del ents
    gc.collect()


def create_rels_csv():
    t0 = time.time()
    total_lines = 920785545
    count = rel_count = 0
    print(f"Processing {FB_EN} and writing rels CSV to {RELS_CSV}...")
    with open(FB_EN, "r", encoding="utf-8") as f,\
         open(RELS_CSV, "w", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow([":START_ID(Entity-ID)", ":END_ID(Entity-ID)", ":TYPE"])
        for line in f:
            count += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sub, prop, obj_raw = parts
            obj = obj_raw.rstrip(" .")
            if sub.startswith("m.") and obj.startswith("m."):
                rel_type = sanitize_rel_type(prop)
                w.writerow([sub, obj, rel_type])
                rel_count += 1
            if count % 1000000 == 0:
                elapsed = time.time() - t0
                rate = count / elapsed if elapsed > 0 else 0
                pct = count / total_lines * 100
                print(
                    f"  {count:,}/{total_lines:,} ({pct:.1f}%)  "
                    f"{rel_count:,} rels  {rate:,.0f} l/s",
                    end="\r",
                )
                if count % 5000000 == 0:
                    cf.flush()
                gc.collect()
    print(f"\n  done: {count:,} lines, {rel_count:,} rels ({fmt(time.time() - t0)})")
    return rel_count


def bulk_import():
    print(f"Running neo4j-admin database import full for database '{DB_NAME}' ...")
    t0 = time.time()
    cmd = [
        BIN("neo4j-admin"), "database", "import", "full",
        "--overwrite-destination=true",
        "--skip-bad-relationships=true",
        "--bad-tolerance=50000000",
        "--nodes=Entity=" + NODES_CSV,
        "--relationships=" + RELS_CSV,
        "--high-parallel-io=on",
        DB_NAME,
    ]
    run_cmd(cmd)
    print(f"  import finished ({fmt(time.time() - t0)})")


def create_indexes():
    import neo4j as neo4j_driver
    driver = neo4j_driver.GraphDatabase.driver(
        "bolt://localhost:7687", auth=("neo4j", "password"))
    with driver.session(database=DB_NAME) as s:
        s.run("CREATE INDEX entity_fbid_idx IF NOT EXISTS FOR (n:Entity) ON (n.id)").consume()
        s.run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (n:Entity) ON (n.name)").consume()
        print("  indexes created")
        for label in ["Entity"]:
            r = s.run(f"MATCH (n:{label}) RETURN count(*) AS cnt").single()
            print(f"  {label}: {r[0]:,}")
        r = s.run("MATCH ()-[r]->() RETURN count(*) AS cnt").single()
        print(f"  Relationships: {r[0]:,}")
    driver.close()


def main():
    t_start = time.time()

    stop_neo4j()
    create_nodes_csv()
    create_rels_csv()
    bulk_import()
    for f in [NODES_CSV, RELS_CSV]:
        if os.path.exists(f):
            os.remove(f)
    import re
    config_path = os.path.join(NEO4J_HOME, "conf", "neo4j.conf")
    with open(config_path) as f:
        conf = f.read()
    if "initial.dbms.default_database" in conf:
        conf = re.sub(
            r"^[#]?\s*initial\.dbms\.default_database\s*=.*",
            f"initial.dbms.default_database={DB_NAME}",
            conf,
            flags=re.MULTILINE,
        )
    else:
        conf += f"\ninitial.dbms.default_database={DB_NAME}\n"
    with open(config_path, "w") as f:
        f.write(conf)
    start_neo4j()
    create_indexes()

    print(f"\nTotal time: {fmt(time.time() - t_start)}")


if __name__ == "__main__":
    main()
