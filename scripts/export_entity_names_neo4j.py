#!/usr/bin/env python3
"""Export every entity name from Neo4j to a compressed, one-per-line file.

The output is meant to be fed to ``utils/populate_postgres.py`` to build a
Postgres-backed ReFactX prefix index of entity names (the index used by
``KnowledgeGraphGeneration``'s ENTITY phase), instead of building it in memory.

Graph model (see ``neo4j.md``): ``(:Entity {id, name})`` blocks with sanitized
Freebase predicates as relationship types.

Examples
--------
Full export (distinct names), bz2 compressed::

    python scripts/export_entity_names_neo4j.py \\
        --outfile /workspace/data/entity_names.txt.bz2

Then build the index (``--prefix " "`` yields the `` <Name>`` sequences the
generation expects; the ``--end-of-triple`` flag is required by the CLI but is
not written by the indexer)::

    python utils/populate_postgres.py /workspace/data/entity_names.txt.bz2 \\
        --model-name Qwen/Qwen3.5-4B \\
        --prefix " " --end-of-triple " ." \\
        --postgres-connection "postgres://user:pwd@host:5432/postgres" \\
        --table-name entity_names --rootkey 500000 --batch-size 5000000 \\
        --switch-parameter 7 --tokenizer-batch-size 10000 \\
        --total-number-of-triples <N>

Only entities that can act as a subject (have at least one outgoing
relationship)::

    python scripts/export_entity_names_neo4j.py --require-outgoing -o names.bz2

Note: this script talks to Neo4j directly (no ``refactx`` / torch import) so it
stays lightweight.
"""
from __future__ import annotations

import bz2
import gzip
import os
import re
import time

import click
from neo4j import GraphDatabase

DEFAULT_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_USER = os.getenv("NEO4J_USER", "neo4j")
DEFAULT_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
DEFAULT_DATABASE = os.getenv("NEO4J_DATABASE") or None


def _batch_size_label(n: int) -> str:
    return f"{n:,}"


@click.command()
@click.option("--outfile", "-o", required=True, type=click.Path(dir_okay=False),
              help="Output file. Compression is chosen by extension: .bz2 / .gz / plain.")
@click.option("--uri", default=DEFAULT_URI, show_default=True, help="Neo4j bolt URI.")
@click.option("--user", default=DEFAULT_USER, show_default=True)
@click.option("--password", default=DEFAULT_PASSWORD, show_default=True)
@click.option("--database", default=DEFAULT_DATABASE, show_default=True,
              help="Neo4j database (default: server default).")
@click.option("--format", "fmt", default="<{name}>", show_default=True,
              help="Python format string applied to each name before writing.")
@click.option("--distinct/--no-distinct", default=True, show_default=True,
              help="Emit DISTINCT names (the graph has ~2x duplicated names).")
@click.option("--require-outgoing", is_flag=True, default=False,
              help="Only export entities with at least one outgoing relationship.")
@click.option("--min-name-length", type=int, default=1, show_default=True)
@click.option("--max-name-length", type=int, default=0, show_default=True,
              help="0 = no limit.")
@click.option("--exclude-regex", default=None,
              help="Skip names matching this Python regex, e.g. '^[^A-Za-z]' to drop "
                   "names that do not start with a letter.")
@click.option("--limit", type=int, default=0, show_default=True,
              help="0 = no limit (useful for a quick smoke test).")
@click.option("--progress-every", type=int, default=1_000_000, show_default=True,
              help="Print progress every N written lines (0 = never).")
def main(outfile, uri, user, password, database, fmt, distinct, require_outgoing,
         min_name_length, max_name_length, exclude_regex, limit, progress_every):
    """Export Neo4j entity names to ``outfile`` for ``utils/populate_postgres.py``."""

    where = ["n.name IS NOT NULL", "n.name <> ''", "size(n.name) >= $min_len"]
    params = {"min_len": min_name_length}
    if max_name_length and max_name_length > 0:
        where.append("size(n.name) <= $max_len")
        params["max_len"] = max_name_length
    if require_outgoing:
        where.append("EXISTS { (n)-->() }")
    if limit and limit > 0:
        params["limit"] = limit

    distinct_kw = "DISTINCT " if distinct else ""
    exclude = re.compile(exclude_regex) if exclude_regex else None
    query = (
        "MATCH (n:Entity) "
        f"WHERE {' AND '.join(where)} "
        f"RETURN {distinct_kw}n.name AS name"
    )
    if limit and limit > 0:
        query += " LIMIT $limit"

    # Pick the opener by extension.
    suffix = os.path.splitext(outfile)[1].lower()
    if suffix == ".bz2":
        opener, mode = bz2.open, "wt"
    elif suffix == ".gz":
        opener, mode = gzip.open, "wt"
    else:
        opener, mode = open, "w"

    print(f"Output:     {outfile}")
    print(f"Neo4j:      {uri} (database={database!r})")
    print(f"Query:      {query}")
    print(f"Distinct:   {distinct} | require outgoing: {require_outgoing} | limit: {limit or 'none'}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()

    written = 0
    skipped = 0
    started = time.time()
    try:
        with driver.session(database=database) as session:
            result = session.run(query, **params)
            with opener(outfile, mode, encoding="utf-8") as out:
                for record in result:
                    name = record["name"]
                    if not name:
                        skipped += 1
                        continue
                    name = name.strip()
                    # Names spanning newlines cannot be represented one-per-line.
                    if "\n" in name or "\r" in name:
                        skipped += 1
                        continue
                    if exclude is not None and exclude.search(name):
                        skipped += 1
                        continue
                    out.write(fmt.format(name=name))
                    out.write("\n")
                    written += 1
                    if progress_every and written % progress_every == 0:
                        print(f"  {_batch_size_label(written)} lines "
                              f"({time.time() - started:.0f}s)")
    finally:
        driver.close()

    elapsed = time.time() - started
    print(f"Done: {_batch_size_label(written)} lines, {skipped} skipped, "
          f"{elapsed:.0f}s")
    if written:
        print(f"Pass --total-number-of-triples {written} to populate_postgres.py.")


if __name__ == "__main__":
    main()
