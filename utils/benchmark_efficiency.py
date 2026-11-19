"""Reproducible micro-benchmarks for ReFactX cache and constrained decoding.

Examples::

    python utils/benchmark_efficiency.py cache
    python utils/benchmark_efficiency.py processor --model Qwen/Qwen3.5-0.8B
    python utils/benchmark_efficiency.py postgres --table qwen36 --samples 200

The PostgreSQL mode reads ``BASE_INDEX_PATH`` (or ``--index``) from ``.env`` and
only performs read-only queries.
"""

from __future__ import annotations

import argparse
import os
import pickle
import statistics
import time

import torch

from refactx.SimpleCache import SimpleCache
from refactx.index import DictIndex, PostgresIngestIndex, PostgresTrieIndex


class _State:
    def __init__(self):
        self.subtree_cache = DictIndex()

    def end_of_triple_reset(self):
        self.subtree_cache.reset()


def _clone_state():
    return _State()


# ---------------------------------------------------------------------------
# Cache micro-benchmarks (no tokenizer, no database)
# ---------------------------------------------------------------------------

def _make_subtree(nodes=500):
    return [nodes, list(range(nodes))]


def _pickle_size(value):
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, tuple):
        next_tokens, subtree = value
        return len(subtree) + len(pickle.dumps(next_tokens))
    return len(pickle.dumps(value))


def bench_simple_cache(entries=1024, subtree_nodes=500, repeats=3):
    sequences = [list(range(10)) + [i] for i in range(entries)]
    subtree = _make_subtree(subtree_nodes)
    best_add = best_hit = float('inf')
    for _ in range(repeats):
        cache = SimpleCache(maxsize=entries)
        start = time.perf_counter()
        for sequence in sequences:
            cache.add(sequence, {1: 2, 3: 4}, subtree)
        best_add = min(best_add, time.perf_counter() - start)

        start = time.perf_counter()
        for sequence in sequences:
            cache.next_tokens(sequence)
        best_hit = min(best_hit, time.perf_counter() - start)

    print(f"SimpleCache entries={entries} subtree_nodes={subtree_nodes}")
    print(f"  add   {best_add * 1000:8.2f} ms")
    print(f"  hit   {best_hit * 1000:8.2f} ms")
    print(f"  stats {cache.stats()}")


def bench_canonical_growth(queries=300, switch_parameter=4):
    """Show how large per-prefix cache entries stay with canonical subtrees."""
    sequences = [[1, 2, child, 1000 + child] for child in range(queries)]
    ingest = PostgresIngestIndex(
        tokenizer_name='bench', rootkey=-100,
        switch_parameter=switch_parameter, table_name='bench', configkey=-200,
    )
    for sequence in sequences:
        ingest.add(sequence)
    rows = {
        tuple(key): [(0, children, subtree, numleaves, childrenleaves)]
        for key, children, numleaves, childrenleaves, subtree in ingest.get_rows()
    }

    class FakeIndex(PostgresTrieIndex):
        def __init__(self):
            super().__init__(
                postgresql_connection='', table_name='bench',
                switch_parameter=switch_parameter, rootkey=-100,
                cache=SimpleCache(maxsize=queries * 2),
            )

        def _query_postgresql(self, sequence):
            return rows.get(tuple(sequence), [])

    index = FakeIndex()
    state = _clone_state()
    start = time.perf_counter()
    for child in range(queries):
        index.next_tokens([1, 2, child], state=state)
    elapsed = time.perf_counter() - start

    values = list(index.cache.cache.values())
    sizes = [_pickle_size(value) for value in values]
    print(f"canonical subtree cache queries={queries}")
    print(f"  elapsed       {elapsed * 1000:8.2f} ms")
    print(f"  entries       {len(values)}")
    print(f"  total size    {sum(sizes) / 1e6:8.3f} MB")
    print(f"  avg entry     {statistics.mean(sizes) / 1e3:8.1f} KB")
    print(f"  max entry     {max(sizes) / 1e3:8.1f} KB")


# ---------------------------------------------------------------------------
# Constrained processor micro-benchmark
# ---------------------------------------------------------------------------

def bench_processor(model_name='Qwen/Qwen3.5-0.8B', batch=8, reps=2):
    from transformers import AutoTokenizer

    from refactx.generate import get_constrained_logits_processor

    torch.set_num_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    index = DictIndex()
    index.set_tokenizer(tokenizer)
    object_text = " ".join(["thing"] * 40)
    for i in range(50):
        index.add(tokenizer(
            f"<Subject{i}> <relation> <{object_text} number {i}> .",
            add_special_tokens=False,
        )["input_ids"])

    feed = tokenizer(
        f"<Subject0> <relation> <{object_text} number 0> .",
        add_special_tokens=False,
    )["input_ids"]
    marker = tokenizer("<fact>", add_special_tokens=False)["input_ids"]
    seed = tokenizer("x", add_special_tokens=False)["input_ids"][:1]
    prose = tokenizer(
        "This is an ordinary sentence with no special marker. " * 8,
        add_special_tokens=False,
    )["input_ids"]
    vocab_size = len(tokenizer)
    steps = len(marker) + len(feed)

    best = float('inf')
    for _ in range(reps):
        processor = get_constrained_logits_processor(
            tokenizer, index, num_beams=1, num_batches=batch,
            sentinel=False, fact_pattern='<fact>', count_pattern='<count>',
            eot=' </fact>\n', return_list=False,
        )
        sequences = [list(seed) for _ in range(batch)]
        processor(torch.tensor(sequences), torch.randn(batch, vocab_size))
        start = time.perf_counter()
        for step in range(steps):
            for row in range(batch):
                if row == 0 and step < len(marker):
                    sequences[row].append(marker[step])
                elif row == 0 and step - len(marker) < len(feed):
                    sequences[row].append(feed[step - len(marker)])
                else:
                    sequences[row].append(prose[step % len(prose)])
            processor(torch.tensor(sequences), torch.randn(batch, vocab_size))
        best = min(best, time.perf_counter() - start)

    print(f"processor model={model_name} batch={batch} steps={steps}")
    print(f"  best wall     {best * 1000:8.2f} ms")
    print(f"  per step      {best / steps * 1000:8.3f} ms")


# ---------------------------------------------------------------------------
# Live PostgreSQL cache benchmark (read-only)
# ---------------------------------------------------------------------------

def bench_postgres(index_url=None, table=None, samples=200, warm_samples=50):
    from dotenv import load_dotenv
    load_dotenv()
    import psycopg
    from transformers import AutoTokenizer

    import refactx

    index_url = index_url or os.getenv('BASE_INDEX_PATH')
    if not index_url:
        raise SystemExit("Set BASE_INDEX_PATH or pass --index")
    table = table or os.getenv('INDEX_TABLE', 'qwen36')
    tok_name = os.getenv('BENCH_TOKENIZER', 'Qwen/Qwen3.5-0.8B')
    tokenizer = AutoTokenizer.from_pretrained(tok_name)

    index = refactx.load_index(f"{index_url}?tablename={table}", tokenizer=tokenizer)
    switch = index.switch_parameter
    with psycopg.connect(index_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT key FROM {table} WHERE cardinality(key)=%s LIMIT %s",
                (switch, samples),
            )
            keys = [row[0] for row in cursor.fetchall()]

    state = _clone_state()
    start = time.perf_counter()
    for key in keys:
        index.next_tokens(key[1:], state=state)
    cold = time.perf_counter() - start

    warm_keys = keys[:warm_samples]
    start = time.perf_counter()
    for key in warm_keys:
        state = _clone_state()
        index.next_tokens(key[1:], state=state)
    warm = time.perf_counter() - start

    print(f"postgres table={table} switch_parameter={switch}")
    print(f"  cold queries  {len(keys):6d} in {cold * 1000:9.2f} ms "
          f"({cold / max(len(keys), 1) * 1000:.3f} ms/query)")
    print(f"  warm hits     {len(warm_keys):6d} in {warm * 1000:9.2f} ms "
          f"({warm / max(len(warm_keys), 1) * 1000:.3f} ms/query)")
    print(f"  stats {index.cache_stats()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'mode', nargs='?', default='all',
        choices=('all', 'cache', 'processor', 'postgres'),
    )
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--index', default=None)
    parser.add_argument('--table', default=None)
    parser.add_argument('--samples', type=int, default=200)
    args = parser.parse_args()

    if args.mode in ('all', 'cache'):
        bench_simple_cache()
        print()
        bench_canonical_growth()
        print()
    if args.mode in ('all', 'processor'):
        bench_processor(args.model)
        print()
    if args.mode in ('all', 'postgres'):
        bench_postgres(args.index, args.table, args.samples)


if __name__ == '__main__':
    main()
