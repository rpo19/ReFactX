# Constrained Generation Speed Report

## Scope

This review covers:

- `refactx/generate.py`
- `refactx/index.py`
- `refactx/SimpleCache.py`
- `refactx/httpindex.py`

The focus is the PostgreSQL-backed index, the in-process `SimpleCache`, and their effect on autoregressive LLM generation.

## Executive summary

Constrained generation does not make the transformer forward pass itself more expensive. The model still computes logits in the usual way. However, the constrained logits processor runs synchronously after every model step, before the next token can be selected. Consequently, all index lookup, cache, Python, tokenization, and masking work directly reduces end-to-end decoding throughput.

The expected impact depends on the index:

| Configuration | Expected impact |
|---|---|
| In-memory `DictIndex` | Moderate CPU overhead; usually manageable |
| Local PostgreSQL with warm cache | Noticeable but potentially acceptable |
| PostgreSQL with cache misses | Significant latency during the first prefixes of each fact |
| HTTP PostgreSQL index | Potentially high latency because lookups are synchronous network requests |
| Beam search | Higher cost due to per-beam work and state copying |

The most concerning costs are synchronous PostgreSQL/HTTP lookups, storing and repeatedly serializing complete subtree-cache snapshots, repeated full-sequence decoding/copying, and allocating a vocabulary-sized mask on every generation step.

## Generation path

`ConstrainedLogitsProcessor.__call__` is invoked by Transformers during each decoding iteration. It currently:

1. Converts each `input_ids` row to a Python list with `tolist()`.
2. Updates per-beam state and checks generation patterns.
3. Invokes `FactGeneration.constrain()` or `CountBranchesGeneration.constrain()` when a pattern is active.
4. Builds a full `scores`-shaped mask and adds it to the logits.

The key code path is:

```python
mask = torch.zeros_like(scores)
...
scores_processed = scores + mask
```

For unconstrained rows, much of this mask is zero, but it is still allocated and added. For constrained rows, the processor sets invalid vocabulary entries to negative infinity and leaves valid entries available.

### Important consequence

The additional work is not part of the transformer forward pass, but it is serialized into the autoregressive loop:

```text
model forward
-> constrained logits processor
-> token selection
-> next model forward
```

Therefore it does slow observed generation speed, even if model GPU utilization and forward-pass timing are unchanged.

## PostgreSQL index analysis

`PostgresTrieIndex.next_tokens()` uses PostgreSQL while the prefix length is at or below `switch_parameter`. Once the prefix is longer than that threshold, it uses the in-memory `state.subtree_cache`.

With the default switch threshold of 7, a new fact can cause several database lookups while its initial prefix is being generated. A cache miss executes:

```python
with self.postgresql_connection.cursor() as cursor:
    cursor.execute(self.select_query, (sequence,))
    query_result = cursor.fetchall()
```

### Positive aspects

- Database access is limited to short prefixes.
- The trie subtree is loaded into memory for subsequent tokens.
- The indexed `INT[]` key is queried with a parameterized value.
- `childrenleaves` allows branch counts to be returned without loading all facts.
- The index is built after bulk ingestion, which is appropriate for the population path.

### Main costs and risks

#### Synchronous database access

A cache miss blocks decoding until PostgreSQL returns. Even a fast local query is substantially slower than an in-memory dictionary lookup. A remote database adds network latency, and the HTTP implementation adds request serialization, transport, response deserialization, and possible retry delays.

#### Multiple queries per fact prefix

The implementation generally resolves short prefixes one at a time. This means a new fact can pay several database round trips before the subtree cache takes over. Prefetching a larger subtree earlier would reduce this cost.

#### Python subtree deserialization and merging

For rows containing a subtree, the implementation performs `pickle.loads()` and merges the result into `state.subtree_cache`. Large subtrees can make this CPU- and memory-intensive. Recursive leaf counting is explicitly slower when `do_count_leaves=True`.

#### Shared connection in server usage

The index keeps one PostgreSQL connection and the HTTP service exposes a global connection. This is suitable for a simple single-stream process but can cause contention or unsafe access patterns with concurrent requests. A connection pool or per-request connection strategy would be more robust.

## `SimpleCache` analysis

`SimpleCache` is an LRU cache with a default size of 1024 entries. It can eliminate repeated database queries for exactly matching prefixes, but its representation is more expensive than necessary.

### String keys

Every lookup and insertion converts the token sequence to a string:

```python
return '.'.join(map(str, sequence))
```

This is O(prefix length) and allocates a new string. Since token IDs are integers, a tuple is a simpler and generally more efficient key:

```python
tuple(sequence)
```

### Pickle on every hit

Entries are pickled on insertion and unpickled on every hit:

```python
value = pickle.dumps([next_tokens, subtree_cache], protocol=5)
...
_next_tokens, subtree_cache = pickle.loads(value)
```

For an in-process cache, direct object storage would avoid this overhead. If isolation from mutation is required, use immutable structures or copy only the small mutable component that needs protection.

### Duplicated subtree snapshots

The largest concern is that every cache entry stores the complete `state.subtree_cache` snapshot:

```python
self.cache.add(sequence, _next_tokens, state.subtree_cache)
```

Adjacent prefixes can therefore contain large, overlapping serialized trie snapshots. This increases:

- memory usage,
- pickle/unpickle time,
- allocation and garbage-collection pressure,
- cache eviction cost.

The cache should store one reusable subtree per database prefix, or store only the direct database result and maintain subtree state separately.

### Cache size and locality

A 1024-entry LRU may be effective for repeated prefixes but may have a low hit rate when requests contain many unique subjects. Increasing the size without reducing entry size could create excessive memory use. Cache hit rate and average entry size should be measured before changing the limit.

The cache is process-local. Multiple workers each maintain an independent cache, reducing aggregate hit rate and increasing database traffic.

## Other generation overhead

### Full sequence conversion and comparison

Each step performs:

```python
sequence = input_ids[i].tolist()
```

and often compares the entire prefix:

```python
if sequence[:-1] != state.input_ids:
```

This creates and compares increasingly long Python lists. Over long generations this can become avoidable O(T²) bookkeeping.

### Repeated full-sequence decoding

`FactGeneration.constrain()` calls `tokenizer.decode(sequence)` for structural decisions such as determining whether the generation is at the object level. Decoding the full sequence repeatedly is unnecessary. The state machine could track subject/relation/object phase incrementally from token IDs or known delimiter sequences.

### Vocabulary-sized mask allocation

A zero mask matching the complete logits tensor is allocated for every processor call. This can be expensive for large vocabularies, batches, and beam counts. Where compatible with the Transformers API, constrained rows could be modified directly or a reusable mask buffer could be used.

### Beam state copying

`ConstrainedStateList.beam_permutation()` may deep-copy `cache_index` and `subtree_cache`. Large trie structures can make beam permutations expensive in both CPU time and memory. Sharing immutable index data and using copy-on-write state would reduce this cost.

## Does constrained generation slow the LLM?

Yes, it slows end-to-end generation, but not the neural model's forward computation itself.

The distinction is important:

- **Transformer compute:** unchanged in principle.
- **Per-token decoding latency:** increased by constraint processing.
- **Tokens per second:** reduced whenever the processor takes meaningful time.
- **First use of a prefix:** especially affected by PostgreSQL or HTTP cache misses.
- **Repeated prefixes:** potentially much cheaper after warm-up.

The likely ordering of bottlenecks is:

1. HTTP or remote PostgreSQL round trips.
2. Pickling and copying large subtree caches.
3. PostgreSQL result processing and trie merging.
4. Beam-state deep copies.
5. Repeated tokenizer decoding and Python sequence handling.
6. Full vocabulary mask allocation.

The actual percentage slowdown should be measured rather than inferred from code alone.

## Recommended improvements, in priority order

### 1. Measure the critical path

Instrument separate timings for:

- transformer forward pass,
- logits processor total time,
- cache lookup,
- PostgreSQL query,
- HTTP request,
- pickle serialization/deserialization,
- subtree merge,
- beam permutation.

Also record:

- cache hit rate,
- queries per generated fact,
- average subtree size,
- constrained versus unconstrained tokens,
- tokens per second.

### 2. Simplify `SimpleCache`

Use tuple keys and store in-process values directly:

```python
key = tuple(sequence)
cache[key] = (next_tokens, subtree)
```

Avoid pickle unless persistence or cross-process transport is actually required.

### 3. Stop caching complete subtree snapshots per prefix

Cache database results or shared immutable subtree objects. Avoid serializing the entire accumulated `state.subtree_cache` for every prefix.

### 4. Prefetch larger subtrees

At the first database lookup for a fact, load enough of the relevant subtree to cover subsequent token generation. This should reduce the number of synchronous queries per fact.

### 5. Add connection pooling

Use a PostgreSQL connection pool for API/server deployments. Avoid sharing one connection across concurrent requests.

### 6. Remove repeated full-sequence decoding

Track fact-generation phase explicitly rather than determining it by repeatedly decoding the entire sequence and counting delimiters.

### 7. Reduce Python sequence copying

Track sequence length and expected prefixes incrementally. Avoid repeated `tolist()` conversion and full-prefix equality comparisons where possible.

### 8. Reduce logits-mask allocation

Investigate in-place masking or reusable buffers, while preserving compatibility with the ordering and expectations of Transformers logits processors.

### 9. Optimize beam state handling

Share immutable index data between beams and use copy-on-write for mutable generation state instead of deep-copying complete trie structures.

## Suggested benchmark

Run the same prompts and model with:

1. unconstrained generation,
2. `DictIndex`,
3. PostgreSQL with cache disabled,
4. PostgreSQL with `SimpleCache`,
5. HTTP PostgreSQL with cache enabled.

Report at least:

```text
mean model-forward ms/token
mean constraint-processing ms/token
mean database ms/query
cache hit rate
queries per fact
end-to-end tokens/sec
first-request versus warmed-up tokens/sec
```

This will identify whether the current deployment is model-bound, CPU-bound, database-bound, or network-bound.

## Conclusion

The constrained generation design can be practical, particularly when the trie subtree is in memory and cache hits are high. However, the current PostgreSQL and `SimpleCache` implementation adds real latency to the decoding loop. The largest opportunity is to avoid synchronous per-prefix database work and eliminate repeated serialization of complete subtree-cache snapshots. After that, reducing Python/tokenizer overhead and mask allocation should improve throughput for in-memory and warmed-cache workloads.

## Second time addition
### Forced sequences
When forcing a certain sequence to be generated it should be faster (cause of parallel gpu operations) to directly feed the entire sequence to the LLM instead of feeding one token per time.
### Multi token prediction
Multi token prediction could be explored. Maybe in constrained setup it could be efficient and effective enough (e.g. comparing two possible path instead of the entire vocabulary could be easier)


## Implemented efficiency changes

This section records the changes that were applied on top of the analysis above,
with measurements from the two T4 GPUs in this workspace.

### 1. `SimpleCache` no longer stores string keys or re-pickles every lookup

`refactx/SimpleCache.py` now:

- keys entries by `tuple(sequence)` instead of `'.'.join(map(str, ...))`;
- stores the small `next_tokens` mapping directly (copied on insert/read);
- keeps the subtree snapshot as bytes and unpickles only that value.

The pickle round-trip for subtrees is intentional: in CPython it is several
times faster than `copy.deepcopy` for the nested list/dict trie representation,
and it isolates the cached snapshot from the mutable decode state. The
important change is that the cached snapshot is now canonical, so it is small
(see item 2). `SimpleCache` also gained `stats()`, `flush()`, `len()`, and a
proper `__bool__` so an empty but configured cache is still treated as enabled.

### 2. PostgreSQL/HTTP caches store canonical per-prefix subtrees

Previously `PostgresTrieIndex` cached `state.subtree_cache` after every lookup.
That object accumulates sibling branches explored earlier, so each cache entry
contained a growing, overlapping snapshot of the trie. The index now:

- parses a query result into a fresh trie rooted at the queried prefix;
- merges those rows into a single canonical subtree for that prefix;
- gives the decode state that subtree and caches an independent copy.

On a cache hit the state is updated with the cached canonical subtree instead of
a full accumulated snapshot. This also removes the spurious "number of leaves
does not match" warnings that appeared when sibling branches were merged into
the same cached tree.

Synthetic growth benchmark (300 sibling prefixes, 4-token switch):

| Version | Cache entries | Total cached size | Time |
|---|---:|---:|---:|
| before | 300 | 0.66 MB | 13 ms |
| after | 300 | 0.034 MB | 4 ms |

Live PostgreSQL (`qwen36`, switch parameter 7, read-only):

```text
cold query      0.4-1.2 ms/query (varies with DB/network cache)
warm cache hit  0.012 ms/query
```

The benchmark is reproducible with:

```bash
python utils/benchmark_efficiency.py cache
python utils/benchmark_efficiency.py postgres --table qwen36 --samples 100
```

### 3. Cheaper constrained logits processing

`refactx/generate.py` now:

- allocates no mask at all when no row is constrained (the common case during
  free-form reasoning);
- uses a single reusable `(1, vocab)` mask row instead of a full
  `scores`-shaped `torch.zeros_like` allocation per step;
- applies the mask in place and preserves the model's relative scores for
  allowed tokens;
- updates per-row state with an O(1) fast path (length + tail check) and only
  materializes the full `tolist()` sequence for constrained rows or when a beam
  actually changed the prefix;
- tracks the monotonic "past subject/relation" and "inside object" conditions
  once instead of decoding the whole prefix on every constrained token.

Constrained-processor micro-benchmark (`Qwen3.5-0.8B` tokenizer, batch 8,
56 decode steps, one constrained row):

| Version | Per decode step | Throughput |
|---|---:|---:|
| before | 12.7 ms | 546 tok/s |
| after | 11.2 ms | 579 tok/s |

Reproduce with:

```bash
python utils/benchmark_efficiency.py processor
```

### 4. Training/evaluation efficiency knobs

`utils/run_trl.py` now:

- enables the KV cache for rollouts and validation (`generation_config.use_cache`)
  while leaving `config.use_cache=False` for gradient checkpointing;
- temporarily re-enables the KV cache around the custom validation
  `model.generate()` call and restores the previous value;
- exposes `attn_implementation` (`--attn-implementation` or
  `REFACTX_ATTN_IMPLEMENTATION`) so SDPA/FlashAttention can be selected
  explicitly;
- exposes `dataset_num_proc` (`--dataset-num-proc`) to parallelize dataset
  formatting;
- exposes `use_generation_cache` (`--use-generation-cache/--no-use-generation-cache`).

The existing 0.8B mixed-index smoke run (`configs/trl_mix_smoke.json`) still
completes one GSPO step and initial/final validation with both `qwen36` and
`qwen36fbwikipops` tables and zero routing misses.

### 5. Tests

Added `tests/test_efficiency.py` covering:

- tuple-key/direct-value cache semantics and isolation between cache reads;
- hit/miss statistics and `flush()`;
- the `__bool__` fix for an empty but configured cache;
- canonical subtree caching (no sibling accumulation, safe cache hits);
- serving longer prefixes from the warmed subtree without another query.

The relevant suites pass:

```text
tests/test_features.py                45 passed
tests/test_routed_index.py             6 passed
tests/test_run_trl_reward.py          33 passed
tests/test_run_trl_tool_metrics.py     7 passed
tests/test_tool_callbacks.py           4 passed
tests/test_efficiency.py              11 passed
```

`tests/test_ctrie.py` still fails because it imports a `DictIndex` attribute
that the current `refactx` package does not export; this is pre-existing and
unrelated to these changes. `tests/test_unicode.py` requires a Wikipedia dump
path argument and is not collectable in a plain test run.

## Batched and vLLM execution

### Optional parallel batch constraints (HuggingFace path)

`ConstrainedLogitsProcessor` now accepts `parallel_workers`. When more than one
row is constrained and the index is safe for concurrent reads, per-row
constrain/mask work runs in a thread pool and each row writes to its own row of
a reusable mask matrix. In-memory `DictIndex`/`RoutedIndex` instances are marked
`is_thread_safe=True`; PostgreSQL and HTTP indexes are not, so parallel workers
are automatically disabled for them (shared connection/client). The option is
off by default and can be enabled with `--parallel-workers`/`parallel_workers`.

Measured effect is workload dependent (the Python index/tokenizer work is
GIL-bound): roughly 10-16% on a 4-row synthetic batch, neutral or slightly worse
at batch 8. For large batches, vLLM is the better parallel path.

### vLLM V1 adapter

Added `refactx/vllm_proc.py` (`RefactxLogitsProcessor`) and
`examples/vllm_offline.py`. The adapter:

- keeps one state machine per persistent vLLM batch row and handles
  added/removed/moved requests;
- reuses the same `FactGeneration`/`CountBranchesGeneration` logic as the HF
  path, including the sentinel and duplicate avoidance;
- masks only constrained rows, with a reusable mask row;
- supports per-request KB routing through `REFACTX_INDEX_TABLES` and
  `extra_args={"index_table": ...}`;
- opts in per request with `extra_args={"refactx": True}`, and records facts
  retrievable via `pop_facts(refactx_id)`;
- exposes `REFACTX_FORCE_CONSTRAINED=1` for benchmarks/tests.

A real vLLM engine run on a T4 successfully decoded a constrained triple and
produced the expected `</fact>` terminator. An 8-request forced-constrained batch
ran at ~77 output tok/s with all rows started with a valid trie prefix.

Environment notes for this workstation: vLLM 0.25 needs

```bash
export LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cu13/lib:\
/opt/conda/lib/python3.11/site-packages/PyNvVideoCodec:$LD_LIBRARY_PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

and the adapter must be imported before `vllm` (it patches the
`ALLOWED_LAYER_TYPES` rename expected by the installed transformers version).

### Constrained vs unconstrained throughput comparison

Added `utils/constrained_throughput.py`, which mirrors the paper comparison:
the same prompt/token budget is decoded once normally and once with constraints
always active, recording wall time between decode steps. On a T4 with
`Qwen/Qwen3.5-0.8B` and the `qwen36` PostgreSQL table:

| Batch | Max tokens | Unconstrained | Constrained | Ratio |
|---:|---:|---:|---:|---:|
| 1 | 128 | 18.9 tok/s | 17.3 tok/s | 0.92x |
| 4 | 128 | 75.7 tok/s | 67.1 tok/s | 0.89x |
| 1 | 64 | 18.9 tok/s | 14.6-15.5 tok/s | 0.79-0.82x |

The short 64-token run pays proportionally more fixed constraint setup; at 128
tokens the steady-state overhead is ~8-11% (always-constrained worst case).
Outputs are JSON with per-step `mean/p50/p95/max` timings for plotting, like
`notebooks/throughput.ipynb`.

```bash
python utils/constrained_throughput.py \
    --model Qwen/Qwen3.5-0.8B --table qwen36 --max-tokens 128 \
    --output logs/throughput_constrained.json
```

### Additional tests

`tests/test_vllm_proc.py` covers request bookkeeping, moves/removals, masking,
unconstrained-row passthrough, trigger switching and fact collection. It is
skipped automatically when `vllm` is unavailable. `tests/test_efficiency.py` now
also checks that parallel and sequential masking produce identical logits and
that non-thread-safe indexes disable parallelism.

```text
tests/test_efficiency.py              11 passed
tests/test_vllm_proc.py                7 passed (with vLLM importable)
```
