"""In-process LRU cache for PostgreSQL index lookups.

The cache stores database results directly.  Token sequences are used as
``tuple`` keys (rather than joined strings), and the caller receives a mutable
copy of the cached ``next_tokens`` mapping.  Only the subtree snapshot is
serialized: this is faster in CPython than deep-copying the nested trie and
keeps the cached object isolated from the decode state that owns the original.

Subtree snapshots are canonical per database prefix (see
``PostgresTrieIndex._next_tokens_from_postgresql``), so adjacent cache entries
no longer contain the same accumulated tree over and over.
"""

from cachetools import LRUCache
import os
import pickle


class SimpleCache:
    def __init__(self, maxsize=None, copy_on_get=True):
        if maxsize is None:
            maxsize = int(os.environ.get('POSTGRES_CACHE_MAXSIZE', 1024))
        self.cache = LRUCache(maxsize=maxsize)
        self.copy_on_get = copy_on_get
        self.hits = 0
        self.misses = 0
        self.adds = 0

    def encode_sequence(self, sequence):
        # tuple(sequence) is O(n) in the prefix length but, unlike the previous
        # '.'.join(map(str, ...)), it does not allocate intermediate strings.
        return tuple(sequence)

    def add(self, sequence, next_tokens, subtree_cache):
        key = self.encode_sequence(sequence)
        # The next-token mapping is small; copy it so later caller mutations
        # cannot change the cached value.
        if next_tokens is not None:
            next_tokens = dict(next_tokens)
        # Subtrees are stored as bytes.  A pickle round-trip is several times
        # faster in CPython than ``copy.deepcopy`` for the nested list/dict
        # trie representation, and it guarantees the cached snapshot cannot be
        # mutated by the decode state that owns the original.
        if self.copy_on_get and subtree_cache is not None:
            subtree_cache = pickle.dumps(subtree_cache, protocol=5)
        self.cache[key] = (next_tokens, subtree_cache)
        self.adds += 1

    def next_tokens(self, sequence, **kwargs):
        key = self.encode_sequence(sequence)
        value = self.cache.get(key)
        if value is None:
            self.misses += 1
            return None, None

        self.hits += 1
        _next_tokens, subtree_cache = value
        if self.copy_on_get:
            if _next_tokens is not None:
                _next_tokens = dict(_next_tokens)
            if subtree_cache is not None:
                subtree_cache = pickle.loads(subtree_cache)
        return _next_tokens, subtree_cache

    def flush(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)

    def __bool__(self):
        # ``if index.cache:`` must stay true even when the LRU is empty;
        # otherwise ``__len__`` would make a configured cache look disabled.
        return True

    def stats(self):
        lookups = self.hits + self.misses
        return {
            'entries': len(self.cache),
            'maxsize': self.cache.maxsize,
            'hits': self.hits,
            'misses': self.misses,
            'adds': self.adds,
            'hit_rate': (self.hits / lookups) if lookups else 0.0,
        }


def __init__(cache_db):
    return SimpleCache(cache_db)
