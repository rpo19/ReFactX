from cachetools import LRUCache
import pickle
import os

class SimpleCache:
    def __init__(self, cache_db, maxsize=None):
        if maxsize is None:
            maxsize = int(os.environ.get('POSTGRES_CACHE_MAXSIZE', 1024))
        self.cache = LRUCache(maxsize=maxsize)

    def encode_sequence(self, sequence):
        # TODO use base64 or similar for improve space efficiency
        return '.'.join(map(str, sequence))

    def add(self, sequence, next_tokens, subtree_cache):
        key = self.encode_sequence(sequence)
        value = pickle.dumps([next_tokens, subtree_cache], protocol=5)
        self.cache[key] = value

    def next_tokens(self, sequence, **kwargs):
        key = self.encode_sequence(sequence)
        value = self.cache.get(key)
        if value:
            _next_tokens, subtree_cache = pickle.loads(value)
        else:
            _next_tokens = None
            subtree_cache = None
        return _next_tokens, subtree_cache

def __init__(cache_db):
    return SimpleCache(cache_db)
