import os
from urllib.parse import urlparse
import redis
import pickle
from dotenv import load_dotenv

load_dotenv()

class RedisCache():
    def __init__(self, cache_db=None):
        redis_base_url = os.environ.get('REDIS_BASE_URL')
        redis_ssl = os.environ.get('REDIS_SSL', 'false').lower() == 'true'
        redis_ssl_ca_certs = os.environ.get('REDIS_SSL_CA_CERTS', '')

        if cache_db is None:
            os.environ.get('REDIS_CACHE_DB', '0')

        if redis_base_url and cache_db is None:
            raise ValueError('When using redis you must configure which db to use (e.g. 0 or 1).')

        redis_config = urlparse(redis_base_url)
        self.redis_connection = redis.Redis(
            host=redis_config.hostname,
            port=redis_config.port,
            password=redis_config.password,
            username=redis_config.username,
            db=cache_db,
            ssl=redis_ssl,
            ssl_ca_certs=redis_ssl_ca_certs,
        )

    def encode_sequence(self, sequence):
        # TODO use base64 or similar for improve space efficiency
        return '.'.join(map(str, sequence))

    def add(self, sequence, next_tokens, subtree_cache):
        redis_key = self.encode_sequence(sequence)
        redis_value = pickle.dumps([next_tokens, subtree_cache], protocol=5)
        self.redis_connection.set(redis_key, redis_value)

    def next_tokens(self, sequence, **kwargs):
        redis_key = self.encode_sequence(sequence)
        redis_value = self.redis_connection.get(redis_key)
        if redis_value:
            _next_tokens, subtree_cache = pickle.loads(redis_value)
        else:
            _next_tokens = None
            subtree_cache = None
        return _next_tokens, subtree_cache

def __init__(cache_db):
    return RedisCache(cache_db)