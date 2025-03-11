from ctrie import PostgresTrieIndex
import psycopg
import redis
from dotenv import load_dotenv
import os

class IndexConfigException(Exception):
    pass

class IndexConfig():
    def __init__(self, postgresql_table, switch_parameter, rootkey, end_of_triple,
                model_name=None,
                postgresql_base_url=None, # if None loads from dotenv
                postgres_db='postgres',
                autocommit=True,
                redis_base_url=None, # if None loads from dotenv
                redis_db=0,
                ):
        load_dotenv()

        self.model_name = model_name # the db depends from the tokenizer
        if postgresql_base_url is None:
            postgresql_base_url = os.environ.get('POSTGRES_BASE_URL')
        if postgresql_base_url is None:
            raise IndexConfigException('postgres_url is None.')
        self.postgresql_url = postgresql_base_url + postgres_db
        self.autocommit = autocommit
        if redis_base_url is None:
            redis_base_url = os.environ.get('REDIS_BASE_URL')
        self.redis_url = redis_base_url + str(redis_db)
        self.redis_connection = redis.Redis().from_url(self.redis_url) if redis_base_url else None
        self.postgresql_connection = psycopg.connect(self.postgresql_url, autocommit = self.autocommit)
        self.postgresql_table = postgresql_table
        self.switch_parameter = switch_parameter
        self.rootkey = rootkey # decided at injestion time
        self.end_of_triple = end_of_triple # each triple must finish with the same token (e.g. "." after space)
        self.index = PostgresTrieIndex(
            rootkey = self.rootkey,
            postgresql_connection = self.postgresql_connection,
            switch_parameter = self.switch_parameter,
            table_name = self.postgresql_table,
            end_of_triple = self.end_of_triple,
            redis_connection = self.redis_connection,
            )

        self.skip_serialize = set(['skip_serialize', 'postgresql_connection', 'redis_connection', 'index'])

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])
