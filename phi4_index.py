from ctrie import PostgresTrieIndex
import psycopg
import redis

class Index():
    def __init__(self):
        self.postgresql_url = 'postgres://postgres:secret@10.0.0.118:5432/postgres'
        self.autocommit = True
        self.redis_url = 'redis://default:12345678@10.0.0.118:6379/0'
        self.redis_connection = redis.Redis().from_url(self.redis_url)
        self.postgresql_connection = psycopg.connect(self.postgresql_url, autocommit = self.autocommit)
        self.postgresql_table = "ctriev3"
        self.switch_parameter = 7
        self.rootkey = 150000
        self.end_of_triple = 662
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
