from ctrie import PostgresTrieIndex
import psycopg

class Index():
    def __init__(self):
        self.postgresql_url = 'postgres://postgres:secret@10.0.0.118:5432/postgres'
        self.autocommit = True
        self.postgresql_connection = psycopg.connect(self.postgresql_url, autocommit = self.autocommit)
        self.postgresql_table = "ctriev3qwen"
        self.switch_parameter = 7
        self.rootkey = 200000 # decided at injestion time
        self.end_of_triple = 659 # each triple must finish with the same token (e.g. "." after space)
        self.index = PostgresTrieIndex(
            rootkey = self.rootkey,
            postgresql_connection = self.postgresql_connection,
            switch_parameter = self.switch_parameter,
            table_name = self.postgresql_table,
            end_of_triple = self.end_of_triple)

        self.skip_serialize = set(['skip_serialize', 'postgresql_connection', 'index'])

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])
