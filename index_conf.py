from ctrie import PostgresTrieIndex
import psycopg

class Index():
    def __init__(self):
        self.postgresql_url = 'postgres://postgres:secret@10.0.0.118:5432/postgres'
        self.autocommit = True
        self.postgresql_connection = psycopg.connect(self.postgresql_url, autocommit = self.autocommit)
        self.postgresql_table = "ctrieV3"
        self.switch_parameter = 6
        self.rootkey = 150000
        self.end_of_triple = 662
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
