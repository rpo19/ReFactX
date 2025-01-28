from ctrie import PostgresTrieIndex, ConstrainedLogitsProcessor
import psycopg

class Index():
    def __init__(self):
        self.postgresql_connection = psycopg.connect('postgres://postgres:secret@10.0.0.118:5432/postgres', autocommit=False)
        self.postgresql_table = "ctrie"
        self.switch_parameter = 8
        self.rootkey = 60000
        self.index = PostgresTrieIndex(
            rootkey = self.rootkey,
            postgresql_connection = self.postgresql_connection,
            switch_parameter = self.switch_parameter,
            table_name = self.postgresql_table)
