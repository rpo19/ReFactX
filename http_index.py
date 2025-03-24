from dotenv import load_dotenv
import os

class IndexConfigException(Exception):
    pass

class IndexConfig():
    def __init__(self, postgresql_table, switch_parameter, rootkey, end_of_triple,
                model_name=None,
                base_url=None,
                rootcert=None
                ):
        load_dotenv()

        self.model_name = model_name # the db depends from the tokenizer
        if base_url is None:
            base_url = os.environ.get('HTTP_BASE_URL')
        if rootcert is None:
            rootcert = os.environ.get('HTTP_ROOTCERT')
        self.base_url = base_url
        self.rootcert = rootcert
        self.postgresql_table = postgresql_table
        self.switch_parameter = switch_parameter
        self.rootkey = rootkey # decided at injestion time
        self.end_of_triple = end_of_triple # each triple must finish with the same token (e.g. "." after space)
        self.index = HTTPPostgresTrieIndex(
            rootkey = self.rootkey,
            switch_parameter = self.switch_parameter,
            table_name = self.postgresql_table,
            end_of_triple = self.end_of_triple,
            base_url = self.base_url,
            rootcert = self.rootcert,
            )

        self.skip_serialize = set(['skip_serialize', 'index'])

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])


from ctrie import HTTPPostgresTrieIndex


index_config = IndexConfig(
    model_name = 'Qwen/Qwen2.5-1.5B-Instruct',
    # postgresql_url=None, # from dotenv
    # redis_url=None, # from dotenv
    postgresql_table = 'ctriev5qwen',
    switch_parameter = 7,
    rootkey = 200000,
    end_of_triple = 659,
    redis_db=0,
)
