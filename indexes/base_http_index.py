from dotenv import load_dotenv
import os
from refactx import HTTPPostgresTrieIndex
import importlib

class IndexConfigException(Exception):
    pass

class IndexConfig():
    def __init__(self, postgresql_table, switch_parameter, rootkey, end_of_triple,
                model_name=None,
                base_url=None,
                rootcert=None,
                cache=None, # if None loads from dotenv
                cache_db=0,
                ):
        load_dotenv()

        self.model_name = model_name # the db depends from the tokenizer
        if base_url is None:
            base_url = os.environ.get('HTTP_BASE_URL')
        self.base_url = base_url
        if rootcert is None:
            rootcert = os.environ.get('HTTP_ROOTCERT')
        if rootcert and rootcert.lower() == 'false':
            rootcert = False
            import urllib3
            urllib3.disable_warnings()

        self.rootcert = rootcert
        self.postgresql_table = postgresql_table
        self.switch_parameter = switch_parameter
        self.rootkey = rootkey # decided at injestion time
        self.end_of_triple = end_of_triple # each triple must finish with the same token (e.g. "." after space)

        if cache is None and os.environ.get('CACHE') is not None:
            cache_class = os.environ.get('CACHE')
            cache_module = importlib.import_module(cache_class)
            self.cache = cache_module.__init__(cache_db)
        else:
            self.cache = cache

        self.index = HTTPPostgresTrieIndex(
            rootkey = self.rootkey,
            switch_parameter = self.switch_parameter,
            table_name = self.postgresql_table,
            end_of_triple = self.end_of_triple,
            base_url = self.base_url,
            cache = self.cache,
            rootcert = rootcert
            )

        self.skip_serialize = set(['skip_serialize', 'index', 'cache'])

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

