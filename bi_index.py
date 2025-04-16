from ctrie import DictIndex
from zipfile import ZipFile
import os
import pandas as pd
from base_dataset_config import QADataset
from tqdm import trange
from transformers import AutoTokenizer

class IndexConfigException(Exception):
    pass

class IndexConfig():
    def __init__(self, model_name, end_of_triple_str=" .", prefix=" ", batch_size=100, add_special_tokens=False):
        self.model_name = model_name # the db depends from the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if not self.tokenizer.is_fast:
            print('WARNING: tokenizer is not fast.')
        self.end_of_triple_str = end_of_triple_str # each triple must finish with the same token (e.g. "." after space)
        self.prefix = prefix
        self.end_of_triple = self.tokenizer(self.end_of_triple_str)['input_ids'][-1] # end_of_triple token

        self.index = DictIndex(self.end_of_triple)

        self.batch_size = batch_size
        self.add_special_tokens = add_special_tokens

        self.skip_serialize = set(['skip_serialize', 'index', 'tokenizer'])

    def load_from_path(self, dataset_path):
        actual_path = QADataset.get_dataset_path(dataset_path)
        bi_data_path2 = os.environ.get('BI_DATA_PATH2')
        if bi_data_path2 is None:
            raise Exception('Missing env var BI_DATA_PATH2.')
        with ZipFile(actual_path) as myzip:
            with myzip.open('anony_triples_v0.csv', pwd=bi_data_path2.encode()) as myfile:
                self.raw_triples = pd.read_csv(myfile)

        self.transform_triples()
        self.tokenize_triples()
        

    def format_triple(self, triple):
        verb = '{prefix}<{head}> <{relation}> <{tail}>{end_of_triple}'.format(
            prefix=self.prefix,
            end_of_triple=self.end_of_triple_str,
            **triple
            )
        return verb

    def transform_triples(self):
        self.verbalized_triples = list(map(self.format_triple, self.raw_triples.to_dict(orient='records')))

    def batch_append(self, nested_token_ids):
        for sequence in nested_token_ids:
            self.index.add(sequence)

    def tokenize_triples(self):
        for batch_start in trange(0, len(self.verbalized_triples), self.batch_size):
            batch_end = batch_start + self.batch_size
            if batch_end >= len(self.verbalized_triples):
                batch_end = len(self.verbalized_triples)

            batch = self.verbalized_triples[batch_start:batch_end]
            ids = self.tokenizer(batch, add_special_tokens=self.add_special_tokens)['input_ids']
            self.batch_append(ids)

    def clean(self):
        del self.raw_triples
        del self.verbalized_triples

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

