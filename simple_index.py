from ctrie import DictIndex
import gzip
from base_dataset_config import QADataset
from transformers import AutoTokenizer
from tqdm import trange

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

    def load_from_path(self, dataset_path='simple_index.txt.gz', enrich=True):
        try:
            actual_path = QADataset.get_dataset_path(dataset_path)
        except:
            actual_path = dataset_path
        with gzip.open(actual_path, 'rt') as myfile:
            self.verbalized_triples = myfile.readlines()
        self.verbalized_triples = list(map(lambda x: x.replace('\n',''), self.verbalized_triples))
        self.tokenize_triples()

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
        del self.verbalized_triples

    def __iter__(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

