from ctrie import DictIndex
from zipfile import ZipFile
import os
import pandas as pd
from base_dataset_config import QADataset
from tqdm import trange
from transformers import AutoTokenizer

class IndexConfigException(Exception):
    pass

# TODO should I add preds in the prompt?
# 'REACHABLE', 'OWN', 'ROLE', 'CONTROL', 'ULTIMATE_CONTROL', 'HAS_HOLDINGS'
# ask to modify template to more real-world questions: no entities but "which person or company is connected to...?"
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

        # X -> has role in A -> cotrolled by -> B 
        # A -> ultimate controller B
        '''
        |-> “X -> has role in A ->  controlled by B”
|-> “A -> ultimate controller B : quanti società A controlla direttamente.”
|-> “A -> ultimate controller B : quanti società B controlla direttamente.”
Tu
10:57
quante società sono controllate dall'ultimate controller di A?
Tu
11:00
chi ricopre un ruolo in società controllate da B?

|-> “X -> has role in A ->  controlled by B”
|-> “A -> ultimate controller B : quanti società A controlla direttamente.”
|-> “A -> ultimate controller B : quanti società B controlla direttamente.”
Tu
10:57
quante società sono controllate dall'ultimate controller di A?
Tu
11:00
chi ricopre un ruolo in società controllate da B?
'''

        # TODO explain them in a prompt? # experiment both with and without
        self.relmapping = {
                'role': dict(mappings=['has role in']), # non implica altre relazioni: e.g., CEO pur non avendo share # anche azienda ruolo in altra aziende.
            'own': dict(mappings=['owns'], inverse=['owned by']), # ha % di share
            'control': dict(mappings=['controls'], inverse=['controlled by']), # controls implica ownership # ultimate control -> coontrol
            'ultimate_control': dict(mappings=['ultimately controls', 'ultimate controller of', 'controls ultimately'], inverse=['ultimately controlled by', 'controlled ultimately by']),
            'has_holdings': dict(mappings=['has holdings in']), # non necessariamente shares, ma ha possibilità di votare o altre. non necessariamente collegate a own
            'reachable': dict(mappings=['connected to', 'reachable from'], reflexive=True), # non solo connesse con altre relazioni (grafo già tutto connesso)
        }

        self.skip_serialize = set(['skip_serialize', 'index', 'tokenizer', 'raw_triples', 'verbalized_triples'])

    def enrich_relations(self):
        for triple in self.raw_triples:
            mappings = self.relmapping[triple['relation']]
            assert isinstance(mappings['mappings'], list)
            for mapping in mappings['mappings']:
                yield {'head': triple['head'], 'tail': triple['tail'], 'relation': mapping}
                if mappings.get('reflexive'):
                    yield {'head': triple['tail'], 'tail': triple['head'], 'relation': mapping}
            inverse_mappings = mappings.get('inverse', [])
            assert isinstance(inverse_mappings, list)
            for inverse_mapping in inverse_mappings:
                yield {'head': triple['tail'], 'tail': triple['head'], 'relation': inverse_mapping}

    def load_from_path(self, dataset_path, enrich=True):
        actual_path = QADataset.get_dataset_path(dataset_path)
        bi_data_path2 = os.environ.get('BI_DATA_PATH2')
        if bi_data_path2 is None:
            raise Exception('Missing env var BI_DATA_PATH2.')
        with ZipFile(actual_path) as myzip:
            with myzip.open('anony_triples_v0.csv', pwd=bi_data_path2.encode()) as myfile:
                self.raw_triples = pd.read_csv(myfile)

        # lowercase relations
        self.raw_triples['relation'] = self.raw_triples['relation'].apply(lambda x: x.lower())
        self.raw_triples = self.raw_triples.to_dict(orient='records')
        if enrich:
            self.raw_triples = list(self.enrich_relations())
        self.transform_triples()
        self.tokenize_triples()        

    def format_triple(self, triple):
        head = triple['head']
        tail = triple['tail']
        relation = triple['relation']
        verb = '{prefix}<{head}> <{relation}> <{tail}>{end_of_triple}'.format(
            prefix=self.prefix,
            end_of_triple=self.end_of_triple_str,
            head=head,
            tail=tail,
            relation=relation,
            )
        return verb

    def transform_triples(self):
        self.verbalized_triples = list(map(self.format_triple, self.raw_triples))

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

