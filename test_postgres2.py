# CREATE TABLE ctrieV2 (
#     id BIGINT GENERATED ALWAYS AS IDENTITY, -- later make PRIMARY KEY
#     key INT[] NOT NULL,
#     children INT[],
#     numleaves INT,
#     childrenleaves INT[],
#     subtree BYTEA
# );

import psycopg
import pickle
from transformers import AutoTokenizer
import sys
import random
import time
import json
import pickle

from ctrie import PostgresTrieIndex


print('Start.')

postgres_connection = sys.argv[1] # 'postgres://postgres:secret@host:5432/postgres'
table_name = sys.argv[2]
rootkey = int(sys.argv[3])
end_of_triple = sys.argv[4]
model_name = sys.argv[5]
switch_parameter = int(sys.argv[6])
random_seed = int(sys.argv[7])
initial_tokens = sys.argv[8] if len(sys.argv) > 8 else ''
if initial_tokens == 'json' and len(sys.argv) > 9:
    initial_tokens = json.loads(sys.argv[9])
    assert isinstance(initial_tokens, list)

random.seed(random_seed)

class TimeMeasure:
    def __init__(self, tag='default', verbose=False, outfile=sys.stderr):
        self.tag = tag
        self.verbose = verbose
        self.start_time = None
        self.duration = None
        self.outfile = outfile

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.verbose:
            print(self.tag, 'duration:', self.duration_ms(self.duration), file=self.outfile)

    def duration_ms(self, duration):
        return "{:.2f} ms".format(duration * 1000)

with TimeMeasure(tag='Loading tokenizer', verbose=True) as tm:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

def merge(dst, src):
    for key in src:
        if key in dst:
            # sum numleaves
            dst[key][0] += src[key][0]
            merge(dst[key][1], src[key][1])
        else:
            # If the key exists only in `src`, the value from the `src` object will be used.
            dst[key] = src[key]


def choose(children, initial_tokens):
    if len(initial_tokens) > 0:
        choice = initial_tokens.pop(0)
        assert choice in children
    else:
        choice = random.choice(children)
    return choice

if isinstance(initial_tokens, str):
    initial_tokens = tokenizer(initial_tokens)['input_ids']

conn = psycopg.connect(postgres_connection, autocommit=False)

index = PostgresTrieIndex(rootkey=rootkey,
                            postgresql_connection=conn,
                            switch_parameter=switch_parameter,
                            table_name=table_name,
                            end_of_triple=end_of_triple)


sentence = []

while True:
    with TimeMeasure(tag=f'Query {len(sentence)}', verbose=True) as tm:
        possible_tokens = index.next_tokens(sentence)

    possible_tokens = list(possible_tokens.keys()) if possible_tokens else []

    if len(possible_tokens) > 0:
        next_token = choose(possible_tokens, initial_tokens)
        print(tokenizer.decode(next_token))
        sentence.append(next_token)
    else:
        print('.')
        break

print(tokenizer.decode(sentence))



