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

sys.path.insert(0, '.')
from mergedeep import merge


print('Start.')

postgres_connection = sys.argv[1] # 'postgres://postgres:secret@host:5432/postgres'
table_name = sys.argv[2]
rootkey = int(sys.argv[3])
model_name = sys.argv[4]
initial_tokens = sys.argv[5] if len(sys.argv) > 5 else ''
if initial_tokens == 'json' and len(sys.argv) > 6:
    initial_tokens = json.loads(sys.argv[6])
    assert isinstance(initial_tokens, list)

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
cur = conn.cursor()

sentence = [rootkey]

while True:
    with TimeMeasure(tag=f'Query {len(sentence)}', verbose=True) as tm:
        cur.execute(f'SELECT children, subtree, numleaves, childrenleaves FROM {table_name} WHERE key = %s;', (sentence,))
        res = cur.fetchall()

    if len(res) > 0:
        exploded_children = set()
        map_subtree = {}
        for i, (children, subtree, numleaves, childrenleaves) in enumerate(res):
            assert numleaves == sum(childrenleaves)
            children = set(children)
            for child in children:
                if child not in map_subtree:
                    map_subtree[child] = set()
                map_subtree[child].add(i)
            exploded_children.update(children)

        next_token = choose(list(exploded_children), initial_tokens)

        sentence.append(next_token)

        corresponding_rows = map_subtree[next_token]
        merged_subtree = {}
        for _, subtree, numleaves, _ in [res[i] for i in corresponding_rows]:
            # if the chosen token has the subtree in the db
            # load the subtree in memory and go on
            if subtree is not None:
                subtree = pickle.loads(subtree)
                assert numleaves == sum(num for num,_ in subtree.values())
                merge(merged_subtree, subtree)

        if len(merged_subtree) > 0:
            with TimeMeasure(tag=f'Subtree generation', verbose=True) as tm:
                 # continue the generation from the subtree
                 print(tokenizer.decode(sentence))
                 print('.\nReached the switch level at len {}. Proceeding with the in-memory sub-tree.\n'.format(len(sentence)))
                 level = merged_subtree[next_token]
                 children = list(level[1].keys())
                 while len(children) > 0:
                     next_token = choose(children, initial_tokens)
                     level = level[1][next_token]
                     children = list(level[1].keys())

                     sentence.append(next_token)

                     print(sentence)

                 print('+')
                 break

    else:
        print('.')
        break

print(tokenizer.decode(sentence))



