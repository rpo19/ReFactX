# CREATE TABLE my_table (
#     id BIGINT, -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     child BYTEA NOT NULL,
#     child_id BIGINT NOT NULL
# );

import psycopg
import pickle
from mergedeep import merge
from transformers import AutoTokenizer
import sys
import random
import time
import json

print('Start.')

postgres_connection = sys.argv[1] # 'postgres://postgres:secret@host:5432/postgres'
model_name = sys.argv[2]
initial_tokens = sys.argv[3] if len(sys.argv) > 3 else ''
if initial_tokens == 'json' and len(sys.argv) > 4:
    initial_tokens = json.loads(sys.argv[4])
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

def choose(children, initial_tokens):
    if len(initial_tokens) > 0:
        choice = initial_tokens.pop(0)
        if isinstance(choice, bytes) and isinstance(next(iter(children)), int):
            choice = tkde(choice)
        assert choice in children
    else:
        choice = random.choice(children)
    return choice

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

if isinstance(initial_tokens, str):
    initial_tokens = tokenizer(initial_tokens)['input_ids']

initial_tokens = list(map(tken, initial_tokens))

conn = psycopg.connect(postgres_connection, autocommit=False)
cur = conn.cursor()

rootkey = 60000
enroot = tken(rootkey)

sentence = [enroot]

while True:
    with TimeMeasure(tag=f'Query {len(sentence)}', verbose=True) as tm:
        cur.execute('SELECT children, subtree FROM ctrie WHERE key = %s;', (b''.join(sentence),))
        res = cur.fetchall()

    if len(res) > 0:
        exploded_children = set()
        map_subtree = {}
        for i, children in enumerate(res):
            children = children[0]
            splitted_children = set(children[i:i+2] for i in range(0, len(children), 2))
            for child in splitted_children:
                if child not in map_subtree:
                    map_subtree[child] = set()
                map_subtree[child].add(i)
            exploded_children.update(splitted_children)

        next_token = choose(list(exploded_children), initial_tokens)

        sentence.append(next_token)

        print(list(map(tkde,sentence)))

        corresponding_rows = map_subtree[next_token]
        merged_subtree = {}
        for _, bsubtree in [res[i] for i in corresponding_rows]:
            # if the chosen token has the subtree in the db
            # load the subtree in memory and go on
            if bsubtree is not None:
                subtree = pickle.loads(bsubtree)
                merge(merged_subtree, subtree)

        if len(merged_subtree) > 0:
            with TimeMeasure(tag=f'Subtree generation', verbose=True) as tm:
                 # continue the generation from the subtree
                 # from now on the token_ids are already int. no need to tkde
                 desentence = list(map(tkde,sentence))
                 print(tokenizer.decode(desentence))
                 print('.\nReached the switch level at len {}. Proceeding with the in-memory sub-tree.\n'.format(len(sentence)))
                 level = merged_subtree[tkde(next_token)]
                 children = list(level.keys())
                 while len(children) > 0:
                     next_token = choose(children, initial_tokens)
                     level = level[next_token]
                     children = list(level.keys())

                     desentence.append(next_token)

                     print(desentence)

                 print('+')
                 break

    else:
        print('.')
        break

print(tokenizer.decode(desentence))



