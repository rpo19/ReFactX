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

postgres_connection = sys.argv[1] # 'postgres://postgres:secret@host:5432/postgres'
model_name = sys.argv[2]
initial_tokens = sys.argv[3] if len(sys.argv) > 3 else ''

tokenizer = AutoTokenizer.from_pretrained(model_name)

def choose(children, initial_tokens):
    if len(initial_tokens) > 0:
        choice = initial_tokens.pop(0) 
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

initial_tokens = list(map(tken, tokenizer(initial_tokens)['input_ids']))

conn = psycopg.connect(postgres_connection, autocommit=False)
cur = conn.cursor()

rootkey = 60000
enroot = tken(rootkey)

sentence = [enroot]

while True:
    
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
            exploded_children.update(set(children[i:i+2] for i in range(0, len(children), 2)))
        
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
            # continue the generation from the subtree
            # from now on the token_ids are already int. no need to tkde
            desentence = list(map(tkde,sentence))
            print(tokenizer.decode(desentence))
            print('.\nReached the switch level. Proceeding with the in-memory sub-tree.\n')
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



