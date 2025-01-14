# CREATE TABLE my_table (
#     id BIGINT, -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     children BYTEA NOT NULL
# );

import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys
from more_itertools import peekable
import itertools

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
root = int(sys.argv[3]) # number that not collides with the vocab (check max vocab) # maybe 60000
batch_size = int(sys.argv[4])
total_rows = int(sys.argv[5]) if len(sys.argv) > 5 else None

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

def batch_append(trie, token_ids):
    level = trie
    for token_id in token_ids:
        if token_id not in level:
            level[token_id] = {}

        level = level[token_id]

def get_rows(trie, rootkey):
    # iterative depth first traversal with a stack
    key = rootkey
    stack = [(key, trie)]
    while len(stack) > 0:
        key, level = stack.pop()
        children = level.keys()
        for child in children:
            stack.append((key + tken(child), level[child]))
        if len(children) > 0:
            # skip adding empty keys to save space
            yield key, b''.join(map(tken, children))


enroot = tken(root)
tbar_update = batch_size
count = 0

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE ctrie;")
        with cur.copy("COPY ctrie (key, children) FROM STDIN WITH (FREEZE)") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    batch = {}

                    while True:
                        try:

                            # Load each pickled object from the bz2 file
                            array = pickle.load(bz2file)
                            batch_append(batch, array)

                            count += 1

                            if count % batch_size == 0:
                                # batch on number or rows processed
                                for row in get_rows(batch, enroot):
                                    copy.write_row(row)
                                # reset batch
                                batch = {}

                            if count % tbar_update == 0:
                                pbar.n = count
                                pbar.refresh()

                        except EOFError:
                            print('Reached end of file.')
                            break  # End of file reached
    conn.commit()

# CREATE INDEXes and PKEY
