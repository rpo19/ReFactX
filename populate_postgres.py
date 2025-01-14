# CREATE TABLE ctrie (
#     id BIGINT GENERATED ALWAYS AS IDENTITY, -- Automatically generates unique values for id -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     children BYTEA, -- either children or subtree must be present (todo add constraint?)
#     subtree BYTEA,
# );


import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
root = int(sys.argv[3]) # number that not collides with the vocab (check max vocab) # maybe 60000
batch_size = int(sys.argv[4])
switch_parameter = int(sys.argv[5]) # after N token save all the branch to the leaf
total_rows = int(sys.argv[6]) if len(sys.argv) > 6 else None

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

def get_rows(trie, rootkey, switch_parameter):
    # iterative depth first traversal with a stack
    key = rootkey
    stack = [(0, key, trie)]
    while len(stack) > 0:
        level, key, level = stack.pop()
        children = level.keys()
        if level >= switch_parameter:
            yield key, b''.join(map(tken, children)), pickle.dumps(level, protocol=5) # highest protocol for best efficiency. supported by python 3.8
        else:
            for child in children:
                stack.append((level + 1, key + tken(child), level[child]))
            if len(children) > 0:
                # skip adding empty keys to save space
                yield key, b''.join(map(tken, children)), None


enroot = tken(root)
tbar_update = batch_size
count = 0

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE ctrie;")
        with cur.copy("COPY ctrie (key, children, subtree) FROM STDIN WITH (FREEZE)") as copy:
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
