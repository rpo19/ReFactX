# CREATE TABLE my_table (
#     id BIGINT, -- later make it PRIMARY KEY
#     key BYTEA NOT NULL,
#     child BYTEA NOT NULL,
#     child_id BIGINT NOT NULL
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
total_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

def explode(token_ids, root, peekable_id):
    key = root
    for token in token_ids:
        entoken = tken(token)
        row = (next(peekable_id), key, entoken, peekable_id.peek())
        key += entoken
        yield row

enroot = tken(root)

peekable_id = peekable(itertools.count())

count = 0

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE ctrie;")
        with cur.copy("COPY ctrie (id, key, child, child_id) FROM STDIN WITH (FREEZE)") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    while True:
                        try:
                            # Load each pickled object from the bz2 file
                            array = pickle.load(bz2file)
                            for row in explode(array, enroot, peekable_id):
                                copy.write_row(row)

                            if count % 10000 == 0:
                                pbar.n = count
                                pbar.refresh()

                            count += 1

                        except EOFError:
                            print('Reached end of file.')
                            break  # End of file reached
    conn.commit()

# CREATE INDEXes and PKEY