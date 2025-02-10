# CREATE TABLE ctriev2 (
#     id BIGINT GENERATED ALWAYS AS IDENTITY, -- later make PRIMARY KEY
#     key INT[] NOT NULL,
#     children INT[],
#     numleaves INT,
#     childrenleaves INT[],
#     subtree BYTEA
# );

import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys
sys.setrecursionlimit(10000) # increase recursion limit for pickling nested trees
import pickle

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
table_name = sys.argv[3]
root = int(sys.argv[4]) # number that not collides with the vocab (check max vocab) # maybe 60000
batch_size = int(sys.argv[5])
switch_parameter = int(sys.argv[6]) # after N token save all the branch to the leaf
total_rows = int(sys.argv[7]) if len(sys.argv) > 7 else None

# {
#     13: (3,
#     {
#         15: (1, {17: (1,{})}),
#         16: (2,
#         {
#             18: (1, {}),
#             19: (1, {})
#         })
#     }),
#     20: (1,
#     {
#         21: (1, {22: (1, {})}
#     )}
# )}
def batch_append(trie, token_ids):
    level = trie # (leaves_count, child_tree)
    for token_id in token_ids:
        if token_id not in level[1]:
            level[1][token_id] = [0, {}] # (leaves_count, child_tree)

        # increment count of current level
        level[0] += 1

        level = level[1][token_id]

def get_rows(trie, rootkey, switch_parameter):
    # iterative depth first traversal with a stack
    key = [rootkey]
    stack = [(0, key, trie)]
    while len(stack) > 0:
        level_num, key, level = stack.pop()
        children = list(level[1].keys())
        childrenleaves = [level[1][c][0] for c in children]
        if level_num >= switch_parameter:
            yield key, children, level[0], childrenleaves, pickle.dumps(level[1])
        else:
            for child in children:
                stack.append((level_num + 1, key + [child], level[1][child]))
            if len(children) > 0:
                # skip adding empty keys to save space
                yield key, children, level[0], childrenleaves, None


tbar_update = batch_size
count = 0

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {table_name};")
        with cur.copy(f"COPY {table_name} (key, children, numleaves, childrenleaves, subtree) FROM STDIN WITH (FREEZE)") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    batch = [0, {}]

                    while True:
                        try:

                            # Load each pickled object from the bz2 file
                            array = pickle.load(bz2file)
                            batch_append(batch, array)

                            count += 1

                            if count % batch_size == 0:
                                # batch on number or rows processed
                                for row in get_rows(batch, root, switch_parameter):
                                    copy.write_row(row)
                                # reset batch
                                batch = [0, {}]

                            if count % tbar_update == 0:
                                pbar.n = count
                                pbar.refresh()

                        except EOFError:
                            print('Reached end of file.')
                            break  # End of file reached
                        except Exception as e:
                            print(f'Encountered exception at {count}')
                            raise e
    conn.commit()

# CREATE INDEXes and PKEY
# CREATE INDEX idx_key_btree ON ctriev2test2 USING BTREE (key);

