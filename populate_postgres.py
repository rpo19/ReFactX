import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys
import pickle

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
table_name = sys.argv[3]
root = int(sys.argv[4]) # number that not collides with the vocab (check max vocab) # maybe 60000
batch_size = int(sys.argv[5])
switch_parameter = int(sys.argv[6]) # after N token save all the branch to the leaf
total_rows = int(sys.argv[7]) if len(sys.argv) > 7 else None

create_table = f'''CREATE TABLE IF NOT EXISTS {table_name} (
    id BIGINT GENERATED ALWAYS AS IDENTITY,
    key INT[] NOT NULL,
    children INT[],
    numleaves INT,
    childrenleaves INT[],
    subtree BYTEA
);'''

create_pkey = f'ALTER TABLE {table_name} ADD PRIMARY KEY (id);'
create_index = f'CREATE INDEX idx_key_btree_{table_name} ON {table_name} USING BTREE (key);'

drop_pkey = f'ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey;'
drop_index = f'DROP INDEX IF EXISTS idx_key_btree_{table_name};'

check_indexes_query = f"SELECT count(*) FROM pg_indexes WHERE tablename = '{table_name}';"

# {
#     13: [3,
#     {
#         15: [1, [17]],
#         16: [2,
#         {
#             18: [0, []],
#             19: [0, []]
#         }]
#     }],
#     20: [1, [21, 22]], # 1 leaf -> list instead of dict
#     22: [2, [3, 37, { # list contains dict if divides later
#                         78: [0, []],
#                         12: [0, []]
#                     }
#             ]
#         ] # 1 leaf -> list instead of dict
# }

def tree_add(tree, sequence):
    level = tree
    cursor = 0
    level_cursor = 0
    prev_levels = [] # need to keep track to increase numleaves
    while cursor < len(sequence):
        if level[0] == 0:
            level[0] = 1
            level[1] = sequence[cursor:]
            break # nothing more to do
        # elif isinstance(level[1], list):
        else:
            if isinstance(level[1][level_cursor], dict):
                # found a branch
                if sequence[cursor] in level[1][level_cursor]:
                    # go on with this branch
                    prev_levels.append(level)
                    level = level[1][level_cursor][sequence[cursor]]
                    level_cursor = 0
                else: # new branch

                    # found another leaf
                    # increment all prev levels
                    prev_levels.append(level)
                    for level in prev_levels:
                        level[0] += 1

                    new_branch = [0, []]
                    level[1][level_cursor][sequence[cursor]] = new_branch
                    level = new_branch # continue with new branch: same as == 0
                    level_cursor = 0
            else: # is int
                if sequence[cursor] != level[1][level_cursor]:
                    # divide in 2 branches
                    new_branch = [0, []]
                    level[1][level_cursor] = {
                        level[1][level_cursor]: [level[0], level[1][level_cursor + 1:]], # previous branch
                        sequence[cursor]: new_branch
                    }

                    # found another leaf
                    # increment all prev levels
                    prev_levels.append(level)
                    for level in prev_levels:
                        level[0] += 1

                    del level[1][level_cursor + 1:]
                    level = new_branch # continue with new branch: same as == 0
                    level_cursor = 0
                else:
                    level_cursor += 1
        cursor += 1

def batch_append(tree, nested_token_ids):
    for sequence in nested_token_ids:
        # print(tree, '+', sequence)
        tree_add(tree, sequence)
        # print('=', tree)

def get_rows(tree, rootkey, switch_parameter):
    # iterative depth first traversal with a stack
    key = [rootkey]
    stack = [(0, key, tree)]
    while len(stack) > 0:
        level_num, key, level = stack.pop()
        if len(level[1]) > 0: # otherwise nothing to do
            if level[0] == 1: # 1 leaf -> children is the sequence to the end of triple
                children = level[1]
                childrenleaves = []
                if len(children) > 0:
                    yield key, children, level[0], childrenleaves, None
            else:
                if isinstance(level[1][0], dict):
                    # branch here
                    children = list(level[1][0].keys())
                    childrenleaves = [level[1][0][c][0] for c in children]
                    next_levels = [level[1][0][child] for child in children]
                else: # is int -> only 1 child
                    children = [level[1][0]]
                    childrenleaves = [level[0]] # same numleaves as parent
                    next_levels = [[level[0], level[1][1:]]]

                if level_num >= switch_parameter:
                    yield key, children, level[0], childrenleaves, pickle.dumps(level[1])
                else:
                    for child, next_level in zip(children, next_levels):
                        stack.append((level_num + 1, key + [child], next_level))
                    if len(children) > 0:
                        # skip adding empty keys to save space
                        yield key, children, level[0], childrenleaves, None
        else:
            print('Found empty tree.')


tbar_update = batch_size
count = 0

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        # Create table and ensure index
        # and pkey are not present for fast ingestion
        cur.execute(create_table)
        cur.execute(drop_pkey)
        cur.execute(drop_index)
        conn.commit()

        cur.execute(check_indexes_query)
        count_indexes = cur.fetchone()[0]
        assert count_indexes == 0, f"Expected 0 indexes, but found {count_indexes}"

        cur.execute(f"TRUNCATE TABLE {table_name};")
        with cur.copy(f"COPY {table_name} (key, children, numleaves, childrenleaves, subtree) FROM STDIN WITH (FREEZE)") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    batch = [0, []]

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
                                batch = [0, []]

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

        print('Ingestion finished.')
        print('Creating index.')
        cur.execute(create_index)
        print('Creating primary key.')
        cur.execute(create_pkey)
        conn.commit()

