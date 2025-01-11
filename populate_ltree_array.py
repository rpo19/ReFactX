import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
total_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None

with psycopg.connect(postgres_connection, autocommit=False) as conn:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE tree_int;")
        with cur.copy("COPY tree_int (path) FROM STDIN WITH (FREEZE)") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    count = 0
                    while True:
                        try:
                            # Load each pickled object from the bz2 file
                            array = pickle.load(bz2file)
                            # create row

                            row = ('{'+','.join(map(str, array))+'}',)
                            # print(array)
                            copy.write_row(row)

                            count += 1

                            if count % 10000 == 0:
                                pbar.n = count
                                pbar.refresh()

                        except EOFError:
                            print('Reached end of file.')
                            break  # End of file reached
    conn.commit()

# CREATE INDEX tree_path_gist_idx ON tree USING GIST (path);
# TODO add index creation (and also primary key?) # CREATE INDEX path_idx ON tree USING BTREE (path); # ALTER TABLE tree ADD PRIMARY KEY (id);
