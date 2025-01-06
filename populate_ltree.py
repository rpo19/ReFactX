import pickle
import bz2
import psycopg
from tqdm import tqdm
import sys

fname = sys.argv[1]
postgres_connection = sys.argv[2] # 'postgres://postgres:secret@host:5432/postgres'
total_rows = int(sys.argv[3]) if len(sys.argv) > 3 else None

with psycopg.connect(postgres_connection, autocommit=True) as conn:
    with conn.cursor() as cur:
        with cur.copy("COPY tree (path) FROM STDIN") as copy:
            with bz2.BZ2File(fname, "rb") as bz2file:
                with tqdm(total=total_rows) as pbar:
                    count = 0
                    while True:
                        try:
                            # Load each pickled object from the bz2 file
                            array = pickle.load(bz2file)
                            # create row
                            row = ('.'.join(map(str, array)),)
                            # print(array)
                            copy.write_row(row)

                            if count % 10000 == 0:
                                pbar.n = count
                                pbar.refresh()

                            count += 1
                        except EOFError:
                            break  # End of file reached
