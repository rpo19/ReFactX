# todo better using https://wdumps.toolforge.org/# ?
import bz2
import re
import os
from tqdm import tqdm
import psycopg
import sys

force = sys.argv[1] if len(sys.argv) > 1 else 'no'

dump_path = '/workspace/data/latest-truthy.nt.bz2'
postgres_connection = 'postgres://postgres:secret@10.0.0.118:5432/postgres'
chunk_size = 128

label_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
r'<http:\/\/www\.w3\.org\/2000\/01\/rdf-schema#label>\s+'
r'"(\S+)"@en\s+\.')

with bz2.BZ2File(dump_path, "r") as dump:

    size = os.path.getsize(dump_path)
    dump.seek(0)
    with psycopg.connect(postgres_connection, autocommit=False) as conn:
        with conn.cursor() as cur:
            if force:
                print('deleting from entitylabels...')
                cur.execute('DELETE FROM entitylabels;')
            conn.commit()

            with tqdm(total=size) as pbar:
                chunk = []
                for bline in dump:
                    line = bline.decode()
                    if 'rdf-schema#label' in line: # not sure if this is faster
                        match = label_regex.match(line)
                        if match:
                            qid, label = match.groups()
                            chunk.append((qid, label))

                        if len(chunk) >= chunk_size:
                            with cur.copy("COPY entitylabels (subject, object) FROM STDIN") as copy:
                                for item in chunk:
                                    copy.write_row(item)
                            conn.commit()
                            chunk = []

                    pbar.n = dump.tell()
                    pbar.refresh()
