import requests
import bz2
import rdflib
from tqdm import tqdm
import psycopg
import sys

# python wikidata_downloader.py postgres://postgres:$PASSWORD@postgres:5432/postgres 1 10485760 2>&1 | pcregrep -M -v 'Failed to convert Literal lexical form to value.*(\n|.)*?isodate.isoerror.ISO8601Error.*'

def select_all(sub, pred, obj):
    return True

def custom_selector(sub, pred, obj):
    # TODO bisogna estendere :  le label sono rdfs:label
    is_wikidata = sub.startswith('http://www.wikidata.org/entity/Q') and pred.startswith('http://www.wikidata.org/prop/direct/P')
    is_label = str(pred) == 'http://www.w3.org/2000/01/rdf-schema#label' and obj.language == 'en'
    return is_wikidata or is_label

# def download_and_process(url, output_file, chunk_size=1024*1024, triple_selector=select_all):
def download_and_process(url, triple_selector, postgres_connection, worker_num, chunk_size):
    # postgres://postgres:$POSTGRES_PASSWORD@postgres:5432/postgres
    with psycopg.connect(postgres_connection, autocommit=False) as conn:
        # find info from work table

        with conn.cursor() as cur:
            cur.execute('SELECT id, bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend FROM work WHERE id = %s;', (int(worker_num),))
            _, bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend = cur.fetchone()

        # Determine the starting byte
        # if os.path.exists(output_file):
        #     start_byte = os.path.getsize(output_file)
        #     print(f"Resuming from byte {start_byte}")

        if lastbyteprocessed > bytestart:
            bytestart = lastbyteprocessed
            print('Starting from last processde byte:', bytestart)

        bytestart = 900000

        headers = {"Range": f"bytes={bytestart}-"}
        # TODO enclose in try except that saves byte_count
        with requests.get(url, headers=headers, stream=True) as response:
            # Check if the server supports range requests
            if response.status_code not in (200, 206):
                raise Exception(f"Failed to download file: {response.status_code}")


            decompressor = bz2.BZ2Decompressor()

            # with open(output_file, "ab") as file:
            byte_count_cum = 0

            with conn.cursor() as cur:

                # with tqdm(total=byteend - bytestart, file=sys.stdout) as pbar:

                staged_line = '' # broken line at the end of the chunk to continue in the next

                relations = []
                attributes = []
                labels = []

                stopnext = False
                for chunk in response.iter_content(chunk_size):
                    byte_count = len(chunk)
                    if chunk:
                        g = rdflib.Graph()
                        decompressed = decompressor.decompress(chunk)
                        # first and last line can ba a problem
                        byte_lines = decompressed.split(b'\n')
                        ok_lines = ''

                        for line in byte_lines:
                            line = line.decode()
                            if not line:
                                # empty line
                                # print('emtpy')
                                continue
                            elif line[0] != '<':
                                # print('no <', line)
                                # nt line should start with <
                                if staged_line:
                                    # in this worker
                                    line = staged_line + line
                                    staged_line = ''
                                    assert line[0] == '<' and line[-1] == '.'
                                else:
                                    # at byteend start ?
                                    actualbytestart_new = byte_count + 1
                                    cur.execute('''UPDATE work SET actualbytestart = %s WHERE id = %s;''', (actualbytestart_new, worker_num))
                                continue
                            elif line[-1] != '.':
                                # print('no .')
                                # nt line should end with .
                                # TODO need to continue in next chunk. if last one more
                                staged_line = line
                                # TODO go over byteend if there is a stagedline
                                continue

                            ok_lines += line + '\n'
                        try:
                            g.parse(data=ok_lines, format="nt")
                        except Exception:
                            assert False

                        selected_triples = (item for item in g if triple_selector(*item))

                        try:
                            for sub, pred, obj in selected_triples:
                                save_triple = sub, pred, obj
                                if str(pred) == 'http://www.w3.org/2000/01/rdf-schema#label':
                                    # is label
                                    sub = int(str(sub)[32:])
                                    obj = str(obj)
                                    labels.append((sub, obj))
                                else:
                                    # is fact
                                    sub = int(str(sub)[32:])
                                    pred = int(str(pred)[37:])
                                    if str(obj).startswith('http://www.wikidata.org/entity/Q'):
                                        obj = int(str(obj)[32:])
                                        relations.append((sub, pred, obj))
                                    # if isinstance(obj, rdflib.term.Literal):
                                    else:
                                        # literal
                                        obj = str(obj)
                                        attributes.append((sub, pred, obj))
                        except Exception as e:
                            print('Exception on triple', save_triple, file=sys.stderr, flush=True)
                            raise e

                    with cur.copy("COPY relations (subject, predicate, object) FROM STDIN") as copy:
                        for item in relations:
                            copy.write_row(item)

                    with cur.copy("COPY attributes (subject, predicate, object) FROM STDIN") as copy:
                        for item in attributes:
                            copy.write_row(item)

                    with cur.copy("COPY labels (subject, object) FROM STDIN") as copy:
                        for item in labels:
                            copy.write_row(item)

                    byte_count_cum += byte_count
                    cur.execute('''UPDATE work SET lastbyteprocessed = %s WHERE id = %s;''', (byte_count_cum, worker_num))

                    if bytestart + byte_count_cum > byteend:
                        cur.execute('''UPDATE work SET actualbyteend = %s WHERE id = %s;''', (byte_count_cum, worker_num))

                    conn.commit()

                    # pbar.update(byte_count)                            # sql copy write

                    if bytestart + byte_count_cum >= byteend:
                        if stopnext or not staged_line:
                            # stop
                            break
                        else:
                            # go over byteend for finishing the line
                            stopnext = True
                    # sql save chunk
                    # sql commit

    print("Download complete.")

if __name__ == "__main__":
    URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.nt.gz"

    postgres_connection = sys.argv[1]
    worker_num = int(sys.argv[2])
    chunk_size = int(sys.argv[3])

    # try:
    download_and_process(URL, custom_selector, postgres_connection, worker_num, chunk_size)
    # except Exception as e:
        # print(f"Download interrupted: {e}")
