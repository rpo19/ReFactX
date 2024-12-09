import requests
import bz2
from tqdm import tqdm
import psycopg
import sys
import re

ntriple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
r'<http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)>\s+'
r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"(\S+)"@en)\s+\.')

label_regex = re.compile(r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|<http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)>)\s+'
r'<http:\/\/www\.w3\.org\/2000\/01\/rdf-schema#label>\s+'
r'"(\S+)"@en\s+\.')

# groups
# relation when 4th is None
# attribute when 3rd is None
# label when 2nd and 3th are None

# match1 = ntriple_regex.match('<http://www.wikidata.org/entity/Q1234> <http://www.wikidata.org/prop/direct/P1235> <http://www.wikidata.org/entity/Q1234> .')
# match2 = ntriple_regex.match('<http://www.wikidata.org/entity/Q1234> <http://www.wikidata.org/prop/direct/P1235> "assfadf"@en .')
# match3 = label_regex.match('<http://www.wikidata.org/entity/Q148> <http://www.w3.org/2000/01/rdf-schema#label> "chinA"@en .')
# match4 = label_regex.match('<http://www.wikidata.org/prop/direct/P1235> <http://www.w3.org/2000/01/rdf-schema#label> "chinA"@en .')
# print(match1, match2)
# print(match3, match4)
# breakpoint()

# python wikidata_downloader.py postgres://postgres:pass@postgres:5432/postgres 1 1048576 2> >(python filter_error.py)

def select_all(sub, pred, obj):
    return True

def custom_selector(sub, pred, obj):
    # TODO bisogna estendere :  le label sono rdfs:label
    is_wikidata = sub.startswith('http://www.wikidata.org/entity/Q') and pred.startswith('http://www.wikidata.org/prop/direct/P')
    is_label = pred == 'http://www.w3.org/2000/01/rdf-schema#label' and obj.endswith('@en')
    if is_label:
        breakpoint()

    return is_wikidata or is_label

# def download_and_process(url, output_file, chunk_size=1024*1024, triple_selector=select_all):
def download_and_process(url, triple_selector, postgres_connection, worker_num, chunk_size):
    # postgres://postgres:$POSTGRES_PASSWORD@postgres:5432/postgres
    with psycopg.connect(postgres_connection, autocommit=False) as conn:
        # find info from work table

        with conn.cursor() as cur:
            cur.execute('SELECT id, bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend FROM work WHERE id = %s;', (int(worker_num),))
            _, bytestart, byteend, lastbyteprocessed, actualbytestart, actualbyteend = cur.fetchone()

        if lastbyteprocessed > bytestart:
            bytestart = lastbyteprocessed
            print('Starting from last processed byte:', bytestart)

        # TODO enclose in try except that saves byte_count
        with requests.get(url, stream=True) as response:
            # Check if the server supports range requests
            if response.status_code not in (200, 206):
                raise Exception(f"Failed to download file: {response.status_code}")


            decompressor = bz2.BZ2Decompressor()

            byte_count_cum = 0

            with conn.cursor() as cur:

                with tqdm(total=byteend - bytestart, file=sys.stdout) as pbar:

                    staged_line = '' # broken line at the end of the chunk to continue in the next

                    for chunk in response.iter_content(chunk_size):
                        relations = []
                        attributes = []
                        entity_labels = []
                        pred_labels = []

                        byte_count = len(chunk)
                        byte_count_cum += byte_count

                        if chunk:
                            decompressed = decompressor.decompress(chunk)

                            if byte_count_cum < bytestart:
                                continue

                            # first and last line can ba a problem
                            byte_lines = decompressed.split(b'\n')
                            ok_lines = []

                            for line in byte_lines:
                                line = line.decode()

                                if staged_line:
                                    line = staged_line + line
                                    staged_line = ''

                                if not line:
                                    # empty line
                                    # print('emtpy')
                                    continue
                                elif line[0] != '<':
                                    # print('no <', line)
                                    # nt line should start with <
                                    continue
                                elif not line.endswith(' .'):
                                    # print('no .')
                                    # nt line should end with .
                                    # TODO need to continue in next chunk. if last one more
                                    staged_line = line
                                    # TODO go over byteend if there is a stagedline
                                    continue
                                else:
                                    ok_lines.append(line)

                            save_triple = ''

                            try:
                                for line in ok_lines:
                                    save_triple = line

                                    triple_match = ntriple_regex.match(line)
                                    # groups
                                    # relation when 4th is None
                                    # attribute when 3rd is None
                                    if triple_match:
                                        sub, pred, ob1, ob2 = triple_match.groups()
                                        if ob1:
                                            # relation
                                            sub = int(sub)
                                            pred = int(pred)
                                            obj = int(ob1)
                                            relations.append((sub, pred, obj))
                                        elif ob2:
                                            # attribute
                                            sub = int(sub)
                                            pred = int(pred)
                                            obj = ob2
                                            attributes.append((sub, pred, obj))
                                        else:
                                            raise Exception('ERROR: Unecpected case triple match.')
                                    else:
                                        label_match = label_regex.match(line)
                                        if label_match:
                                            sub, pred, obj = label_match.groups()
                                            if sub:
                                                # entity label
                                                entity_labels.append((sub, obj))
                                            elif pred:
                                                # entity label
                                                pred_labels.append((pred, obj))
                                            else:
                                                raise Exception('ERROR: Unecpected case triple match.')

                            except Exception as e:
                                print('Exception on triple', save_triple, file=sys.stderr, flush=True)
                                raise e

                        with cur.copy("COPY relations (subject, predicate, object) FROM STDIN") as copy:
                            for item in relations:
                                copy.write_row(item)

                        with cur.copy("COPY attributes (subject, predicate, object) FROM STDIN") as copy:
                            for item in attributes:
                                copy.write_row(item)

                        with cur.copy("COPY entitylabels (subject, object) FROM STDIN") as copy:
                            for item in entity_labels:
                                copy.write_row(item)

                        with cur.copy("COPY predlabels (predicate, object) FROM STDIN") as copy:
                            for item in pred_labels:
                                copy.write_row(item)

                        cur.execute('''UPDATE work SET lastbyteprocessed = %s WHERE id = %s;''', (bytestart + byte_count_cum, worker_num))

                        conn.commit()

                        pbar.update(byte_count)


    print("Download complete.")

if __name__ == "__main__":
    URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.nt.bz2"

    postgres_connection = sys.argv[1]
    worker_num = int(sys.argv[2])
    chunk_size = int(sys.argv[3])

    # try:
    download_and_process(URL, custom_selector, postgres_connection, worker_num, chunk_size)
    # except Exception as e:
        # print(f"Download interrupted: {e}")
