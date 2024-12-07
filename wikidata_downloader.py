import os
import requests
import bz2
from rdflib import Graph
import pdb
from tqdm import tqdm

ntriple_regex = r'/(<.+>) (<.+>) (.+) \./'

def download_and_decompress(response, chunk_size, decompressor):
    for chunk in response.iter_content(chunk_size=chunk_size):
        byte_count = len(chunk)
        if chunk:
            g = Graph()
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
                    continue
                elif line[-1] != '.':
                    # print('no .')
                    # nt line should end with .
                    # TODO need to continue in next chunk. if last one more
                    continue
                else:
                    ok_lines += line + '\n'

            g.parse(data=ok_lines, format="nt")

            for subject, predicate, object in g:
                yield 'triple', (subject, predicate, object)

        yield 'byte_count', byte_count

def select_all(sub, pred, obj):
    return True

def custom_selector(sub, pred, obj):
    is_wikidata = sub.startswith('http://www.wikidata.org/entity/Q') and pred.startswith('http://www.wikidata.org/prop/direct/P')
    # pdb.set_trace()
    return is_wikidata

# def download_and_process(url, output_file, chunk_size=1024*1024, triple_selector=select_all):
def download_and_process(url, output_file, chunk_size, triple_selector, size=None):
    # Determine the starting byte
    start_byte = 0
    if os.path.exists(output_file):
        start_byte = os.path.getsize(output_file)
        print(f"Resuming from byte {start_byte}")

    headers = {"Range": f"bytes={start_byte}-"}
    # TODO enclose in try except that saves byte_count
    with requests.get(url, headers=headers, stream=True) as response:
        # Check if the server supports range requests
        if response.status_code not in (200, 206):
            raise Exception(f"Failed to download file: {response.status_code}")

        # Open the file in append mode

        decompressor = bz2.BZ2Decompressor()

        # with open(output_file, "ab") as file:
        byte_count_cum = 0
        with tqdm(total=size) as pbar:
            for key, item in download_and_decompress(response, chunk_size, decompressor):
                # if chunk:  # Filter out keep-alive chunks
                #     chunks.append(chunk)
                #     # decomp = decompressor.decompress(chunk)
                if key == 'byte_count':
                    # update count # save it in case of exception
                    byte_count_cum += item
                    pbar.update(item)
                    continue

                assert key == 'triple'

                if not triple_selector(*item):
                    # skip triple not selected
                    # print('selector skip', item)
                    # if 'Q31' in str(item[0]):
                    #     print(item)
                    continue

                sub, pred, obj = item
                # TODO scrivo csv gzippato e parallelizzo
                sub = str(sub)
                pred = str(pred)
                obj = str(obj)


    print("Download complete.")

def process_chunk(chunk):
    # Example processing of the chunk (customize for your needs)
    print(f"Processing a chunk of size {len(chunk)}")

if __name__ == "__main__":
    URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.nt.bz2"
    OUTPUT_FILE = "latest-all.nt.bz2"

    # try:
    chunk_size = 1024*1024
    size = 178101602468
    download_and_process(URL, OUTPUT_FILE, chunk_size, custom_selector, size)
    # except Exception as e:
        # print(f"Download interrupted: {e}")
