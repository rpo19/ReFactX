import os
import requests
import bz2
import json
import ijson

def download_and_decompress(response, chunk_size, decompressor):
    stack = b''
    byte_count = 0
    for chunk in response.iter_content(chunk_size=chunk_size):
        byte_count += len(chunk)
        if chunk:
            decompressed = decompressor.decompress(chunk)
            for line in decompressed.split(b'\n'):
                if line != b'[' and line != b']':
                    if line[-1:] == b',':
                        item =  json.loads(stack + line[:-1]) # except final comma
                        yield item, byte_count
                        stack = b''
                    else:
                        stack += line

def download_and_process(url, output_file, chunk_size=1024*1024):
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
        chunks = []

        # with open(output_file, "ab") as file:
        for item, byte_count in download_and_decompress(response, chunk_size, decompressor):
            # if chunk:  # Filter out keep-alive chunks
            #     chunks.append(chunk)
            #     # decomp = decompressor.decompress(chunk)

            subject = item['labels']['en']['value']
            for property, claim_obj in item['claims'].items():
                # only keeping claim[0] # most recent one?
                value = claim_obj[0]['mainsnak']['datavalue']['value']
                print(subject, property, value)
                # print(label, claim[0]['property'], claim[0]['datavalue']['value'])

                # TODO davvero tanti tipi diversi di value
                # valutare triple turtle. probabilmente meglio turtle e filtrare solo per le entita?

                import pdb
                pdb.set_trace()
                # Process the chunk (modify this as needed for your use case)
                # process_chunk(chunk)
            # print(f"Downloaded {file.tell()} bytes", end="\r")
    print("Download complete.")

def process_chunk(chunk):
    # Example processing of the chunk (customize for your needs)
    print(f"Processing a chunk of size {len(chunk)}")

if __name__ == "__main__":
    URL = "https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2"
    OUTPUT_FILE = "latest-all.json.bz2"

    try:
        download_and_process(URL, OUTPUT_FILE)
    except Exception as e:
        print(f"Download interrupted: {e}")
