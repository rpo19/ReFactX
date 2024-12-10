import bz2
import os
from tqdm import tqdm

# real use case:
# you have N workers to decompress and process bzip2 blocks # TODO how to stop when the block is finished?
# 1. you start to find all the starting position of bzip2 blocks with these functions
# 2. while you find the position you assign the block to the N workers
# 3. as soon as a worker finished its job you assign another block
# 4. you need to check data broken into more blocks: stage incomplete data (e.g. broken line or json dictionary) at start and end of each block
#    # and process it when you have found the respective start-end pair

def find_next_sequence(stream, sequence):
    seekbegin = stream.tell()
    _seq_index = 0
    _b = stream.read(1)
    _b_index = 0
    while len(_b) > 0:
        _b = _b[0]
        if _b == sequence[_seq_index]:
            _seq_index += 1
        else:
            _seq_index = 0
        if _seq_index == len(sequence):
            # found
            return _b_index - len(sequence) + 1 + seekbegin
        _b_index += 1
        _b = stream.read(1)

def find_all_compressed_magic(fd):

    end_of_file = fd.seek(0, os.SEEK_END)
    fd.seek(0)

    compressed_magic = bytes.fromhex('314159265359')

    next_idx = find_next_sequence(fd, compressed_magic)

    with tqdm(total=end_of_file) as pbar:
        while next_idx is not None and fd.tell() < end_of_file:
            yield next_idx

            fd.seek(next_idx + 6) # seek after compressed magic bytes

            pbar.n = next_idx
            pbar.refresh()

            next_idx = find_next_sequence(fd, compressed_magic)

def testit(fname, chunk_size=1024 - 6):
    fd = open(fname, 'rb')

    end_of_file = fd.seek(0, os.SEEK_END)
    fd.seek(0)

    header = fd.read(4)
    compressed_magic = bytes.fromhex('314159265359')

    next_idx = find_next_sequence(fd, compressed_magic)

    with tqdm(total=end_of_file) as pbar:
        while next_idx is not None and fd.tell() < end_of_file:

            # print(next_idx)
            fd.seek(next_idx)

            chunk = header + fd.read(chunk_size)

            bz2.BZ2Decompressor().decompress(chunk)

            pbar.n = next_idx
            pbar.refresh()

            next_idx = find_next_sequence(fd, compressed_magic)
