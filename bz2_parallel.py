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

# simpler solution
# 1. download big dump in bz2
# 2. run bzip2recovery to divide the big bz2 into its blocks
# 3. process each block in parallel (will have blocken lines at the beginning and end)
# 4. process the broken lines: look for last line of blockN and merge it with blockN+1 to process

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

def testit2(fname, magic_idx=None, chunk_size=1024, n_blocks=None):
    fd = open(fname, 'rb')

    header = fd.read(4)

    end_of_file = fd.seek(0, os.SEEK_END)
    fd.seek(0)

    if not magic_idx:
        magic_idx = list(find_all_compressed_magic(fd))

    finaldata = b''

    for i in range(len(magic_idx)):
        if i > n_blocks:
            break
        current_idx = magic_idx[i]
        next_idx = magic_idx[i+1] if i < len(magic_idx) else end_of_file

        fd.seek(current_idx)

        decompressor = bz2.BZ2Decompressor()

        data = b''
        begin = True
        while fd.tell() < next_idx:
            print('.',end='')
            remaining_size = next_idx - fd.tell()
            read_size = remaining_size if chunk_size > remaining_size else chunk_size

            if begin:
                chunk = header + fd.read(read_size - len(header))
                begin = False
            else:
                chunk = fd.read(read_size)

            data += decompressor.decompress(chunk)

        finaldata += data

        print()
        print('B', data[:64].decode())
        print('E', data[-64:].decode())
        print('len', len(data))
        print(fd.tell(), next_idx)

    return finaldata, fd.tell(), i - 1


from multiprocessing import Process

def process_block(name):
    print 'hello', name

if __name__ == '__main__':
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

def testit3(fname, magic_idx=None, chunk_size=1024 - 6):
    fd = open(fname, 'rb')

    header = fd.read(4)

    end_of_file = fd.seek(0, os.SEEK_END)
    fd.seek(0)

    if not magic_idx:
        magic_idx = list(find_all_compressed_magic(fd))

    for i in range(len(magic_idx)):
        next_idx = magic_idx[i+1] if i < len(magic_idx) else end_of_file

        fd.seek(next_idx)

        decompressor = bz2.BZ2Decompressor()

        data = b''
        while fd.tell() < next_idx:
            remaining_size = next_idx - fd.tell()
            read_size = remaining_size if chunk_size > remaining_size else chunk_size
            chunk = header + fd.read(read_size)

            data += decompressor.decompress(chunk)
            if len(data) == read_size:
                print('begin', data[:64])

        print('end', data[-64:])
