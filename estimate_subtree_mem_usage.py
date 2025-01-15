# calculate the probability of having only one

import pickle
import bz2
from tqdm import tqdm
import sys
import numpy

fname = sys.argv[1]
root = int(sys.argv[2]) # number that not collides with the vocab (check max vocab) # maybe 60000
sample_size = int(sys.argv[3])
switch_parameter = int(sys.argv[4]) # calculate for levels >= switch_parameter (speed up)

##### https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

####

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

def batch_append(trie, token_ids):
    level = trie
    for token_id in token_ids:
        if token_id not in level:
            level[token_id] = {}

        level = level[token_id]

def calc_size_by_level(root):
    stack = [root]
    levelnum = 0
    sizes = {}
    while len(stack) > 0:
        new_stack = []
        for level in stack:
            if levelnum not in sizes:
                sizes[levelnum] = []
            if levelnum >= switch_parameter:
                sizes[levelnum].append(total_size(level))

            new_stack.extend(list(level.values()))

        levelnum += 1
        stack = new_stack

    return sizes

enroot = tken(root)
tbar_update = sample_size // 100
count = 0

with bz2.BZ2File(fname, "rb") as bz2file:
    with tqdm(total=sample_size) as pbar:
        batch = {}

        while True:
            try:

                # Load each pickled object from the bz2 file
                array = pickle.load(bz2file)
                batch_append(batch, array)

                count += 1

                if count % tbar_update == 0:
                    pbar.n = count
                    pbar.refresh()

                if count >= sample_size:
                    break

            except EOFError:
                print('Reached end of file.')
                break  # End of file reached

sizes_dict = calc_size_by_level(batch)

print('int size', total_size(32))

for levelnum, sizes in sizes_dict.items():
    if len(sizes) > 0:
        print(levelnum, min(sizes), max(sizes), float(numpy.mean(sizes)), numpy.quantile(sizes, [0.75,0.9,0.99]))
