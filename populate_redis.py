import pickle
import bz2
import redis
from tqdm import tqdm
import sys
import ctrie

fname = sys.argv[1]
redis_url = sys.argv[2] # redis://default:12345678@10.0.0.118:6379/0'
root_key = int(sys.argv[3])
total_rows = int(sys.argv[4]) if len(sys.argv) > 4 else None

redis_connection = redis.Redis().from_url(redis_url)

tqdm_args = dict(total=total_rows,mininterval=1)

def load_iterator():
    with bz2.BZ2File(fname, "rb") as bz2file:
        while True:
            try:
                # Load each pickled object from the bz2 file
                array = pickle.load(bz2file)

                yield array

            except EOFError:
                print('Reached end of file.')
                break  # End of file reached

iterator = load_iterator()

ctrie.ModDisjunctiveTrie(redis_connection, root_key, iterator).append(iterator, tqdm_args=tqdm_args)
