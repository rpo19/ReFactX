import psycopg
import redis
from transformers import AutoTokenizer
import sys
import random
import time
import json
import click
from ctrie import PostgresTrieIndex, ConstrainedState, DictIndex, TripleNotFoundException, EmptyIndexException

class TimeMeasure:
    def __init__(self, tag='default', verbose=False, outfile=sys.stdout):
        self.tag = tag
        self.verbose = verbose
        self.start_time = None
        self.duration = None
        self.outfile = outfile

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.verbose:
            print(self.tag, 'duration:', self.duration_ms(self.duration), file=self.outfile)

    def duration_ms(self, duration):
        return "{:.2f} ms".format(duration * 1000)

def merge(dst, src):
    for key in src:
        if key in dst:
            # sum numleaves
            dst[key][0] += src[key][0]
            merge(dst[key][1], src[key][1])
        else:
            # If the key exists only in `src`, the value from the `src` object will be used.
            dst[key] = src[key]

class InputTokenException(Exception):
    pass

def choose(children, initial_tokens):
    if len(initial_tokens) > 0:
        choice = initial_tokens.pop(0)
        # if not choice in children:
        #     import pdb
        #     pdb.set_trace()
        if choice not in children:
            raise InputTokenException(f'ERROR: Input token {choice} NOT in allowed tokens.')
    else:
        choice = random.choice(children)
    return choice

@click.command()
@click.option("--postgres-url", required=True, help="Postgres connection URL")
@click.option("--redis-url", required=False, default=None, help="Postgres connection URL")
@click.option("--table-name", required=True, help="Table name")
@click.option("--rootkey", type=int, required=True, help="Root key")
@click.option("--end-of-triple", type=int, required=True, help="End of triple")
@click.option("--model-name", required=True, help="Model name")
@click.option("--switch-parameter", type=int, required=True, help="Switch parameter")
@click.option("--random-seed", type=int, required=False, help="Random seed")
@click.option("--initial-tokens", default='', help="Initial tokens")
@click.option("--json-tokens", required=False, help="JSON tokens")
@click.option("--dump-subtree-cache", required=False, default=False, is_flag=True, help="Dump subtree cache")
@click.option("--dump-oneleaf-cache", required=False, default=False, is_flag=True, help="Dump oneleaf cache")
@click.option("--verbose", required=False, default=False, is_flag=True, help="Verbose mode")
@click.option("--generations", required=False, default=1, help="Number of triples to generate")
def main(postgres_url, redis_url, table_name, rootkey, end_of_triple, model_name, switch_parameter,
        random_seed, initial_tokens, json_tokens, dump_subtree_cache, dump_oneleaf_cache, verbose, generations):
    if json_tokens:
        assert not initial_tokens, 'ERROR: specify either intitial tokens or json tokens. Not both.'
        initial_tokens = json.loads(json_tokens)
        assert isinstance(initial_tokens, list)

    if random_seed is None:
        random_seed = random.randint(0, 2**32)
    random.seed(random_seed)
    print('Seed:', random_seed)

    print('Start.')

    with TimeMeasure(tag='Loading tokenizer', verbose=verbose) as tm:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    print('Initial tokens:', end=' ')
    if isinstance(initial_tokens, str):
        print(initial_tokens)
        initial_tokens = tokenizer(initial_tokens)['input_ids']
        print(initial_tokens)
    else:
        print(tokenizer.decode(initial_tokens))
        print(initial_tokens)

    postgres_connection = psycopg.connect(postgres_url, autocommit=False)
    redis_connection = redis.Redis().from_url(redis_url) if redis_url else None

    index = PostgresTrieIndex(rootkey=rootkey,
                                postgresql_connection=postgres_connection,
                                redis_connection=redis_connection,
                                switch_parameter=switch_parameter,
                                table_name=table_name,
                                end_of_triple=end_of_triple)

    state = ConstrainedState(
                begin_pattern = [],
                end_pattern = [],
                cache_index = DictIndex(end_of_triple=index.end_of_triple),
                subtree_cache = DictIndex(end_of_triple=index.end_of_triple),
                oneleaf_cache = DictIndex(end_of_triple=index.end_of_triple)
            )

    print_initial_tokens_numleaves = True
            
    with TimeMeasure(tag=f'Total time (Max generations: {generations})', verbose=True) as tm:
        for i in range(1, generations+1):
            print(i, '-'*30)
            sequence = []
            initial_tokens_run = initial_tokens.copy()
            while True:
                n = len(sequence)
                with TimeMeasure(tag=f'Query {n}', verbose=verbose) as tm:
                    if verbose:
                        print(sequence)
                        print(tokenizer.decode(sequence))
                    possible_tokens_dict = index.next_tokens(sequence, state=state)
                    possible_tokens_dict_debug = possible_tokens_dict.copy()
                    try:
                        visited_tokens = state.cache_index.next_tokens(sequence)
                        # print(visited_tokens, end=' = ')
                        state.cache_index.subtract_tokens(possible_tokens_dict, visited_tokens)
                        # print(possible_tokens)
                    except EmptyIndexException:
                        # ignore when the cache index is empty
                        pass
                    except TripleNotFoundException:
                        # ignore if triple not in cache index
                        pass

                if dump_subtree_cache and len(state.subtree_cache) > 0:
                    print('DUMP - subtree cache:')
                    print(state.subtree_cache)

                if dump_oneleaf_cache and len(state.oneleaf_cache) > 0:
                    print('DUMP - oneleaf cache:')
                    print(state.oneleaf_cache)

                possible_tokens = list(possible_tokens_dict.keys()) if possible_tokens_dict else []

                if len(possible_tokens) > 0:
                    try:
                        next_token = choose(possible_tokens, initial_tokens_run)
                        numleaves = possible_tokens_dict[next_token]
                        if verbose:
                            print(f'choosing {next_token}: numleaves: {numleaves}')

                        sequence.append(next_token)

                        if print_initial_tokens_numleaves and len(initial_tokens)>0 and sequence == initial_tokens:
                            print('Numleaves for initial_tokens:', numleaves)
                            print_initial_tokens_numleaves = False

                    except InputTokenException as e:
                        print(e)
                        break
                else:
                    if verbose:
                        print('.')
                    if not sequence[-1] == end_of_triple:
                        print('ERROR: invalid triple!', sequence)
                    else:
                        print(sequence)
                    state.cache_add(sequence)
                    state.end_of_triple_reset()
                    break

            print('TPL:', tokenizer.decode(sequence))


if __name__ == "__main__":
    main()




