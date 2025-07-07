import psycopg
from transformers import AutoTokenizer
import sys
import random
import time
import json
import click
from ctrie import PostgresTrieIndex, ConstrainedState, DictIndex, TripleNotFoundException, EmptyIndexException
import importlib
from base_postgres_index_config import IndexConfigException
from tqdm import tqdm

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

def process_initial_tokens(initial_tokens, tokenizer, add_special_tokens, verbose=False):
    if isinstance(initial_tokens, str):
        if verbose:
            print(f'|{initial_tokens}|')
        initial_tokens = tokenizer(initial_tokens, add_special_tokens=add_special_tokens)['input_ids']
        if verbose:
            print(initial_tokens)
    elif verbose:
        print(f'|{tokenizer.decode(initial_tokens)}|')
        print(initial_tokens)
    return initial_tokens

class IndexArgumentException(Exception):
    def __init__(self, index_module, postgres_url, table_name, rootkey, switch_parameter, end_of_triple, model_name):
        message = f'''ERROR: you must either set --index-module or manually set all the following:
--postgres-url, --table-name, --rootkey, --switch-parameter, --end-of-triple, --model-name
Actual values:
--index-module {index_module}
# or
--postgres-url {postgres_url}
--table-name {table_name}
--rootkey {rootkey}
--switch-parameter {switch_parameter}
--end-of-triple {end_of_triple}
--model-name {model_name}
'''
        super().__init__(message)
    pass

@click.command()
# db
@click.option("--index-module", required=False, help="Index Module with index config")
@click.option("--postgres-url", required=False, help="Postgres connection URL")
@click.option("--cache-class", required=False, default=None, help="Cache class")
@click.option("--cache-db", required=False, default=0, help="Cache db")
@click.option("--table-name", required=False, help="Table name")
@click.option("--rootkey", type=int, required=False, help="Root key")
@click.option("--switch-parameter", type=int, required=False, help="Switch parameter")
@click.option("--end-of-triple", type=int, required=False, help="End of triple")
@click.option("--model-name", required=False, help="Model name")
@click.option("--flush-cache", is_flag=True, required=False, help="Flush cache db at program start")
#
@click.option("--random-seed", type=int, required=False, help="Random seed")
@click.option("--initial-tokens", default='', help="Initial tokens")
@click.option("--initial-tokens-file", default='', help="File containing all the initial tokens. One per line")
@click.option("--json-tokens", required=False, help="JSON tokens")
@click.option("--dump-subtree-cache", required=False, default=False, is_flag=True, help="Dump subtree cache")
@click.option("--verbose", required=False, default=False, is_flag=True, help="Verbose mode")
@click.option("--generations", required=False, default=1, help="Number of triples to generate")
@click.option("--add-special-tokens", required=False, is_flag=True, help="Add special tokens when tokenizing initial tokens")
def main(index_module, postgres_url, cache_class, cache_db, table_name, rootkey, end_of_triple, model_name, flush_cache, switch_parameter,
        random_seed, initial_tokens, json_tokens, dump_subtree_cache, verbose, generations, add_special_tokens):
    if index_module:

        if index_module.endswith('.py'):
            index_module = index_module[:-3]
        index_module = importlib.import_module(index_module)
        index_config = getattr(index_module, 'index_config')

        end_of_triple = index_config.end_of_triple
        index = index_config.index

        model_name = index_config.model_name
    elif any(map(lambda x: x is None, [postgres_url, table_name, rootkey, switch_parameter, end_of_triple, model_name])):
        raise IndexArgumentException(index_module, postgres_url, table_name, rootkey, switch_parameter, end_of_triple, model_name)
    else:
        postgres_connection = psycopg.connect(postgres_url, autocommit=False)
        # TODO add in-memory cache

        if cache_class is not None:
            if cache_db is None:
                raise IndexConfigException('When using caching you must configure which db to use (e.g. 0)')
            cache_module = importlib.import_module(cache_class)
            cache = cache_module.__init__(cache_db)
        else:
            cache = None

        index = PostgresTrieIndex(rootkey=rootkey,
                                    postgresql_connection=postgres_connection,
                                    cache=cache,
                                    switch_parameter=switch_parameter,
                                    table_name=table_name,
                                    end_of_triple=end_of_triple)

    if flush_cache:
        print('Redis flush db and close.')
        index.cache.flush()
        sys.exit(0)

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

    if initial_tokens_file:
        # assert not initial_tokens # initial tokens is the prefix
        assert not json_tokens
        with open(initial_tokens_file) as init_fd:
            all_initial_tokens = init_fd.readlines()

        all_initial_tokens = [item.replace('\n','') for item in all_initial_tokens]
        all_initial_tokens = [initial_tokens + item for item in all_initial_tokens]
        all_initial_tokens = [process_initial_tokens(item, tokenizer, add_special_tokens, False) for item in all_initial_tokens]
    else:
        if initial_tokens:
            initial_tokens = process_initial_tokens(initial_tokens, tokenizer, add_special_tokens, True)
        else:
            initial_tokens = []
        print('||\n[]')
        all_initial_tokens = [initial_tokens]


    state = ConstrainedState(
                begin_pattern = [],
                end_pattern = [],
                cache_index = DictIndex(end_of_triple=index.end_of_triple),
                subtree_cache = DictIndex(end_of_triple=index.end_of_triple),
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
                    possible_tokens_dict, extra = index.next_tokens(sequence, state=state)
                    possible_tokens_dict_debug = possible_tokens_dict.copy()
                    if verbose and extra and extra.get('found_subtree'):
                        print('found_subtree')
                    try:
                        visited_tokens, _ = state.cache_index.next_tokens(sequence)
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

                possible_tokens = list(possible_tokens_dict.keys()) if possible_tokens_dict else []

                if len(possible_tokens) > 0:
                    try:
                        next_token = choose(possible_tokens, initial_tokens_run)
                        numleaves = possible_tokens_dict[next_token]
                        if verbose:
                            print(sequence)
                            print(tokenizer.decode(sequence))
                        possible_tokens_dict, extra = index.next_tokens(sequence, state=state)
                        possible_tokens_dict_debug = possible_tokens_dict.copy()
                        if verbose and extra and extra.get('found_subtree'):
                            print('found_subtree')
                        try:
                            visited_tokens, _ = state.cache_index.next_tokens(sequence)
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

                        if verbose and extra and extra.get('tokens_from_oneleaf', False) and next_token in extra.get('tokens_from_oneleaf', {}):
                            print('chose token from oneleaf')
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

            print('triple: "{}"'.format(tokenizer.decode(sequence)))


if __name__ == "__main__":
    main()




