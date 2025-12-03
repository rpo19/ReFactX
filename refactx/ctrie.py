from transformers.generation.logits_process import LogitsProcessor
import torch
import pickle
from psycopg import sql
from copy import deepcopy
import requests
from requests.adapters import HTTPAdapter, Retry
import math
import types
import os
import gzip
from tqdm import trange
from urllib.parse import urlparse, parse_qs
from .SimpleCache import SimpleCache


# must import and initialize
DEFAULT_CONFIGKEY=-200
CONSTRAINED_STATES = None

def patch_model(model):
    _get_running_beams_for_next_iteration_original = model.__class__._get_running_beams_for_next_iteration
    
    def _get_running_beams_for_next_iteration_patch(self,*args, **kwargs):
        global CONSTRAINED_STATES
        running_sequences, running_beam_scores, running_beam_indices = _get_running_beams_for_next_iteration_original(self, *args, **kwargs)
        CONSTRAINED_STATES.beam_idx = running_beam_indices
        return running_sequences, running_beam_scores, running_beam_indices    
    
    model._get_running_beams_for_next_iteration = types.MethodType(_get_running_beams_for_next_iteration_patch, model)

def load_index(url, tokenizer, add_special_tokens=False, clean=True, batch_size=100, configkey=DEFAULT_CONFIGKEY, cache='default'):
    if os.path.isfile(url):
        return _load_index_from_txt(url, tokenizer, add_special_tokens, clean, batch_size)
    elif url.startswith('postgresql://') or url.startswith('postgres://'):
        return _load_index_from_postgresql(url, tokenizer, configkey=configkey, cache=cache)
    elif url.startswith('http://') or url.startswith('https://'):
        raise NotImplementedError('automatic load of http indexes not implemented yet.')

def _load_index_from_txt(path, tokenizer, add_special_tokens=False, clean=True, batch_size=100):
    index = DictIndex(tokenizer)
    index.load_from_path(path)
    index.tokenize_triples(add_special_tokens, batch_size)
    if clean:
        index.clean()

    return index

def _load_index_from_postgresql(url, tokenizer, configkey=DEFAULT_CONFIGKEY, cache='default'):
    # postgres://user:pwd@host:port/dbname?table_name=tablename&switch_parameter=7&rootkey=500000
    # Parse the URL
    parsed = urlparse(url)

    # Remove the query part
    url_without_query = url.split("?", 1)[0]

    # Parse the query into a dict (values are lists by default)
    parsed_query = parse_qs(parsed.query)

    print("url_without_query:", url_without_query)
    print("parsed_query:", parsed_query)

    table_name = parsed_query['table_name'][0] if isinstance(parsed_query['table_name'], list) else parsed_query['table_name']

    import psycopg
    postgresql_connection = psycopg.connect(url_without_query)

    if cache == 'default':
        cache = SimpleCache(0)

    index = PostgresTrieIndex(
        postgresql_connection = postgresql_connection,
        table_name = table_name,
        cache = cache,
        tokenizer=tokenizer,
        configkey=configkey
        )
    
    return index

def populate_postgres_index(file_reader, postgres_url, tokenizer, table_name, batch_size=5000, rootkey = -100, configkey=DEFAULT_CONFIGKEY, switch_parameter = 7, total_number_of_triples=None, prefix='', tokenizer_batch_size=5000, add_special_tokens=False, count_leaves=True, debug=False):
    if not tokenizer.is_fast:
        print('WARNING: tokenizer is not fast.')

    index = PostgresIngestIndex(
                rootkey=rootkey,
                configkey=configkey,
                switch_parameter=switch_parameter,
                table_name=table_name,
                tokenizer=tokenizer)

    def batch_append(nested_token_ids, index):
        for sequence in nested_token_ids:
            index.add(sequence)

    tbar_update = batch_size
    count = 0

    import psycopg
    from tqdm import tqdm
    with psycopg.connect(postgres_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Create table and ensure index
            # and pkey are not present for fast ingestion
            cur.execute(index.create_table_query)
            cur.execute(index.drop_pkey_query)
            cur.execute(index.drop_index_query)
            conn.commit()

            cur.execute(index.check_indexes_query)
            count_indexes = cur.fetchone()[0]
            assert count_indexes == 0, f"Expected 0 indexes, but found {count_indexes}"

            cur.execute(index.truncate_query)
            with cur.copy(index.copy_query) as copy:
                with tqdm(total=total_number_of_triples) as pbar:
                    tokenizer_batch = []
                    for count, bline in enumerate(file_reader):
                        try:
                            line = bline.decode()
                            if line[-1] == '\n':
                                line = line[:-1]

                            line = prefix + line

                            tokenizer_batch.append(line)

                            if len(tokenizer_batch) == tokenizer_batch_size:
                                ids = tokenizer(tokenizer_batch, add_special_tokens=add_special_tokens)['input_ids']
                                tokenizer_batch = []

                                batch_append(ids, index)

                            if count % batch_size == 0 and count > 0:
                                # batch on number or rows processed
                                for row in index.get_rows():
                                    copy.write_row(row)

                                if count_leaves:
                                    try:
                                        lfc = index.count_leaves(fail_on_wrong_num=True)
                                    except WrongNumleavesException as e:
                                        print(e, 'at', count)

                                # reset batch
                                index.reset()

                                if debug:
                                    print('DEBUG! Breaking after first batch.')
                                    break

                            if count % tbar_update == 0:
                                pbar.n = count
                                pbar.refresh()

                        except EOFError:
                            print('Reached end of file.')
                            break  # End of file reached
                        except Exception as e:
                            print(f'Encountered exception at {count}')
                            raise e

                config_row = index.get_config_row()
                copy.write_row(config_row)

            conn.commit()

            print('Ingestion finished.')
            print('Creating index.')
            cur.execute(index.create_index_query)
            print('Creating primary key.')
            cur.execute(index.create_pkey_query)
            conn.commit()

def apply_prompt_template(prompt_template, tokenizer, question=None):
    if question is None:
        # only prompt for caching
        return tokenizer.apply_chat_template(prompt_template, tokenize=False, add_generation_prompt=False)
    else:
        question_w_role = {'role':'user', 'content': question}
        return tokenizer.apply_chat_template(prompt_template + [question_w_role], tokenize=False, add_generation_prompt=True)

class EmptyIndexException(Exception):
    pass

class TripleNotFoundException(Exception):
    pass

class WrongNumleavesException(Exception):
    pass

class Index():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def add(self, sequence):
        pass

    def next_tokens(self, sequence: list, **kwargs) -> dict:
        token = 0
        numleaves = 10
        extra = {}
        return {token: numleaves,}, extra

    def subtract_tokens(self, tokens_a: list, tokens_b: list) -> None:
        for token in list(tokens_a.keys()):
            if token in tokens_b:
                diff = tokens_a[token] - tokens_b[token] # TODO transform subtract tokens in a prob modifier
                if diff <= 0:
                    # branch completely visited
                    # forbid generating again same triple
                    del tokens_a[token]
                else:
                    tokens_a[token] = diff

    def load_from_path(self, path):
        fopen = open, 'r'
        if path.endswith('.gz'):
            fopen = gzip.open, 'rt'
        with fopen[0](path, fopen[1]) as fd:
            self.verbalized_triples = fd.readlines()
        self.verbalized_triples = list(map(lambda x: x.replace('\n',''), self.verbalized_triples))

    def batch_append(self, nested_token_ids):
        for sequence in nested_token_ids:
            self.add(sequence)

    def tokenize_triples(self, add_special_tokens=False, batch_size=100):
        for batch_start in trange(0, len(self.verbalized_triples), batch_size):
            batch_end = batch_start + batch_size
            if batch_end >= len(self.verbalized_triples):
                batch_end = len(self.verbalized_triples)

            batch = self.verbalized_triples[batch_start:batch_end]
            ids = self.tokenizer(batch, add_special_tokens=add_special_tokens)['input_ids']
            self.batch_append(ids)

    def clean(self):
        del self.verbalized_triples

class DictIndex(Index):
    def __init__(self, tokenizer, tree = None):
        super().__init__(tokenizer)
        self.tree = None
        self.reset(tree)

    def reset(self, tree = None):
        if tree is None:
            self.tree = [0, []]
        else:
            self.tree = tree

    def __eq__(self, other):
        if self is other:
            return True
        elif type(self) != type(other):
            return False
        else:
            return self.tree == other.tree

    def __ne__(self, other):
        return not self == other

    def __len__(self):
        length = 0
        if self.tree:
            length = self.tree[0]
        return length

    def _str_tree_dict(self, tree_dict, level=0, spacer='\t'):
        if len(tree_dict) == 0:
            _str = '{}'
        else:
            _str = '{\n'
            _str += ',\n'.join('{}{}: {}'.format(spacer * level, key, self._str_tree(value, level+1)) for key, value in tree_dict.items())
            _str += '}'
        return _str

    def _str_tree(self, tree, level=0, spacer='\t'):
        if not tree:
            return None.__str__()
        cursor = 0
        _str_dict = ''
        while cursor < len(tree[1]):
            if isinstance(tree[1][cursor], dict):
                if cursor > 0:
                    _str_dict += ', '
                _str_dict += self._str_tree_dict(tree[1][cursor], level+1)
                break
            cursor += 1
        _str = '{}[{}, [{}{}]]'.format(spacer * level, tree[0], ', '.join(map(str, tree[1][:cursor])), _str_dict)
        return _str

    def __str__(self):
        return self._str_tree(self.tree)

    def __repr__(self):
        return '{}\n{}'.format(super().__repr__(), self.__str__())

    def __short_repr__(self):
        return super().__repr__()

    def __json__(self, copy=True):
        return {
            'tree': deepcopy(self.tree) if copy else self.tree,
        }

    def copy(self):
        copy_of_index = DictIndex(tree=deepcopy(self.tree))
        return copy_of_index

    def add(self, sequence, new_leaf=False):
        # could be replaced with to_dict and merge
        # but could be useful to avoid recursion
        # supports duplicates

        level = self.tree
        cursor = 0
        level_cursor = 0 # necessary to avoid recursion
        prev_levels = [] # need to keep track to increase numleaves
        while cursor < len(sequence):
            if level_cursor == len(level[1]):
                # level[0] = 1 # increment in the end
                level[1][level_cursor:] = sequence[cursor:]
                if level[0] == 0:
                    new_leaf = True
                break # nothing more to do
            # elif isinstance(level[1], list):
            else:
                if isinstance(level[1][level_cursor], dict):
                    # found a branch
                    if sequence[cursor] in level[1][level_cursor]:
                        # go on with this branch
                        prev_levels.append(level)
                        level = level[1][level_cursor][sequence[cursor]]
                        level_cursor = 0
                    else: # new branch
                        # found another leaf
                        # save prev levels for final increment
                        prev_levels.append(level)
                        new_leaf = True

                        new_branch = [0, []]
                        level[1][level_cursor][sequence[cursor]] = new_branch
                        level = new_branch # continue with new branch: end in == 0
                        level_cursor = 0
                else: # is int
                    if sequence[cursor] != level[1][level_cursor]:
                        # divide in 2 branches
                        new_branch = [0, []]
                        level[1][level_cursor] = {
                            level[1][level_cursor]: [level[0], level[1][level_cursor + 1:]], # previous branch
                            sequence[cursor]: new_branch
                        }

                        # found another leaf
                        # save prev levels for final increment
                        prev_levels.append(level)
                        new_leaf = True

                        del level[1][level_cursor + 1:]
                        level = new_branch # continue with new branch: ends in == 0
                        level_cursor = 0
                    else:
                        level_cursor += 1
            cursor += 1

        if new_leaf:
            level[0] += 1
            for level in prev_levels:
                level[0] += 1

    def next_tokens(self, sequence, **kwargs):
        level = self.tree
        cursor = 0
        level_cursor = 0

        if level[0] == 0:
            raise EmptyIndexException()

        # visit the tree following sequence
        while cursor < len(sequence) and level_cursor < len(level[1]):
            if isinstance(level[1][level_cursor], dict):
                # found a branch
                if sequence[cursor] in level[1][level_cursor]:
                    # go on with this branch
                    level = level[1][level_cursor][sequence[cursor]]
                    level_cursor = 0
                else: # new branch
                    raise TripleNotFoundException(sequence)
            else: # is int
                if sequence[cursor] != level[1][level_cursor]:
                    sequence_positive = [id for id in sequence if id >= 0]
                    raise TripleNotFoundException(str(sequence)+self.tokenizer.decode(sequence_positive))
                else:
                    level_cursor += 1
            cursor += 1

        # get next_tokens
        if level_cursor >= len(level[1]):
            # end of triple
            _next_tokens = {}
        elif isinstance(level[1][level_cursor], dict):
            # multiple children
            _next_tokens = {k:v[0] for k,v in level[1][level_cursor].items()}
        else: # is int
            _next_tokens = {level[1][level_cursor]: level[0]}

        if len(_next_tokens) == 0:
            sequence_positive = [id for id in sequence if id >= 0]
            raise TripleNotFoundException(str(sequence) + self.tokenizer.decode(sequence_positive))

        return _next_tokens, {}

    def to_dict(self, sequence, numleaves, subtree=[]):
        if subtree:
            sequence = [*sequence, *subtree]
        return [numleaves, sequence]

    def count_leaves(self, tree=None, update_numleaves=False, fail_on_wrong_num=False):
        if tree is None:
            tree = self.tree
        numleaves = 0
        for cursor in range(len(tree[1])):
            if isinstance(tree[1][cursor], dict):
                for childtree in tree[1][cursor].values():
                    numleaves += self.count_leaves(childtree)
                break
        if numleaves == 0 and len(tree[1]) > 0:
            # no dict found
            numleaves = 1
        if numleaves != tree[0]:
            msg = f'WrongNumleaves: tree {tree[0]} != count {numleaves}'
            if fail_on_wrong_num:
                raise WrongNumleavesException(msg)
            else:
                print(msg)
        if update_numleaves:
            tree[0] = numleaves
        return numleaves

    def merge_dict(self, src, dst, dst_numleaves=None, update_numleaves=True):
        if dst_numleaves is None:
            dst_numleaves = sum(map(lambda x: x[0], dst.values()))
        for key in src:
            if key in dst:
                # sum numleaves
                old_numleaves = dst[key][0]
                new_numleaves = self.merge(src[key], dst[key], update_numleaves)

                if update_numleaves:
                    dst_numleaves = dst_numleaves - old_numleaves + new_numleaves
                    # assert dst_numleaves >= 0
            else:
                # If the key exists only in `src`, the value from the `src` object will be used.
                # found new branch to add
                dst[key] = src[key]
                if update_numleaves:
                    dst_numleaves += src[key][0]

        return dst_numleaves


    def merge(self, src, dst=None, update_numleaves=True):
        '''
        Assumption: no duplicates
        TODO: check for duplicates
        Does not support substrings (triples always end with .)
        '''
        if dst is None:
            dst = self.tree
        if len(src) > 0:
            cursor = 0
            while cursor < len(src[1]):
                if cursor == len(dst[1]):
                    # dst and src are single branch sequences so far
                    dst[1][cursor:] = src[1][cursor:]
                    # src includes dst
                    if update_numleaves or dst[0] == 0: # or initialize
                        dst[0] = src[0]
                    # no more to do
                    return dst[0]

                elif isinstance(dst[1][cursor], dict) or isinstance(src[1][cursor], dict):
                    # found branch division
                    if not isinstance(src[1][cursor], dict):
                        src[1][cursor] = {
                            src[1][cursor]: [src[0], src[1][cursor + 1:]], # src branch
                        }
                    if not isinstance(dst[1][cursor], dict):
                        dst[1][cursor] = {
                            dst[1][cursor]: [dst[0], dst[1][cursor + 1:]], # dst branch
                        }
                        del dst[1][cursor + 1:]
                    # now both are dict
                    new_numleaves = self.merge_dict(src[1][cursor], dst[1][cursor], dst_numleaves=dst[0], update_numleaves=update_numleaves)
                    if update_numleaves:
                        dst[0] = new_numleaves
                    return dst[0]
                else:
                    # int int
                    if dst[1][cursor] != src[1][cursor]:
                        # divide in 2 branches
                        dst[1][cursor] = {
                            dst[1][cursor]: [dst[0], dst[1][cursor + 1:]], # dst branch
                            src[1][cursor]: [src[0], src[1][cursor + 1:]] # src branch
                        }
                        del dst[1][cursor + 1:]
                        # src and dst are different branches
                        if update_numleaves:
                            dst[0] += src[0]
                        return dst[0]
                    # else:
                    #     pass # go on with the loop

                cursor += 1
        return dst[0]

class Cache():
    def __init__(self, cache_db):
        self.cache_db = cache_db

    # key --> (children, childrenleaves, subtree)
    def add(self, sequence):
        pass

    def next_tokens(self, sequence, **kwargs):
        next_tokens, subtree_cache = None, None
        return next_tokens, subtree_cache

class PostgresTrieIndex(Index):
    def __init__(self, postgresql_connection, table_name, tokenizer, switch_parameter : int = 7, rootkey : int = -100, configkey = DEFAULT_CONFIGKEY, cache: Cache = None, return_state = False, do_count_leaves=False):
        super().__init__(tokenizer)
        self.rootkey = rootkey
        self.configkey = configkey
        self.postgresql_connection = postgresql_connection
        self.switch_parameter = switch_parameter
        self.table_name = table_name
        self.cache = cache
        self.base_select_query = sql.SQL('SELECT id, children, subtree, numleaves, childrenleaves FROM {} WHERE key = %s::integer[];')
        self.select_query = self.base_select_query.format(sql.Identifier(self.table_name))
        self.return_state = return_state
        self.do_count_leaves = do_count_leaves # slower if true

        if self.postgresql_connection:
            self.get_config()

    def get_config(self):
        with self.postgresql_connection.cursor() as cursor:
            # doesnt work because of a type problem???
            cursor.execute(self.select_query, ([self.configkey,],))
            query_result = cursor.fetchall()

            config = None
            for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                if subtree:
                    config = pickle.loads(subtree)

            if config:
                print('Applying index config...')
                if 'switch_parameter' in config:
                    self.switch_parameter = config['switch_parameter']

    def flush_cache(self):
        if self.cache:
            self.cache.flush()

    def _merge_next_tokens(self, src, dst):
        if src is not None:
            for key in src:
                if key in dst:
                    dst[key] += src[key]
                else:
                    dst[key] = src[key]

    def next_tokens(self, sequence, **kwargs):
        state = kwargs['state']

        sequence = [self.rootkey] + sequence

        _next_tokens = {}
        if len(sequence) <= self.switch_parameter:
            postgres_seq = sequence[:self.switch_parameter] # max length of sequences indexed in postgres

            _next_tokens = self._next_tokens_from_postgresql(postgres_seq, state = state)
        else:
            # continue in the subtree (added previously)
            try:
                _next_tokens, _ = state.subtree_cache.next_tokens(sequence)
            except EmptyIndexException:
                pass
            except TripleNotFoundException:
                pass

        if len(_next_tokens) == 0:
            state.end_of_triple_reset()

        extra = {
            'found_subtree': len(sequence) > self.switch_parameter,
            }
        if self.return_state:
            extra['state'] = state

        return _next_tokens, extra

    def _next_tokens_from_postgresql(self, sequence, state):
        found_in_cache = False
        if self.cache:
            # key --> (children, childrenleaves, subtree)
            # better saving entire or incremental cache?
            # better entire to reduce computation
            _next_tokens, new_subtree_cache = self.cache.next_tokens(sequence)
            found_in_cache = _next_tokens is not None
            if found_in_cache:
                # do not reset the cache if not found
                if new_subtree_cache:
                    state.subtree_cache = new_subtree_cache

        if not found_in_cache:
            with self.postgresql_connection.cursor() as cursor:
                cursor.execute(self.select_query, (sequence,))
                query_result = cursor.fetchall()

            _next_tokens = {}
            if len(query_result) > 0:
                totalnumleaves = 0
                totalnumleaves_subtree = 0
                for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                    totalnumleaves += numleaves
                    for child, childleaves in zip(children, childrenleaves):
                        if (child not in _next_tokens):
                            _next_tokens[child] = 0
                        _next_tokens[child] += childleaves

                    if subtree:
                        totalnumleaves_subtree += numleaves
                        subtree = pickle.loads(subtree)
                        current_tree = state.subtree_cache.to_dict(sequence, numleaves, subtree)
                        if self.do_count_leaves and numleaves != state.subtree_cache.count_leaves(current_tree):
                            print('WARNING: number of leaves does not match after COUNT LEAVES.')

                        merge_numleaves = state.subtree_cache.merge(current_tree, update_numleaves=True)
                        numleaves_diff = totalnumleaves_subtree - merge_numleaves
                        if numleaves_diff != 0:
                            # TODO debug
                            # reduce all the upper numleaves by the difference
                            # WORKAROUND to solve duplicates (or wrong count) problem
                            print('WARNING: number of leaves does not match.')

            # TODO maybe do it only when there is no subtree cache to save space and bandwidth
            # (first calls are the slowest)
            if self.cache:
                try:
                    self.cache.add(sequence, _next_tokens, state.subtree_cache)
                except Exception as e:
                    print('WARNING: failed to populate cache', e)

        # print(_next_tokens)
        return _next_tokens

class HTTPPostgresError(Exception):
    pass

def deserialize(data):
    return pickle.loads(data)

def serialize(obj):
    return pickle.dumps(obj)

class HTTPPostgresTrieIndex(PostgresTrieIndex):
    def __init__(self, table_name, base_url: str, rootkey : int = -100, switch_parameter : int = 7, cache: Cache = None, return_state = False, do_count_leaves=False, rootcert=None, timeout=15, retry=5):
        super().__init__(rootkey, None, switch_parameter, table_name, cache, return_state, do_count_leaves)
        # self.select_query = sql.SQL('SELECT id, children, subtree, numleaves, childrenleaves FROM {} WHERE key = %s;').format(sql.Identifier(self.table_name))
        self.base_url = base_url[:-1] if base_url.endswith('/') else base_url
        self.select_url = f'/select'
        self.rootcert = rootcert
        self.timeout = timeout
        self.retry = retry

        self.session = requests.Session()

        retries = Retry(total=self.retry)

        self.session.mount(self.base_url, HTTPAdapter(max_retries=retries))

    def _next_tokens_from_postgresql(self, sequence, state):
        found_in_cache = False
        if self.cache:
            # key --> (children, childrenleaves, subtree)
            # better saving entire or incremental cache?
            # better entire to reduce computation
            _next_tokens, new_subtree_cache = self.cache.next_tokens(sequence)
            found_in_cache = _next_tokens is not None
            if found_in_cache:
                # do not reset the cache if not found
                if new_subtree_cache:
                    state.subtree_cache = new_subtree_cache

        if not found_in_cache:
            data = serialize(dict(sequence=sequence,table_name=self.table_name))
            response = self.session.post(self.base_url + self.select_url,
                data=data,
                headers={'Content-Type': 'application/octet-stream'},
                verify=self.rootcert,
                timeout=self.timeout,
            )
            if not response.ok:
                raise HTTPPostgresError(response.text)
            query_result = deserialize(response.content)

            _next_tokens = {}
            if len(query_result) > 0:
                totalnumleaves = 0
                totalnumleaves_subtree = 0
                for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                    totalnumleaves += numleaves
                    for child, childleaves in zip(children, childrenleaves):
                        if (child not in _next_tokens):
                            _next_tokens[child] = 0
                        _next_tokens[child] += childleaves

                    if subtree:
                        totalnumleaves_subtree += numleaves
                        subtree = pickle.loads(subtree)
                        current_tree = state.subtree_cache.to_dict(sequence, numleaves, subtree)
                        if self.do_count_leaves and numleaves != state.subtree_cache.count_leaves(current_tree):
                            print('WARNING: number of leaves does not match after COUNT LEAVES.')

                        merge_numleaves = state.subtree_cache.merge(current_tree, update_numleaves=True)
                        numleaves_diff = totalnumleaves_subtree - merge_numleaves
                        if numleaves_diff != 0:
                            # TODO debug
                            # reduce all the upper numleaves by the difference
                            # WORKAROUND to solve duplicates (or wrong count) problem
                            print('WARNING: number of leaves does not match.')

            # TODO maybe do it only when there is no subtree cache to save space and bandwidth
            # (first calls are the slowest)
            if self.cache:
                try:
                    self.cache.add(sequence, _next_tokens, state.subtree_cache)
                except Exception as e:
                    print('WARNING: failed to populate cache', e)

        return _next_tokens

class PostgresIngestIndex(PostgresTrieIndex, DictIndex):
    '''
    Use only for ingest time
    '''
    def __init__(self, tokenizer, rootkey : int, switch_parameter : int, table_name, configkey : int):
        postgresql_connection = '' # inference
        PostgresTrieIndex.__init__(
            self,
            postgresql_connection=postgresql_connection,
            table_name=table_name,
            tokenizer=tokenizer,
            switch_parameter=switch_parameter,
            rootkey=rootkey,
            configkey=configkey,
        )
        DictIndex.__init__(self, tokenizer=tokenizer)

        self.create_table_query = sql.SQL('''CREATE TABLE IF NOT EXISTS {} (
            id BIGINT GENERATED ALWAYS AS IDENTITY,
            key INT[] NOT NULL,
            children INT[],
            numleaves INT,
            childrenleaves INT[],
            subtree BYTEA
        );''').format(sql.Identifier(self.table_name))
        self.create_pkey_query = sql.SQL('ALTER TABLE {} ADD PRIMARY KEY (id);').format(
            sql.Identifier(self.table_name))
        self.create_index_query = sql.SQL('CREATE INDEX {} ON {} USING BTREE (key);').format(
            sql.Identifier('idx_key_btree_' + self.table_name), sql.Identifier(self.table_name))
        self.drop_pkey_query = sql.SQL('ALTER TABLE {} DROP CONSTRAINT IF EXISTS {};').format(
            sql.Identifier(self.table_name), sql.Identifier(self.table_name + '_pkey'))
        self.drop_index_query = sql.SQL('DROP INDEX IF EXISTS {};').format(
            sql.Identifier('idx_key_btree_' + self.table_name))
        self.check_indexes_query = sql.SQL("SELECT count(*) FROM pg_indexes WHERE tablename = '{}';").format(
            sql.Identifier(self.table_name))

        self.truncate_query = sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(self.table_name))

        self.copy_query = sql.SQL('''COPY {} (key, children, numleaves, childrenleaves, subtree)
            FROM STDIN WITH (FREEZE)''').format(sql.Identifier(self.table_name))
        
    def get_config_row(self):
        config = {
            'switch_parameter': self.switch_parameter
        }
        return [self.configkey], None, None, None, pickle.dumps(config)

    def get_rows(self):
        # iterative depth first traversal with a stack
        # level[0] is numleaves
        key = [self.rootkey]
        stack = [(key, self.tree)]
        while len(stack) > 0:
            key, level = stack.pop()
            if len(level[1]) > 0: # otherwise nothing to do
                if isinstance(level[1][0], dict):
                    # branch here
                    children = list(level[1][0].keys())
                    childrenleaves = [level[1][0][child][0] for child in children]
                    next_levels = [level[1][0][child] for child in children]
                else: # is int -> only 1 child
                    children = [level[1][0]]
                    childrenleaves = [level[0]] # same numleaves as parent
                    next_levels = [[level[0], level[1][1:]]]

                if len(key) >= self.switch_parameter:
                    yield key, children, level[0], childrenleaves, pickle.dumps(level[1])
                else:
                    for child, next_level in zip(children, next_levels):
                        stack.append((key + [child], next_level))
                    if len(children) > 0:
                        # skip adding empty keys to save space
                        yield key, children, level[0], childrenleaves, None
            else:
                print('Found empty tree.')

class ConstrainedStateList():
    # states is list of list [num_batches, num_beams]
    def __init__(self, states, num_beams = 1, num_batches = 1, debug=False, debug_tokenizer=None):
        self.states = states
        assert isinstance(states, list) and isinstance(states[0], list), 'ERROR: states is not a list of lists'
        assert len(states) == num_batches and len(states[0]) == num_beams, 'ERROR: states size does not match num_batches or num_beams'
        self.num_beams = num_beams
        self.num_batches = num_batches # used for computing beam id
        self.beam_idx = [] # torch.tensor([-1]*num_batches*num_beams).view(num_batches,num_beams,1) # running beam idx

        self.debug = debug
        self.debug_tokenizer = debug_tokenizer
        assert not self.debug or self.debug_tokenizer is not None

        self.num_permutations = 0 # for debugging

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # TODO test # seems it is not working as expected
            batch_key, beam_key = key

            if isinstance(batch_key, int) and isinstance(beam_key, int):
                return self.states[batch_key][beam_key]
            else:
                # Convert int to slice to normalize
                if isinstance(batch_key, int):
                    batch_key = slice(batch_key, batch_key + 1)
                if isinstance(beam_key, int):
                    beam_key = slice(beam_key, beam_key + 1)

                # Slice the states
                sliced_states = [row[beam_key] for row in self.states[batch_key]]

                new_num_batches = len(sliced_states)
                new_num_beams = len(sliced_states[0]) if sliced_states else 0

                return ConstrainedStateList(
                    sliced_states,
                    num_beams=new_num_beams,
                    num_batches=new_num_batches
                )

        elif isinstance(key, slice):
            return ConstrainedStateList(
                self.states[key],
                num_beams=self.num_beams,
                num_batches=len(self.states[key])
            )
        elif isinstance(key, int):
            return self.states[key]
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def __len__(self):
        return len(self.states) * len(self.states[0])

    def reset(self):
        for batch in self.states:
            for state in batch:
                state.reset()

    def get_batch_idx(self, idx):
        return int(idx // self.num_beams)
    
    def get_beam_idx(self, idx):
        return int(idx % self.num_beams)
    
    def get_last_beam_z(self):
        z = -1
        ids = (self.beam_idx[0, 0] != -1).nonzero(as_tuple=False).squeeze()
        if ids.numel() > 0:
            if ids.dim() == 0:
                z = ids.item()
            else:
                z = ids[-1].item()
        return z

    def beam_permutation(self):
        if len(self.beam_idx) > 0: # ignore first call
            assert self.beam_idx.shape[0] * self.beam_idx.shape[1] == self.num_beams * self.num_batches, f'ERROR: beam_idx size unexpected: {len(self.beam_idx)} != {self.num_beams} * {self.num_batches}'
            # copies = self[:,:] # new object
            copies = []
            for batch_i in range(self.num_batches):
                batch_copies = []
                for beam_i in range(self.num_beams):
                    batch_copies.append(self[batch_i, beam_i].dump())
                copies.append(batch_copies)
            # copies = [[self[batch_i, beam_i].dump() for beam_i in range(self.num_beams)] for batch_i in range(self.num_batches)]
            last_beam_z = self.get_last_beam_z()
            # skip first call
            if last_beam_z >= 0:
                if self.debug:
                    for batch_idx in range(self.beam_idx.shape[0]):
                        for num_beam in range(self.beam_idx.shape[1]):
                            print((batch_idx, num_beam), end='')
                            self.states[batch_idx][num_beam].print_debug(self.debug_tokenizer)

                for batch_idx in range(self.beam_idx.shape[0]):
                    for num_beam in range(self.beam_idx.shape[1]):
                        replacement_idx = self.beam_idx[batch_idx, num_beam, last_beam_z]
                        replacement_batch_idx = self.get_batch_idx(replacement_idx)
                        local_beam_idx = self.get_beam_idx(replacement_idx)
                        assert replacement_batch_idx == batch_idx, f'ERROR: permutating between different batches! {replacement_batch_idx} --> {batch_idx}, with num_beams {self.num_beams}. replacement_idx {replacement_idx}'
                        # copy only when to change
                        if num_beam != local_beam_idx:
                            # self.states[batch_idx][num_beam].copy(copies[batch_idx][local_beam_idx], copy=True)
                            self.states[batch_idx][num_beam].load(copies[batch_idx][local_beam_idx], copy=True)
                            self.num_permutations += 1
                            if self.debug:
                                print(f'permutation {self.num_permutations}: ({batch_idx},{local_beam_idx}) into {batch_idx}{num_beam}')

"""
Pattern should be recognized as soon as it is generated. Usually you want to end it with $
"""
class PatternConstrainedState():
    def __init__(self, pattern, tokenizer, cache_index, subtree_cache, state=0, debug=False, regex_window=10) -> None:

        self.NORMAL_GENERATION = 0 # even numbers for normal
        self.CONSTRAINED_GENERATION = 1 # odd numbers for constrained

        self.token_ids = [] # keep all the token ids

        self.tokenizer = tokenizer
        self.regex_window = regex_window # regex will be performed on the last N tokens

        # if the switch pattern is finally found --> CONSTRAINED_GENERATION
        self.pattern = pattern

        self.state = state

        self.cursor = 0 # how many tokens since last change in state

        self.cache_index = cache_index
        self.generated_triples = []

        self.subtree_cache = subtree_cache

        self._first_call = True

        self.debug = debug
        self.debug_history = []

    def print_debug(self, tokenizer, print_class=False, end_with_newline=True):
        if print_class:
            print('{} '.format(self), end='')
        # for item in self.debug_history:
        #     print('{} ({}) --> {}, '.format(tokenizer.decode(item['token']), item['token'], item['state']), end='')
        if end_with_newline:
            print()

    def first_call(self):
        if self._first_call:
            self._first_call = False
            return True
        else:
            return False

    def cache_add(self, sequence):
        self.cache_index.add(sequence, new_leaf=True)
        # removing end of triple
        self.generated_triples.append(sequence[:-1])

    def is_constrained(self):
        return self.state % 2 == self.CONSTRAINED_GENERATION

    def end_of_triple_reset(self):
        self.subtree_cache.reset()
        # reset to normal generation
        self.state = 0

    def reset(self):
        self.state = 0
        self.token_ids = []
        self.history = ()
        self.cursor = 0
        self.generated_triples = []
        self.cache_index.reset()
        self.end_of_triple_reset()

    def copy(self, other, copy=True):
        self.pattern = other.pattern
        self.state = other.state
        self.tokenizer = other.tokenizer
        self.regex_window = other.regex_window
        self.token_ids = deepcopy(other.token_ids)

        self.history = other.history  # Assuming it's immutable or should be shallow copied
        self.cursor = other.cursor

        self.generated_triples = other.generated_triples.copy() if copy else other.generated_triples
        self.cache_index = deepcopy(other.cache_index) if copy else other.cache_index
        self.subtree_cache = deepcopy(other.subtree_cache) if copy else other.subtree_cache

        self.debug = other.debug
        self.debug_history = deepcopy(other.debug_history) if copy else other.debug_history


    def update(self, new_token):
        state = self.state
        self.token_ids.append(new_token)
        self.cursor += 1

        text = self.tokenizer.decode(self.token_ids[-self.regex_window:])

        _match = text.endswith(self.pattern)
        if _match:
            state = self.CONSTRAINED_GENERATION

        self._update_state(state)

        if self.debug:
            self.debug_history.append({
                'state': self.state,
                'token': new_token,
                'token_ids': deepcopy(self.token_ids)
            })

    def _update_state(self, state, initial_cursor = 0):
        if state != self.state:
            self.history = (self.state, self.cursor)

            self.state = state
            self.cursor = initial_cursor
            self.token_ids = []

    def get_cursor(self):
        return self.cursor

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, index, states, tokenizer=None, error_strategy=0, avoid_duplicates=True):
        self.index = index
        self.states = states
        self.error_strategy = error_strategy
        self.avoid_duplicates = avoid_duplicates

        self.ERROR_STRATEGY_WARN = 0
        self.ERROR_STRATEGY_FAIL = 1

        self.tokenizer=tokenizer # for debugging

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == len(self.states), \
            f'Error: number of states ({len(self.states)}) should match `num_batches * num_beams` ({input_ids.shape[0]})'

        self.states.beam_permutation()

        # TODO create a mask of zeros same shape as scores and same device
        mask = torch.zeros_like(scores)

        for i in range(input_ids.shape[0]):
            batch_idx = self.states.get_batch_idx(i)
            beam_i = self.states.get_beam_idx(i)
            sequence = input_ids[i].tolist()

            if not self.states[batch_idx, beam_i].first_call():
                last_token = sequence[-1]
                self.states[batch_idx, beam_i].update(last_token)

            if self.states[batch_idx, beam_i].is_constrained(): # odd number means constrained generation
                # constrained generation
                mask[i] = -math.inf # set for all tokens by default
                constrain_generation_sequence = sequence[len(sequence) - self.states[batch_idx, beam_i].get_cursor():]
                self.constrained_generation(
                    constrain_generation_sequence, mask, i, state=self.states[batch_idx, beam_i])

                # else:
                #     # normal generation
                #     # scores are not altered
                #     pass

        scores_processed = scores + mask

        return scores_processed

    def constrained_generation(self, sequence, mask: torch.FloatTensor, mask_idx, state):

        possible_tokens, _ = self.index.next_tokens(sequence, state = state)
        if self.avoid_duplicates:
            try:
                visited_tokens, _ = state.cache_index.next_tokens(sequence)
                # print(visited_tokens, end=' = ')
                state.cache_index.subtract_tokens(possible_tokens, visited_tokens)
                # print(possible_tokens)
            except EmptyIndexException:
                # ignore when the cache index is empty
                pass
            except TripleNotFoundException:
                # ignore if triple not in cache index
                pass

        possible_tokens = list(possible_tokens.keys()) # TODO transform subtract tokens in a prob modifier

        if len(possible_tokens) == 0:
            # end of constrained generation
            # send end of string
            generated_triple = sequence
            state.cache_add(generated_triple)
            # ensure to reset after eof triple
            state.end_of_triple_reset()
            # end of constrained generation
            mask[mask_idx, :] = 0
        else:
            mask[mask_idx, possible_tokens] = 0

