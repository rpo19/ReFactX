from transformers.generation.logits_process import LogitsProcessor
import torch
import pickle
from psycopg import sql
from copy import deepcopy
import requests
from requests.adapters import HTTPAdapter, Retry
import math

class EmptyIndexException(Exception):
    pass

class TripleNotFoundException(Exception):
    pass

class WrongNumleavesException(Exception):
    pass

class Index():
    def __init__(self, end_of_triple: int) -> None:
        self.end_of_triple = end_of_triple # e.g. "." to ensure the triple is ended and valid

    def config(self) -> dict:
        return dict(end_of_triple=self.end_of_triple)

    def triple_is_valid(self, sequence: list) -> bool:
        return sequence[-1] == self.end_of_triple

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

class DictIndex(Index):
    def __init__(self, end_of_triple, tree = None):
        super().__init__(end_of_triple)
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
            'end_of_triple': self.end_of_triple,
        }

    def copy(self):
        copy_of_index = DictIndex(self.end_of_triple, tree=deepcopy(self.tree))
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
                    raise TripleNotFoundException(sequence)
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

        if len(_next_tokens) == 0 and not self.triple_is_valid(sequence):
            # end of triple must match end_of_triple
            # otherwise the triple is not valid
            raise TripleNotFoundException(sequence)

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

    # key --> (children, childrenleaves, oneleaf, subtree)
    def add(self, sequence):
        pass

    def next_tokens(self, sequence, **kwargs):
        next_tokens, oneleaf_cache, subtree_cache = None, None, None
        return next_tokens, oneleaf_cache, subtree_cache

class PostgresTrieIndex(Index):
    def __init__(self, rootkey : int, end_of_triple: int, postgresql_connection, switch_parameter : int, table_name, cache: Cache = None, return_state = False, do_count_leaves=False):
        super().__init__(end_of_triple)
        self.rootkey = rootkey
        self.postgresql_connection = postgresql_connection
        self.switch_parameter = switch_parameter
        self.table_name = table_name
        self.cache = cache
        self.base_select_query = sql.SQL('SELECT id, children, subtree, numleaves, childrenleaves FROM {} WHERE key = %s;')
        self.select_query = self.base_select_query.format(sql.Identifier(self.table_name))
        self.return_state = return_state
        self.do_count_leaves = do_count_leaves # slower if true

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

        _next_tokens_cache = None
        try:
            _next_tokens_cache, _ = state.oneleaf_cache.next_tokens(sequence)
        except EmptyIndexException:
            pass
        except TripleNotFoundException:
            pass

        _next_tokens = {}
        if len(sequence) <= self.switch_parameter:
            postgres_seq = sequence[:self.switch_parameter] # max length of sequences indexed in postgres

            _next_tokens = self._next_tokens_from_postgresql(postgres_seq, state = state)
        else:
            # continue in the subtree
            try:
                _next_tokens, _ = state.subtree_cache.next_tokens(sequence)
            except EmptyIndexException:
                pass
            except TripleNotFoundException:
                pass

        self._merge_next_tokens(_next_tokens_cache, _next_tokens)

        if len(_next_tokens) == 0:
            if self.triple_is_valid(sequence):
                state.end_of_triple_reset()
            else:
                # end of triple must match end_of_triple
                # otherwise the triple is not valid
                raise TripleNotFoundException(sequence)

        extra = {
            'found_subtree': len(sequence) > self.switch_parameter,
            'tokens_from_oneleaf': _next_tokens_cache
            }
        if self.return_state:
            extra['state'] = state

        return _next_tokens, extra

    def _next_tokens_from_postgresql(self, sequence, state):
        found_in_cache = False
        if self.cache:
            # key --> (children, childrenleaves, oneleaf, subtree)
            # better saving entire or incremental cache?
            # better entire to reduce computation
            _next_tokens, new_oneleaf_cache, new_subtree_cache = self.cache.next_tokens(sequence)
            found_in_cache = _next_tokens is not None
            if found_in_cache:
                # do not reset the cache if not found
                if new_oneleaf_cache:
                    state.oneleaf_cache = new_oneleaf_cache
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
                totalnumleaves_oneleaf = 0
                for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                    totalnumleaves += numleaves
                    if numleaves == 1:
                        totalnumleaves_oneleaf += 1
                        # triple finish
                        # children is the entire triple
                        child = children[0]
                        if child not in _next_tokens:
                            _next_tokens[child] = 0
                        _next_tokens[child] += 1
                        # save the rest in cache
                        current_tree = state.oneleaf_cache.to_dict([*sequence, *children], numleaves)
                        merge_numleaves = state.oneleaf_cache.merge(current_tree, update_numleaves=True)
                    else:
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
                    self.cache.add(sequence, _next_tokens, state.oneleaf_cache, state.subtree_cache)
                except Exception as e:
                    print('WARNING: failed to populate cache', e)

        return _next_tokens

class HTTPPostgresError(Exception):
    pass

def deserialize(data):
    return pickle.loads(data)

def serialize(obj):
    return pickle.dumps(obj)

class HTTPPostgresTrieIndex(PostgresTrieIndex):
    def __init__(self, rootkey : int, end_of_triple: int, switch_parameter : int, table_name, base_url: str, cache: Cache = None, postgresql_connection = None, redis_connection = None, return_state = False, do_count_leaves=False, rootcert=None, timeout=15, retry=5):
        super().__init__(rootkey, end_of_triple, None, switch_parameter, table_name, cache, return_state, do_count_leaves)
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
            # key --> (children, childrenleaves, oneleaf, subtree)
            # better saving entire or incremental cache?
            # better entire to reduce computation
            _next_tokens, new_oneleaf_cache, new_subtree_cache = self.cache.next_tokens(sequence)
            found_in_cache = _next_tokens is not None
            if found_in_cache:
                # do not reset the cache if not found
                if new_oneleaf_cache:
                    state.oneleaf_cache = new_oneleaf_cache
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
                totalnumleaves_oneleaf = 0
                for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                    totalnumleaves += numleaves
                    if numleaves == 1:
                        totalnumleaves_oneleaf += 1
                        # triple finish
                        # children is the entire triple
                        child = children[0]
                        if child not in _next_tokens:
                            _next_tokens[child] = 0
                        _next_tokens[child] += 1
                        # save the rest in cache
                        current_tree = state.oneleaf_cache.to_dict([*sequence, *children], numleaves)
                        merge_numleaves = state.oneleaf_cache.merge(current_tree, update_numleaves=True)
                    else:
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
                    self.cache.add(sequence, _next_tokens, state.oneleaf_cache, state.subtree_cache)
                except Exception as e:
                    print('WARNING: failed to populate cache', e)

        return _next_tokens

class PostgresIngestIndex(PostgresTrieIndex, DictIndex):
    '''
    Use only for ingest time
    '''
    def __init__(self, rootkey : int, switch_parameter : int, table_name):
        end_of_triple = -1 # only needed during inference
        postgresql_connection = '' # inference
        PostgresTrieIndex.__init__(self, rootkey, end_of_triple, postgresql_connection, switch_parameter, table_name)
        DictIndex.__init__(self, end_of_triple)

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

    def get_rows(self):
        # iterative depth first traversal with a stack
        # level[0] is numleaves
        key = [self.rootkey]
        stack = [(key, self.tree)]
        while len(stack) > 0:
            key, level = stack.pop()
            if len(level[1]) > 0: # otherwise nothing to do
                if level[0] == 1: # 1 leaf -> children is the sequence to the end of triple
                    children = level[1]
                    childrenleaves = []
                    if len(children) > 0:
                        yield key, children, level[0], childrenleaves, None
                else:
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
    def __init__(self, states, pad_token_id, num_beams = 1, num_batches = 1):
        self.states = states
        assert isinstance(states, list) and isinstance(states[0], list), 'ERROR: states is not a list of lists'
        assert len(states) == num_batches and len(states[0]) == num_beams, 'ERROR: states size does not match num_batches or num_beams'
        self.num_beams = num_beams
        self.num_batches = num_batches # used for computing beam id
        self.beam_idx = [] # torch.tensor([-1]*num_batches*num_beams).view(num_batches,num_beams,1) # running beam idx
        self.beam_sent_finished = torch.tensor([False]*num_batches*num_beams).view(num_batches,num_beams) # place to save beam indexes permutation
        self.pad_token_id = pad_token_id

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
                    pad_token_id=self.pad_token_id,
                    num_beams=new_num_beams,
                    num_batches=new_num_batches
                )

        elif isinstance(key, slice):
            return ConstrainedStateList(
                self.states[key],
                pad_token_id=self.pad_token_id,
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
            self.beam_idx.shape[0] * self.beam_idx.shape[1] == self.num_beams * self.num_batches, f'ERROR: beam_idx size unexpected: {len(self.beam_idx)} != {self.num_beams} * {self.num_batches}'
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
                for batch_idx in range(self.beam_idx.shape[0]):
                    for num_beam in range(self.beam_idx.shape[1]):
                        if not self.beam_sent_finished[batch_idx, num_beam]:
                            replacement_idx = self.beam_idx[batch_idx, num_beam, last_beam_z]
                            replacement_batch_idx = self.get_batch_idx(replacement_idx)
                            local_beam_idx = self.get_beam_idx(replacement_idx)
                            assert replacement_batch_idx == batch_idx, f'ERROR: permutating between different batches! {replacement_batch_idx} --> {batch_idx}, with num_beams {self.num_beams}. replacement_idx {replacement_idx}'
                            # copy only when to change
                            if num_beam != local_beam_idx:
                                # self.states[batch_idx][num_beam].copy(copies[batch_idx][local_beam_idx], copy=True)
                                self.states[batch_idx][num_beam].load(copies[batch_idx][local_beam_idx], copy=True)

class ConstrainedState():
    def __init__(self, begin_pattern, end_pattern, cache_index, subtree_cache, oneleaf_cache, state=0) -> None:

        self.NORMAL_GENERATION = 0 # even numbers for normal
        self.CONSTRAINED_GENERATION = 1 # odd numbers for constrained

        # switching to constrain generation
        self.BEGIN_SWITCH = 2
        # if the switch pattern is finally found --> CONSTRAINED_GENERATION
        self.begin_pattern = begin_pattern # {'<': {'fact':{'>':{}}, '_<': {'fact':{'>':{}}}
        self.begin_pattern_current = self.begin_pattern
        # if end_pattern is found --> NORMAL_GENERATION
        self.end_pattern = end_pattern

        self.state = state

        self.history = () # (prev_state, prev_cursor)

        self.cursor = 0 # how many tokens since last change in state

        self.cache_index = cache_index
        self.generated_triples = []

        self.subtree_cache = subtree_cache
        self.oneleaf_cache = oneleaf_cache

        self._first_call = True

    def first_call(self):
        if self._first_call:
            self._first_call = False
            return True
        else:
            return False

    def cache_add(self, sequence):
        self.cache_index.add(sequence, new_leaf=True)
        self.generated_triples.append(sequence)

    def is_constrained(self):
        return self.state % 2 == self.CONSTRAINED_GENERATION

    def end_of_triple_reset(self):
        self.subtree_cache.reset()
        self.oneleaf_cache.reset()

    def reset(self):
        self.state = 0
        self.history = ()
        self.cursor = 0
        self.generated_triples = []
        self.cache_index.reset()
        self.end_of_triple_reset()

    def __json__(self, copy=True):
        return {
            'begin_pattern': self.begin_pattern,
            'begin_pattern_current': self.begin_pattern_current,
            'end_pattern': self.end_pattern,
            'state': self.state,
            'history': self.history,
            'cursor': self.cursor,
            'generated_triples': self.generated_triples.copy() if copy else self.generated_triples,
            'cache_index': self.cache_index.__json__(copy),
            'oneleaf_cache': self.oneleaf_cache.__json__(copy),
            'subtree_cache': self.subtree_cache.__json__(copy),
        }

    def from_json(self, data, copy=True):
        self.begin_pattern = data['begin_pattern']
        self.begin_pattern_current = data['begin_pattern_current']
        self.end_pattern = data['end_pattern']
        self.state = data['state']

        self.history = data['history']
        self.cursor = data['cursor']

        self.generated_triples = data['generated_triples'].copy() if copy else data['generated_triples']
        if copy:
            self.cache_index = DictIndex(end_of_triple=data['cache_index']['end_of_triple'], tree=deepcopy(data['cache_index']['tree']))
            self.subtree_cache = DictIndex(end_of_triple=data['subtree_cache']['end_of_triple'], tree=deepcopy(data['subtree_cache']['tree']))
            self.oneleaf_cache = DictIndex(end_of_triple=data['oneleaf_cache']['end_of_triple'], tree=deepcopy(data['oneleaf_cache']['tree']))
        else:
            self.cache_index = DictIndex(**data['cache_index'])
            self.subtree_cache = DictIndex(**data['subtree_cache'])
            self.oneleaf_cache = DictIndex(**data['oneleaf_cache'])

        return self

    def dump(self, copy=True):
        return {
            'begin_pattern': self.begin_pattern,
            'begin_pattern_current': self.begin_pattern_current,
            'end_pattern': self.end_pattern,
            'state': self.state,
            'history': self.history,
            'cursor': self.cursor,
            'generated_triples': self.generated_triples.copy() if copy else self.generated_triples,
            'cache_index': self.cache_index.copy() if copy else self.cache_index,
            'oneleaf_cache': self.oneleaf_cache.copy() if copy else self.oneleaf_cache,
            'subtree_cache': self.subtree_cache.copy() if copy else self.subtree_cache,
        }

    def load(self, data, copy=True):
        self.begin_pattern = data['begin_pattern']
        self.begin_pattern_current = data['begin_pattern_current']
        self.end_pattern = data['end_pattern']
        self.state = data['state']

        self.history = data['history']
        self.cursor = data['cursor']

        self.generated_triples = data['generated_triples'].copy() if copy else data['generated_triples']
        self.cache_index = deepcopy(data['cache_index']) if copy else data['cache_index']
        self.subtree_cache = deepcopy(data['subtree_cache']) if copy else data['subtree_cache']
        self.oneleaf_cache = deepcopy(data['oneleaf_cache']) if copy else data['oneleaf_cache']

    def copy(self, other, copy=True):
        self.begin_pattern = other.begin_pattern
        self.begin_pattern_current = other.begin_pattern_current
        self.end_pattern = other.end_pattern
        self.state = other.state

        self.history = other.history  # Assuming it's immutable or should be shallow copied
        self.cursor = other.cursor

        self.generated_triples = other.generated_triples.copy() if copy else other.generated_triples
        self.cache_index = deepcopy(other.cache_index) if copy else other.cache_index
        self.subtree_cache = deepcopy(other.subtree_cache) if copy else other.subtree_cache
        self.oneleaf_cache = deepcopy(other.oneleaf_cache) if copy else other.oneleaf_cache


    def update(self, new_token):
        rollback = False
        state = self.state
        self.cursor += 1
        if self.state == self.NORMAL_GENERATION:
            if new_token in self.begin_pattern_current:
                self.begin_pattern_current = self.begin_pattern_current[new_token]
                if len(self.begin_pattern_current) == 0: # reached pattern leaf
                    state = self.CONSTRAINED_GENERATION
                    self.begin_pattern_current = self.begin_pattern # reset pattern
                else:
                    state = self.BEGIN_SWITCH

        elif self.state == self.BEGIN_SWITCH:
            if new_token in self.begin_pattern_current:
                self.begin_pattern_current = self.begin_pattern_current[new_token]
                if len(self.begin_pattern_current) == 0: # reached pattern leaf
                    state = self.CONSTRAINED_GENERATION
                    self.begin_pattern_current = self.begin_pattern # reset pattern
            else:
                self.begin_pattern_current = self.begin_pattern # reset pattern
                rollback = True

        elif self.state == self.CONSTRAINED_GENERATION:
            if new_token == self.end_pattern:
                state = self.NORMAL_GENERATION

        if rollback:
            self._rollback()
        else:
            self._update_state(state)

    def _update_state(self, state, initial_cursor = 0):
        if state != self.state:
            self.history = (self.state, self.cursor)

            self.state = state
            self.cursor = initial_cursor

    def get_cursor(self):
        return self.cursor

    def _rollback(self):
        prev_state, prev_cursor = self.history
        self.history = (self.state, self.cursor)
        self.state = prev_state
        self.cursor = prev_cursor + self.cursor

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, index, end_token, states, tokenizer=None, error_strategy=0):
        self.index = index
        self.end_token = end_token
        self.states = states
        self.error_strategy = error_strategy

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
            if not self.states.beam_sent_finished[batch_idx, beam_i]:
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
            if not self.index.triple_is_valid(sequence):
                if self.error_strategy == self.ERROR_STRATEGY_FAIL:
                    raise TripleNotFoundException(sequence)
                elif self.error_strategy == self.ERROR_STRATEGY_WARN:
                    print('WARNING:', TripleNotFoundException(sequence))
            possible_tokens = [self.end_token]
            generated_triple = sequence + [self.end_token]
            state.cache_add(generated_triple)
            # ensure to reset after eof triple
            state.end_of_triple_reset()

        mask[mask_idx, possible_tokens] = 0

