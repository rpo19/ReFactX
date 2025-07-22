from transformers.generation.logits_process import LogitsProcessor
from transformers import StoppingCriteria
import torch
import pickle
from psycopg import sql

class EmptyIndexException(Exception):
    pass

class TripleNotFoundException(Exception):
    pass

class Index():
    def __init__(self) -> None:
        pass

    def next_tokens(self, sequence, **kwargs):
        token = 0
        numleaves = 10
        return {token: numleaves,}

class DictIndex(Index):
    def __init__(self, end_of_triple, tree = None):
        self.tree = None
        self.reset(tree)
        self.end_of_triple = end_of_triple

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

    def add(self, sequence, new_leaf=False):
        # could be replaced with to_dict and merge
        # but could be useful to avoid recursion
        level = self.tree
        cursor = 0
        level_cursor = 0
        prev_levels = [] # need to keep track to increase numleaves
        while cursor < len(sequence):
            if level[0] == 0:
                # level[0] = 1 # increment in the end
                level[1] = sequence[cursor:]
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
                        # increment all prev levels
                        prev_levels.append(level)
                        new_leaf = True

                        new_branch = [0, []]
                        level[1][level_cursor][sequence[cursor]] = new_branch
                        level = new_branch # continue with new branch: same as == 0
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
                        # increment all prev levels
                        prev_levels.append(level)
                        new_leaf = True

                        del level[1][level_cursor + 1:]
                        level = new_branch # continue with new branch: same as == 0
                        level_cursor = 0
                    else:
                        level_cursor += 1
            cursor += 1

        if new_leaf:
            level[0] += 1
            for level in prev_levels:
                level[0] += 1

    def next_tokens(self, sequence):
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
                    raise TripleNotFoundException()
            else: # is int
                if sequence[cursor] != level[1][level_cursor]:
                    raise TripleNotFoundException()
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

        if len(_next_tokens) == 0 and sequence[-1] != self.end_of_triple:
            # end of triple must match end_of_triple
            # otherwise the triple is not valid
            raise TripleNotFoundException(sequence)

        return _next_tokens

    def to_dict(self, sequence, numleaves, subtree=[]):
        if subtree:
            sequence = [*sequence, *subtree]
        return [numleaves, sequence]

    def count_leaves(self, tree=None, update_numleaves=False):
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

class PostgresTrieIndex(Index):
    def __init__(self, rootkey : int, end_of_triple: int, postgresql_connection, switch_parameter : int, table_name):
        self.rootkey = rootkey
        self.end_of_triple = end_of_triple # e.g. "." to ensure the triple is ended and valid
        self.postgresql_connection = postgresql_connection
        self.switch_parameter = switch_parameter+1 # counting the rootkey
        self.table_name = table_name
        self.select_query = sql.SQL('SELECT id, children, subtree, numleaves, childrenleaves FROM {} WHERE key = %s;').format(sql.Identifier(self.table_name))
        self.subtree_cache = DictIndex(self.end_of_triple)
        self.oneleaf_cache = DictIndex(self.end_of_triple)

    def _merge_next_tokens(self, src, dst):
        if src is not None:
            for key in src:
                if key in dst:
                    pass # keep value in dst
                else:
                    dst[key] = src[key]

    def next_tokens(self, sequence, **kwargs):
        sequence = [self.rootkey] + sequence

        _next_tokens_cache = None
        try:
            _next_tokens_cache = self.oneleaf_cache.next_tokens(sequence)
        except EmptyIndexException:
            pass
        except TripleNotFoundException:
            pass

        _next_tokens = {}
        if len(sequence) <= self.switch_parameter:
            postgres_seq = sequence[:self.switch_parameter] # max length of sequences indexed in postgres

            _next_tokens = self._next_tokens_from_postgresql(postgres_seq)
        else:
            # continue in the subtree
            # subtree_seq = sequence[self.switch_parameter:]

            # TODO really need to reset subtree_cache all the time
            # probably doesnt change much

            try:
                _next_tokens = self.subtree_cache.next_tokens(sequence)
            except EmptyIndexException:
                pass

        self._merge_next_tokens(_next_tokens_cache, _next_tokens)

        if len(_next_tokens) == 0:
            if sequence[-1] == self.end_of_triple:
                self.oneleaf_cache.reset()
                self.subtree_cache.reset()
            else:
                # end of triple must match end_of_triple
                # otherwise the triple is not valid
                raise TripleNotFoundException(sequence)

        return _next_tokens

    def _next_tokens_from_postgresql(self, sequence):
        with self.postgresql_connection.cursor() as cursor:
            cursor.execute(self.select_query, (sequence,))
            query_result = cursor.fetchall()

        _next_tokens = {}
        if len(query_result) > 0:
            totalnumleaves = 0
            for row, (query_id, children, subtree, numleaves, childrenleaves) in enumerate(query_result):
                totalnumleaves += numleaves
                if numleaves == 1:
                    # triple finish
                    # children is the entire triple
                    child = children[0]
                    if child not in _next_tokens:
                        _next_tokens[child] = 0
                    _next_tokens[child] += 1
                    # save the rest in cache
                    current_tree = self.oneleaf_cache.to_dict([*sequence, *children], numleaves)
                    merge_numleaves = self.oneleaf_cache.merge(current_tree, update_numleaves=True)

                else:
                    for child, childleaves in zip(children, childrenleaves):
                        if child not in _next_tokens:
                            _next_tokens[child] = 0
                        _next_tokens[child] += childleaves

                    if subtree:
                        subtree = pickle.loads(subtree)
                        current_tree = self.subtree_cache.to_dict(sequence, numleaves, subtree)
                        # if not numleaves == self.subtree_cache.count_leaves(current_tree):
                        #     pass
                        merge_numleaves = self.subtree_cache.merge(current_tree, update_numleaves=True)
                        numleaves_diff = numleaves - merge_numleaves
                        if numleaves_diff != 0:
                            # TODO debug
                            # reduce all the upper numleaves by the difference
                            # WORKAROUND to solve duplicates (or wrong count) problem
                            pass

        return _next_tokens

class ConstrainedStateList():
    def __init__(self, states):
        self.states = states

        self.beam_idx = [] # place to save beam indexes permutation

    def __getitem__(self, arg):
        return self.states[arg]

    def __len__(self):
        return len(self.states)

    def reset(self):
        for state in self.states:
            state.reset()

    def beam_permutation(self):
        copies = [state.dump(copy=False) for state in self.states]
        for i, (state, beam_i) in enumerate(zip(self.states, self.beam_idx)):
            if i != beam_i:
                state.load(copies[beam_i], copy=True)

class ConstrainedState():
    def __init__(self, begin_pattern, end_pattern, cache_index, state=0) -> None:

        self.NORMAL_GENERATION = 0 # even numbers for normal
        self.CONSTRAINED_GENERATION = 1 # odd numbers for constrained

        # switching to constrain generation
        self.BEGIN_SWITCH = 2
        # if the switch pattern is finally found --> CONSTRAINED_GENERATION
        self.begin_pattern = begin_pattern
        # if end_pattern is found --> NORMAL_GENERATION
        self.end_pattern = end_pattern

        self.state = state

        self.history = () # (prev_state, prev_cursor)

        self.cursor = 0 # how many tokens since last change in state

        self.cache_index = cache_index
        self.generated_triples = []

    def cache_add(self, sequence):
        self.cache_index.add(sequence, new_leaf=True)
        self.generated_triples.append(sequence)

    def is_constrained(self):
        return self.state % 2 == self.CONSTRAINED_GENERATION

    def reset(self):
        self.state = 0
        self.history = ()
        self.cursor = 0

    def dump(self, copy=True):
        return {
            "begin_pattern": self.begin_pattern,
            "end_pattern": self.end_pattern,
            "state": self.state,
            "history": self.history,
            "cursor": self.cursor,
        }

    def load(self, data, copy=True):
        self.begin_pattern = data["begin_pattern"]
        self.end_pattern = data["end_pattern"]
        self.state = data["state"]

        self.history = data["history"]
        self.cursor = data["cursor"]

    def copy(self, other):
        self.begin_pattern = other.begin_pattern
        self.end_pattern = other.end_pattern
        self.state = other.state

        self.history = other.history  # Assuming it's immutable or should be shallow copied
        self.cursor = other.cursor

    def update(self, new_token):
        rollback = False
        state = self.state
        self.cursor += 1
        if self.state == self.NORMAL_GENERATION:
            if new_token == self.begin_pattern[0]:
                if len(self.begin_pattern) == 1:
                    state = self.CONSTRAINED_GENERATION
                else:
                    state = self.BEGIN_SWITCH

        elif self.state == self.BEGIN_SWITCH:
            if new_token == self.begin_pattern[self.cursor]:
                if self.cursor == len(self.begin_pattern) - 1:
                    state = self.CONSTRAINED_GENERATION
            else:
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
    def __init__(self, index, end_token, states, tokenizer=None):
        self.index = index
        self.end_token = end_token
        self.states = states
        self.first_call = True

        self.tokenizer=tokenizer # for debugging

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        assert input_ids.shape[0] == len(self.states), 'Error: number of states should match `batch_size * num_beams`'

        self.states.beam_permutation()

        for i in range(input_ids.shape[0]):
            input_sequence = input_ids[i].tolist()

            if self.first_call:
                self.first_call = False
            else:
                last_token = input_sequence[-1]
                self.states[i].update(last_token)

            if self.states[i].is_constrained(): # odd number means constrained generation
                # constrained generation
                constrain_generation_sequence = input_sequence[len(input_sequence) - self.states[i].get_cursor():]
                scores[[i],:] = self.constrained_generation(
                    constrain_generation_sequence, scores[[i],:], state=self.states[i])
            # else:
            #     # normal generation
            #     # scores are not altered
            #     pass

            if self.tokenizer:
                print(i, self.states[i].state, self.states[i].is_constrained(), [self.tokenizer.convert_ids_to_tokens(t) for t in scores[i].argsort(descending=True)[:3].tolist()])

        return scores

    def subtract_tokens(self, tokens_a, tokens_b):
        for token in list(tokens_a.keys()):
            if token in tokens_b:
                diff = tokens_a[token] - tokens_b[token] # TODO transform subtract tokens in a prob modifier
                tokens_a[token] = diff
                if diff == 0:
                    # branch completely visited
                    # forbid generating again same triple
                    del tokens_a[token]

    def constrained_generation(self, input_sequence, scores: torch.FloatTensor, state):

        possible_tokens = self.index.next_tokens(input_sequence)
        # print(possible_tokens, end=' - ')
        try:
            visited_tokens = state.cache_index.next_tokens(input_sequence)
            # print(visited_tokens, end=' = ')
            #self.subtract_tokens(possible_tokens, visited_tokens)
            # print(possible_tokens)
        except EmptyIndexException:
            # ignore when the cache index is empty
            # print()
            pass
        except TripleNotFoundException:
            # ignore if triple not in cache index
            # print()
            pass

        possible_tokens = list(possible_tokens.keys()) # TODO transform subtract tokens in a prob modifier

        if len(possible_tokens) == 0:
            # end of constrained generation
            # send end of string
            possible_tokens = [self.end_token]
            generated_triple = input_sequence + [self.end_token]
            state.cache_add(generated_triple)

        possible_scores = scores[:, possible_tokens]

        scores[:,:] = -float('inf')
        scores[:, possible_tokens] = possible_scores

        return scores

class GetAnswer(StoppingCriteria):
    # strategy=all strategy=any. strategy can be all or any python functions
    def __init__(self, answer, eofanswer, strategy=all):
        self.prompt = None
        self.answer = answer
        self.eofanswer = set(eofanswer)
        self.strategy = strategy

    def __call__(self, input_ids, scores, **kwargs):
        outcome = self.strategy(
            self.get_answer(input_ids[i].tolist(), return_answer=False) for i in range(input_ids.shape[0]))
        return outcome

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_answer(self, sequence, return_answer=True):
        sequence = sequence[len(self.prompt):] # remove prompt
        answer_cursor = 0
        answer_tokens = []
        answer_is_complete = False
        token_id = 0
        while token_id < len(sequence):
            token = sequence[token_id]
            token_id += 1
            if token == self.answer[answer_cursor]:
                answer_cursor += 1
                if answer_cursor >= len(self.answer):
                    #answer_found = True
                    break
        #if answer_found:
        while token_id < len(sequence):
            token = sequence[token_id]
            token_id += 1
            if token in self.eofanswer:
                answer_is_complete = True
                break
            else:
                answer_tokens.append(token)

        outcome = (answer_is_complete, answer_tokens) if return_answer else answer_is_complete

        return outcome