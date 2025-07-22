from transformers.generation.logits_process import LogitsProcessor
from transformers import StoppingCriteria
import torch
import pickle
from psycopg import sql
import marisa_trie
class Index():
    def __init__(self) -> None:
        pass

    def next_tokens(self, sequence, **kwargs):
        return set()

class PostgresTrieIndex(Index):
    def __init__(self, rootkey : int, end_of_triple: int, postgresql_connection, switch_parameter : int, table_name):
        self.rootkey = rootkey
        self.end_of_triple = end_of_triple # e.g. "." to ensure the triple is ended and valid
        self.postgresql_connection = postgresql_connection
        self.switch_parameter = switch_parameter+1 # counting the rootkey
        self.table_name = table_name
        self.select_query = sql.SQL('SELECT children, subtree, numleaves, childrenleaves FROM {} WHERE key = %s;').format(sql.Identifier(self.table_name))

    def next_tokens(self, sequence, **kwargs):
        sequence = [self.rootkey] + sequence
        postgres_seq = sequence[:self.switch_parameter] # max length of sequences indexed in postgres

        _next_tokens, subtree = self._next_tokens_from_postgresql(postgres_seq)

        if len(_next_tokens) == 0 and sequence[-1] != self.end_of_triple:
            # end of triple must match end_of_triple
            # otherwise the triple is not valid
            raise KeyError('Triple not found.', sequence)

        if len(sequence) >= self.switch_parameter:
            # continue in the subtree
            subtree_seq = sequence[self.switch_parameter:]

            _next_tokens = self._next_tokens_from_subtree(subtree, subtree_seq)

        return _next_tokens

    def _next_tokens_from_subtree(self, subtree, subtree_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `sequence`.
        """
        start = subtree

        for current_token in subtree_seq:
            if current_token not in start:
                start = {}
                break
            start = start[current_token][1]

        _next_tokens = {k:v[0] for k,v in start.items()}

        return _next_tokens

    def _merge(self, dst, src):
        for key in src:
            if key in dst:
                # sum numleaves
                dst[key][0] += src[key][0]
                self._merge(dst[key][1], src[key][1])
            else:
                # If the key exists only in `src`, the value from the `src` object will be used.
                dst[key] = src[key]

    def _next_tokens_from_postgresql(self, sequence):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `sequence`.
        """
        with self.postgresql_connection.cursor() as cursor:
            cursor.execute(self.select_query, (sequence,))
            query_result = cursor.fetchall()

        _next_tokens = {}
        merged_subtree = {}
        subtrees_list = []
        if len(query_result) > 0:
            for children, subtree, numleaves, childrenleaves in query_result:
                for child, childleaves in zip(children, childrenleaves):
                    if child not in _next_tokens:
                        _next_tokens[child] = 0
                    _next_tokens[child] += childleaves
                if subtree:
                    subtree = pickle.loads(subtree)
                    self._merge(merged_subtree, subtree)

        return _next_tokens, merged_subtree

class ConstrainedStateList():
    def __init__(self, number, *args, **kwargs):
        self.states = []
        for i in range(number):
            self.states.append(ConstrainedState(*args, **kwargs))

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
    def __init__(self, begin_pattern, end_pattern, state=0) -> None:

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
                self._rollback()

        elif self.state == self.CONSTRAINED_GENERATION:
            if new_token == self.end_pattern:
                state = self.NORMAL_GENERATION

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
                    constrain_generation_sequence, scores[[i],:])
            # else:
            #     # normal generation
            #     # scores are not altered
            #     pass

            if self.tokenizer:
                print(i, constrained, [self.tokenizer.convert_ids_to_tokens(t) for t in scores[i].argsort(descending=True)[:10].tolist()])

        return scores

    def constrained_generation(self, input_sequence, scores: torch.FloatTensor, **kwargs):

        possible_tokens = self.index.next_tokens(input_sequence)
        possible_tokens = list(possible_tokens.keys()) # TODO

        if len(possible_tokens) == 0:
            # end of constrained generation
            # send end of string
            possible_tokens = [self.end_token]

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