from transformers.generation.logits_process import LogitsProcessor
from transformers import StoppingCriteria
import torch
import pickle
from mergedeep import merge
from psycopg import sql

class PostgresTrieIndex:
    def __init__(self, rootkey : int, postgresql_connection, switch_parameter : int, table_name):
        self.rootkey = rootkey
        self.postgresql_connection = postgresql_connection
        self.switch_parameter = switch_parameter+1 # counting the rootkey
        self.table_name = table_name
        self.select_query = sql.SQL('SELECT children, subtree FROM {} WHERE key = %s;').format(sql.Identifier(self.table_name))

    def tken(self, token):
        # token encode
        encoded = token.to_bytes(2, byteorder='big', signed=False)
        return encoded

    def tkde(self, bbytes):
        # token decode
        decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
        return decoded

    def next_tokens(self, current_seq):
        current_seq = [self.rootkey] + current_seq
        postgres_seq = current_seq[:self.switch_parameter] # max length of sequences indexed in postgres

        postgres_byte_seq = b''.join(map(self.tken, postgres_seq))
        _next_tokens, subtree = self._next_tokens_from_postgresql(postgres_byte_seq)

        if len(current_seq) >= self.switch_parameter:
            # continue in the subtree
            subtree_seq = current_seq[self.switch_parameter:]

            _next_tokens = self._next_tokens_from_subtree(subtree, subtree_seq)

        return _next_tokens

    def _next_tokens_from_subtree(self, subtree, subtree_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = subtree

        for current_token in subtree_seq:
            if current_token not in start:
                start = {}
                break
            start = start[current_token]

        return set(start.keys())

    def _load_merge_subtrees(self, subtrees_list):
        merged_subtree = {}
        for bsubtree in subtrees_list:
            if bsubtree is not None:
                subtree = pickle.loads(bsubtree)
                merge(merged_subtree, subtree)
        return merged_subtree

    def _next_tokens_from_postgresql(self, byte_sequence):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        with self.postgresql_connection.cursor() as cursor:
            cursor.execute(self.select_query, (byte_sequence,))
            query_result = cursor.fetchall()

        exploded_children = set()
        merged_subtree = {}
        subtrees_list = []
        if len(query_result) > 0:
            for children, subtree in query_result:
                splitted_children = set(children[i:i+2] for i in range(0, len(children), 2))
                exploded_children.update(splitted_children)
                if subtree:
                    subtrees_list.append(subtree)

        merged_subtree = self._load_merge_subtrees(subtrees_list)
        children_token_ids = set(map(self.tkde, exploded_children))

        return children_token_ids, merged_subtree

class ConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, index, switch_pattern, end_token, tokenizer=None):
        self.index = index
        self.switch_pattern = switch_pattern
        self.switch_pattern_reversed = list(reversed(self.switch_pattern))
        self.end_token = end_token

        self.tokenizer=tokenizer # for debugging

    def _find_sequence_index(self, lst, seq):
        """Find the first index of the sequence in the list."""
        for i in range(len(lst) - len(seq) + 1):
            if lst[i:i+len(seq)] == seq:
                return i
        raise ValueError(f"{seq} is not in list")

    def _find_if_constrained(self, sequence):
        """
        Given a sequence it analyze the sequence in reverse:
        if the switch_pattern is found first (starting from the end of the sequence) it means we are in CONSTRAINED GENERATION
        otherwise if the end_token is found first we are in NORMAL GENERATION
        """
        reversed_sequence = list(reversed(sequence))
        try:
            end_token_index = reversed_sequence.index(self.end_token)
        except ValueError:
            end_token_index = None

        try:
            switch_pattern_index = self._find_sequence_index(reversed_sequence, self.switch_pattern_reversed)
        except ValueError:
            switch_pattern_index = None

        if switch_pattern_index is None and end_token_index is None:
            constrained = False
        elif end_token_index is None or switch_pattern_index < end_token_index:
            # switch pattern is last generated
            constrained = True
        elif switch_pattern_index is None or end_token_index < switch_pattern_index:
            constrained =  False
        else:
            raise Exception("Should not happen. Note that this implementation assumes nor switch_pattern not end_token can be inside the triples.")

        constrain_generation_sequence = []
        if constrained:
            constrain_generation_sequence = sequence[len(sequence)-switch_pattern_index:]

        return constrained, constrain_generation_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for i in range(input_ids.shape[0]):
            input_sequence = input_ids[i].tolist()

            constrained, constrain_generation_sequence = self._find_if_constrained(input_sequence)

            if constrained:
                # constrained generation
                scores[[i],:] = self.constrained_generation(constrain_generation_sequence, scores[[i],:])
            # else:
            #     # normal generation
            #     # scores are not altered
            #     pass
            
            if self.tokenizer:
                print(i, constrained, [self.tokenizer.convert_ids_to_tokens(t) for t in scores[i].argsort(descending=True)[:10].tolist()])

        return scores

    def constrained_generation(self, input_sequence, scores: torch.FloatTensor):

        possible_tokens = self.index.next_tokens(input_sequence)

        if len(possible_tokens) == 0:
            # end of constrained generation
            # send end of string
            possible_tokens = [self.end_token]

        possible_tokens = list(possible_tokens)
        possible_scores = scores[:, possible_tokens]

        scores[:,:] = -float('inf')
        scores[:, possible_tokens] = possible_scores

        return scores

class GetAnswer(StoppingCriteria):
    # todo: do not in reverse. consider multiple samples in the batch
    # terminate when all batches generate end token?
    # strategy=all strategy=any. strategy can be all or any python functions
    def __init__(self, answer, eofanswer, strategy=all):
        self.prompt = None
        self.answer = answer
        self.eofanswer = set(eofanswer)
        self.strategy = strategy

    def __call__(self, input_ids, scores, **kwargs):
        outcome = self.strategy(
            self.get_answer(input_ids[i], return_answer=False) for i in range(input_ids.shape[0]))
        return outcome

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_answer(self, input_ids, return_answer=True):
        sequence = input_ids[len(self.prompt):].tolist() # remove prompt
        eofanswer_count = 0
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