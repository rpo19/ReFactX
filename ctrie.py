from transformers.generation.logits_process import LogitsProcessor
import torch
import pickle
from mergedeep import merge

def tken(token):
    # token encode
    encoded = token.to_bytes(2, byteorder='big', signed=False)
    return encoded

def tkde(bbytes):
    # token decode
    decoded = int.from_bytes(bbytes, byteorder='big', signed=False)
    return decoded

class ModDisjunctiveTrie:
    # def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
    #     r"""
    #     A helper class that builds a trie with the words represented in `nested_token_ids`.
    #     """
    #     self.max_height = max([len(one) for one in nested_token_ids])

    #     root = {}
    #     for token_ids in nested_token_ids:
    #         level = root
    #         for tidx, token_id in enumerate(token_ids):
    #             if token_id not in level:
    #                 level[token_id] = {}

    #             level = level[token_id]

    #     if no_subsets and self.has_subsets(root, nested_token_ids):
    #         raise ValueError(
    #             "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
    #             f" {nested_token_ids}."
    #         )

    #     self.trie = root

    def __init__(self, postgresql_connection, rootkey: int):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """

        self.reset_cache()
        self.postgresql_connection = postgresql_connection
        self.rootkey = rootkey

    def reset_cache(self):
        self.cache_sequence_prefix = []
        self.cached_tree = {}
        self.reset_subtree_cache()

    def seq_startswith(self, seq1, seq2):
        if len(seq2) == 0:
            return False
        subseq1 = seq1[:len(seq2)]
        return subseq1 == seq2

    def reset_subtree_cache(self):
        self.cached_bsubtrees = {}
        self.cached_bsubtrees_map = {}

    def next_tokens(self, current_seq):
        current_seq = [self.rootkey] + current_seq
        encoded_sequence = map(tken, current_seq)
        byte_sequence = b''.join(encoded_sequence)

        if byte_sequence in self.cached_bsubtrees_map:
            # if the chosen token has the subtree cached in memory
            # load the subtree and proceed in memory
            if len(self.cached_tree) == 0:
                # need to load and merge the matching subtrees
                subtrees_ids = self.cached_bsubtrees_map[byte_sequence]
                subtrees_list = (self.cached_bsubtrees[id] for id in subtrees_ids)
                self._load_merge_subtrees(subtrees_list)
                self.cache_sequence_prefix = current_seq[:-1] # todo debug
                self.reset_subtree_cache()

        if self.seq_startswith(current_seq, self.cache_sequence_prefix):
            _next_tokens = self._next_tokens_from_dict(current_seq)
        else:
            _next_tokens = self._next_tokens_from_postgresql(byte_sequence)

        return _next_tokens

    def _next_tokens_from_postgresql(self, byte_sequence):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        with self.postgresql_connection.cursor() as cursor:
            cursor.execute('SELECT children, subtree FROM ctrie WHERE key = %s;', (byte_sequence,))
            query_result = cursor.fetchall()

            exploded_children = set()
            if len(query_result) > 0:
                self.reset_cache() # reset to refill again
                for subtree_id, (children, subtree) in enumerate(query_result):
                    splitted_children = set(children[i:i+2] for i in range(0, len(children), 2))
                    if subtree is not None:
                        self.cached_bsubtrees[subtree_id] = subtree
                        for child in splitted_children:
                            full_child_sequence = byte_sequence + child
                            if full_child_sequence not in self.cached_bsubtrees_map:
                                self.cached_bsubtrees_map[full_child_sequence] = set()
                            self.cached_bsubtrees_map[full_child_sequence].add(subtree_id)
                    exploded_children.update(set(children[i:i+2] for i in range(0, len(children), 2)))

        children_token_ids = list(map(tkde, exploded_children))
        
        return children_token_ids

    def _load_merge_subtrees(self, subtrees_list):
        merged_subtree = {}
        for bsubtree in subtrees_list:
            if bsubtree is not None:
                subtree = pickle.loads(bsubtree)
                merge(merged_subtree, subtree)
        self.cached_tree = merged_subtree

    def _next_tokens_from_dict(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.cached_tree

        subtree_seq = current_seq[len(self.cache_sequence_prefix):]

        for current_token in subtree_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root = None):
        if root is None:
            root = self.trie
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        else:
            return sum([self.count_leaves(nn) for nn in next_nodes])

class CtrieLogitsProcessor(LogitsProcessor):
    def __init__(self, ctrie, eos_token = None, tokenizer = None):
        self.ctrie = ctrie
        self.sequence = []
        self.prompt = None
        self.eos_token = eos_token
        self.tokenizer = tokenizer # debug

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.prompt is None:
            self.prompt = input_ids[0].tolist()
        
        self.sequence = input_ids[0].tolist()[len(self.prompt):] # ignore prompt

        possible_tokens = self.ctrie.next_tokens(self.sequence)
        if not possible_tokens and self.eos_token is not None:
            # send end of string
            possible_tokens = [self.eos_token]

        possible_scores = scores[:, possible_tokens]

        if self.tokenizer is not None:
            highest_scores = self.tokenizer.convert_ids_to_tokens(scores[0,:].argsort(descending=True)[:10])
            scores[:,:] = -float('inf')
            scores[:, possible_tokens] = possible_scores
            constrained_highest_scores = self.tokenizer.convert_ids_to_tokens(scores[0,:].argsort(descending=True)[:10])
            print('normal:', highest_scores)
            print('consrtained:', constrained_highest_scores)

        else:
            scores[:,:] = -float('inf')
            scores[:, possible_tokens] = possible_scores
        return scores