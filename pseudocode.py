"""
PSEUDOCODE for ReFactX

This code describes the most important functions behind ReFactX.
Some auxiliary functions are only described with comments and not with pseudocode.

Beam search or batched generation are ignored for simplicity.
"""

import numpy as np
import math

################################################################################
# Auxiliary functions
################################################################################

class InMemoryPrefixTree():
    def __init__(self, tree, numleaves):
        # saves tree and numleaves
        self.tree = tree
        self.numleaves = numleaves
    """
    Given a sequence, returns the list of allowed next tokens from the sequence
    and the number of leaves reachable from each of them.
    """
    def next_tokens(self, fact_sequence: List) -> List, List:
        # ...
        # visits the tree
        return next_tokens, numleaves

class ReachedLeavesTree():
    def __init__(self):
        self.tree = {} # initialize empty
    def add(self, fact_sequence: List) -> None:
        # ...
        # saves the fact_sequence in the tree
        pass
    """
    Given a sequence, returns the list of already visited tokens from that sequence
    with the number of times they have been already visited
    """
    def visited(self, fact_sequence: List) -> List, List:
        # ...
        # visits the tree
        return visited_tokens, visited_times

"""
Runs the query (containing a prefix sequence) on the prefix-tree db returning:
- the tokens reachable from the sequence
- the number of leaves reachable from each token
- the subtrees (List of subtree) reachable from each token. If available they are loaded in memory for the next generation.
"""
def run_sql_query(query) -> List, List, List:
    # ...
    # run sql query on the db
    return next_tokens, numleaves, subtrees

"""
Since the db is ingested in batched it can contain duplicated prefixes.
So the subtrees must be merged in a single in-memory tree.
"""
def merge_subtrees(subtrees):
    # ...
    return merged_subtree

################################################################################

IN_MEMORY_SUBTREE = InMemoryPrefixTree() # initialize empty in-memory prefix tree
REACHED_LEAVES = ReachedLeavesTree() # tree saving already reached leaves to avoid generating duplicate facts

def sampling_function(logits):
    # simple greedy sampling takes the token with highest score
    return np.argmax(logits)

def refactx_next_tokens(fact_sequence, Lc_prefix_length):
    global IN_MEMORY_SUBTREE
    if len(fact_sequence) < Lc_prefix_length:
        next_tokens, numleaves, subtrees = run_sql_query(f'SELECT nexttokens, numleaves, subtrees FROM prefixtree WHERE prefix = {fact_sequence};')
        if subtrees is not None:
            merged_subtree = merge_subtrees(subtrees)
            IN_MEMORY_SUBTREE = merged_subtree
    else:
        next_tokens, numleaves = IN_MEMORY_SUBTREE.next_tokens(fact_sequence)

    return next_tokens, numleaves

def generate(LLM, input_sequence: List, max_new_tokens: int, eos_token: int, fact_token: int):
    state = 'normal'
    total_max_tokens = len(input_sequence) + max_new_tokens
    sequence = input_sequence
    next_token = sequence[-1]
    fact_sequence = [] # initialize empty sequence
    while next_token != eos_token and len(sequence) < total_max_tokens:
        logits = LLM(sequence) # get a score for all the tokens in the LLM vocabulary
        if state == 'constrained':
            allowed_next_tokens, numleaves = refactx_next_tokens(fact_sequence)
            if len(allowed_next_tokens) > 0: # fact is not complete yet
                # avoid generating duplicate facts
                visited_tokens, visited_times = REACHED_LEAVES.visited(fact_sequence) # gets the tokens already reached from fact_sequence and how many times
                numleaves[visited_tokens] -= visited_times
                # remove the tokens for which the number of reachable leaves (reduced by the already reached) is zero
                for idx, count in enumerate(numleaves):
                    if count == 0:
                        del allowed_next_tokens[idx]

                logits[~allowed_next_tokens] = -math.inf # set the logits of forbidden tokens to -inf
            else: # reached the end of a fact
                state = 'normal'
                REACHED_LEAVES.add(fact_sequence) # save reached leaf
                IN_MEMORY_SUBTREE = InMemoryPrefixTree() # reset in memory subtree

        next_token = sampling_function(logits)
        sequence.append(next_token)
        if state == 'constrained':
            fact_sequence.append(next_token)
        elif state == 'normal':
            if next_token == fact_token:
                state = 'constrained'
    return sequence