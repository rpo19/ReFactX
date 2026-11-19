"""Regression tests for the efficiency-focused cache and generation changes."""

import unittest

from refactx.SimpleCache import SimpleCache
from refactx.index import DictIndex, PostgresIngestIndex, PostgresTrieIndex


class TestSimpleCache(unittest.TestCase):
    def test_tuple_keys_and_direct_storage(self):
        cache = SimpleCache(maxsize=4)
        cache.add([1, 2, 3], {7: 2}, {"subtree": [1]})
        next_tokens, subtree = cache.next_tokens([1, 2, 3])
        self.assertEqual(next_tokens, {7: 2})
        self.assertEqual(subtree, {"subtree": [1]})
        self.assertIn((1, 2, 3), cache.cache)
        self.assertNotIn("1.2.3", cache.cache)

    def test_values_are_isolated_from_caller_mutation(self):
        cache = SimpleCache(maxsize=4)
        source_tokens = {7: 2}
        source_subtree = {"branch": [1, 2]}
        cache.add([1], source_tokens, source_subtree)

        next_tokens, subtree = cache.next_tokens([1])
        next_tokens[7] = 999
        subtree["branch"].append(3)
        source_tokens[7] = 111

        next_tokens, subtree = cache.next_tokens([1])
        self.assertEqual(next_tokens, {7: 2})
        self.assertEqual(subtree, {"branch": [1, 2]})

    def test_stats_and_flush(self):
        cache = SimpleCache(maxsize=2)
        self.assertEqual(cache.next_tokens([1]), (None, None))
        cache.add([1], {2: 1}, None)
        self.assertEqual(cache.next_tokens([1])[0], {2: 1})
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)

        cache.flush()
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.next_tokens([1]), (None, None))

    def test_copy_on_get_can_be_disabled(self):
        cache = SimpleCache(maxsize=2, copy_on_get=False)
        subtree = {"branch": [1]}
        cache.add([1], {2: 1}, subtree)
        _, first = cache.next_tokens([1])
        first["branch"].append(2)
        _, second = cache.next_tokens([1])
        self.assertIs(first, second)
        self.assertEqual(second["branch"], [1, 2])


class _SubtreeState:
    def __init__(self):
        self.subtree_cache = DictIndex()
        self.reset_calls = 0

    def end_of_triple_reset(self):
        self.subtree_cache.reset()
        self.reset_calls += 1


class _FakePostgresIndex(PostgresTrieIndex):
    """Postgres trie whose rows come from an in-memory ingest index."""

    def __init__(self, rows, switch_parameter=4, rootkey=-100):
        super().__init__(
            postgresql_connection='',
            table_name='fake',
            switch_parameter=switch_parameter,
            rootkey=rootkey,
            cache=SimpleCache(maxsize=16),
        )
        self._rows = dict(rows)
        self.queried = []

    def _query_postgresql(self, sequence):
        key = tuple(sequence)
        self.queried.append(key)
        return self._rows.get(key, [])


def _ingest_rows(sequences, switch_parameter=4, rootkey=-100):
    ingest = PostgresIngestIndex(
        tokenizer_name='fake-tokenizer',
        rootkey=rootkey,
        switch_parameter=switch_parameter,
        table_name='fake',
        configkey=-200,
    )
    for sequence in sequences:
        ingest.add(sequence)
    rows = {}
    for key, children, numleaves, childrenleaves, subtree in ingest.get_rows():
        rows[tuple(key)] = [(0, children, subtree, numleaves, childrenleaves)]
    return rows, rootkey


class TestPostgresCanonicalSubtreeCache(unittest.TestCase):
    def _make_index(self, switch_parameter=4):
        # Two facts share the prefix [1, 2], then diverge.
        rows, rootkey = _ingest_rows(
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 9, 10],
                [1, 2, 7, 8, 11],
            ],
            switch_parameter=switch_parameter,
            rootkey=-100,
        )
        return _FakePostgresIndex(rows, switch_parameter=switch_parameter,
                                  rootkey=rootkey)

    def test_cache_hit_returns_canonical_copy(self):
        index = self._make_index()
        prefix = [1, 2, 3]  # with rootkey this is exactly switch_parameter=4
        state = _SubtreeState()

        next_tokens, _ = index.next_tokens(prefix, state=state)
        self.assertTrue(next_tokens)
        first_tree = state.subtree_cache.tree
        self.assertTrue(first_tree[1])
        self.assertEqual(index.cache_misses, 1)
        self.assertEqual(index.cache_hits, 0)

        # Mutating the state must not corrupt the cached snapshot.
        state.subtree_cache.add([-100, 1, 2, 3, 99, 100])

        other_state = _SubtreeState()
        next_tokens_hit, _ = index.next_tokens(prefix, state=other_state)
        self.assertEqual(next_tokens_hit, next_tokens)
        self.assertEqual(index.cache_hits, 1)
        self.assertNotIn(
            99, _children_after(other_state.subtree_cache.tree, [-100, 1, 2, 3]))

    def test_sibling_prefix_does_not_accumulate_into_cached_tree(self):
        index = self._make_index()
        state = _SubtreeState()

        index.next_tokens([1, 2, 3], state=state)
        tree_a = state.subtree_cache.tree
        index.next_tokens([1, 2, 7], state=state)
        tree_b = state.subtree_cache.tree

        # The second lookup replaces the state's subtree with the canonical
        # tree for [1, 2, 7] instead of merging both branches into one huge
        # snapshot.
        self.assertIsNot(tree_a, tree_b)
        self.assertEqual(_children_after(tree_a, [-100, 1, 2]), {3})
        self.assertEqual(_children_after(tree_b, [-100, 1, 2]), {7})
        self.assertEqual(_children_after(tree_b, [-100, 1, 2, 7]), {8})

    def test_longer_prefix_is_served_from_subtree_cache(self):
        index = self._make_index()
        state = _SubtreeState()
        index.next_tokens([1, 2, 3], state=state)

        # [1,2,3,4] is longer than switch_parameter, so it must be resolved
        # from the warmed subtree rather than PostgreSQL.
        queries_before = index.queries
        next_tokens, extra = index.next_tokens([1, 2, 3, 4], state=state)
        self.assertEqual(index.queries, queries_before)
        self.assertTrue(extra["found_subtree"])
        self.assertEqual(set(next_tokens), {5})


def _children_after(tree, prefix):
    """Return the tokens reachable immediately after ``prefix`` in a trie."""
    if not tree or not tree[1]:
        return set()
    level = tree
    cursor = 0
    level_cursor = 0
    while cursor < len(prefix) and level_cursor < len(level[1]):
        if isinstance(level[1][level_cursor], dict):
            if prefix[cursor] not in level[1][level_cursor]:
                return set()
            level = level[1][level_cursor][prefix[cursor]]
            level_cursor = 0
        else:
            if prefix[cursor] != level[1][level_cursor]:
                return set()
            level_cursor += 1
        cursor += 1
    if cursor != len(prefix) or level_cursor >= len(level[1]):
        return set()
    if isinstance(level[1][level_cursor], dict):
        return set(level[1][level_cursor].keys())
    return {level[1][level_cursor]}


if __name__ == "__main__":
    unittest.main()


class TestConstrainedProcessorInPlaceMasking(unittest.TestCase):
    def setUp(self):
        from transformers import AutoTokenizer
        self.torch = __import__('torch')
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def _make_processor(self, constrained_rows):
        from refactx.generate import (
            ConstrainedLogitsProcessor,
            ConstrainedStateList,
            PatternConstrainedState,
        )

        class FakeGeneration:
            done = False

            def constrain(self, sequence, mask, mask_idx):
                mask[mask_idx, :] = float('-inf')
                mask[mask_idx, [1, 3]] = 0

        states = []
        for is_constrained in constrained_rows:
            state = PatternConstrainedState(
                tokenizer=self.tokenizer,
                cache_index=DictIndex(),
                subtree_cache=DictIndex(),
            )
            if is_constrained:
                state.active_generation = FakeGeneration()
            states.append(state)

        state_list = ConstrainedStateList(
            [[state] for state in states], num_beams=1, num_batches=len(states),
        )
        return ConstrainedLogitsProcessor(state_list, self.tokenizer)

    def test_masks_in_place_and_preserves_allowed_scores(self):
        processor = self._make_processor([True])
        scores = self.torch.randn(1, len(self.tokenizer))
        original = scores.clone()
        result = processor(self.torch.tensor([[5]]), scores)

        self.assertIs(result, scores)
        self.torch.testing.assert_close(result[0, [1, 3]], original[0, [1, 3]])
        blocked = [i for i in range(result.shape[1]) if i not in (1, 3)]
        self.assertTrue(self.torch.isinf(result[0, blocked]).all())
        self.assertTrue((result[0, blocked] < 0).all())

    def test_unconstrained_rows_are_not_modified(self):
        processor = self._make_processor([True, False])
        scores = self.torch.randn(2, len(self.tokenizer))
        original = scores.clone()
        processor(self.torch.tensor([[5], [7]]), scores)

        self.torch.testing.assert_close(scores[1], original[1])
        self.assertTrue(self.torch.isfinite(scores[0, 1]))
        self.assertTrue(self.torch.isinf(scores[0, 0]))


class TestParallelBatchProcessor(unittest.TestCase):
    def setUp(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        from refactx.index import DictIndex
        self.index = DictIndex()
        self.index.set_tokenizer(self.tokenizer)
        self.triple = "<Paris> <capital of> <France> ."
        self.index.add(self.tokenizer(self.triple, add_special_tokens=False)['input_ids'])
        self.feed = self.tokenizer(
            self.triple, add_special_tokens=False)['input_ids']

    def _processor(self, workers):
        from refactx.generate import (
            ConstrainedLogitsProcessor,
            ConstrainedStateList,
            activate_constrained_fact,
            build_pattern_state,
        )
        rows = []
        for _ in range(2):
            state = build_pattern_state(
                self.tokenizer, self.index, sentinel=False, eot=' </fact>\n')
            activate_constrained_fact(
                state, self.index, sentinel=False, eot=' </fact>\n')
            rows.append([state])
        states = ConstrainedStateList(rows, num_beams=1, num_batches=2)
        return ConstrainedLogitsProcessor(
            states, self.tokenizer, parallel_workers=workers)

    def test_parallel_matches_sequential_masks(self):
        sequential = self._processor(0)
        parallel = self._processor(2)
        seed = self.tokenizer('x', add_special_tokens=False)['input_ids'][:1]
        sequences = [list(seed), list(seed)]
        generator = self.torch.Generator().manual_seed(0)
        sequential(sequential_input := self.torch.tensor(sequences),
                   self.torch.zeros(2, len(self.tokenizer)))
        parallel(self.torch.tensor(sequences),
                 self.torch.zeros(2, len(self.tokenizer)))
        for token in self.feed:
            for row in range(2):
                sequences[row].append(token)
            scores = self.torch.randn(2, len(self.tokenizer), generator=generator)
            seq_scores = scores.clone()
            par_scores = scores.clone()
            sequential(self.torch.tensor(sequences), seq_scores)
            parallel(self.torch.tensor(sequences), par_scores)
            self.torch.testing.assert_close(seq_scores, par_scores)

    def test_non_thread_safe_index_disables_parallelism(self):
        from refactx.generate import get_constrained_logits_processor

        class NotThreadSafe:
            switch_parameter = 7
            rootkey = -100
            is_thread_safe = False

            def next_tokens(self, sequence, **kwargs):
                return {}, {}

        processor = get_constrained_logits_processor(
            self.tokenizer, NotThreadSafe(), num_batches=1,
            parallel_workers=4, return_list=False,
        )
        self.assertEqual(processor.parallel_workers, 0)

    @property
    def torch(self):
        return __import__('torch')
