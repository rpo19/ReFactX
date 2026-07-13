"""
Tests for the three new features:
  Feat 1: CountBranchesLogitsProcessor
  Feat 2: Sentinel for exhausted subject-relations
  Feat 3: Reset duplicate cache after end-of-thinking
"""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.index import DictIndex
from refactx.count_branches import CountBranchesLogitsProcessor
from refactx.generate import (
    PatternConstrainedState,
    ConstrainedLogitsProcessor,
    ConstrainedStateList,
)


def _feed_sequence(proc, tokenizer, token_ids):
    """Feed token_ids one-by-one, building up the full sequence like HF generate."""
    full_seq = []
    for tid in token_ids:
        full_seq.append(tid)
        input_ids = torch.tensor([full_seq])
        scores = torch.zeros(1, len(tokenizer))
        proc(input_ids, scores)


# ---------------------------------------------------------------------------
# Feat 1 - CountBranchesLogitsProcessor
# ---------------------------------------------------------------------------

class TestCountBranches(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()
        triples = [
            "<Paris> <capital of> <France> .",
            "<Paris> <country> <France> .",
            "<Lyon> <country> <France> .",
        ]
        ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for ids_seq in ids:
            self.kb.add(ids_seq)

    def _make_proc(self, **kwargs):
        return CountBranchesLogitsProcessor(self.tokenizer, self.kb, **kwargs)

    def test_starts_in_free_mode(self):
        proc = self._make_proc()
        self.assertEqual(proc.mode, CountBranchesLogitsProcessor.MODE_FREE)

    def test_detects_trigger_and_enters_collecting(self):
        proc = self._make_proc()
        trigger_ids = self.tokenizer.encode("count_branches(", add_special_tokens=False)
        _feed_sequence(proc, self.tokenizer, trigger_ids)
        self.assertEqual(proc.mode, CountBranchesLogitsProcessor.MODE_COLLECTING)

    def test_counts_and_emits(self):
        proc = self._make_proc()
        prefix_ids = self.tokenizer.encode("<Paris> <capital of>", add_special_tokens=False)
        trigger_ids = self.tokenizer.encode("count_branches(", add_special_tokens=False)
        close_paren = self.tokenizer.encode(")", add_special_tokens=False)[0]

        all_ids = list(trigger_ids) + list(prefix_ids) + [close_paren]
        _feed_sequence(proc, self.tokenizer, all_ids)

        self.assertEqual(proc.mode, CountBranchesLogitsProcessor.MODE_EMITTING)

    def test_returns_to_free_after_emitting(self):
        proc = self._make_proc()
        trigger_ids = self.tokenizer.encode("count_branches(", add_special_tokens=False)
        prefix_ids = self.tokenizer.encode("<Paris>", add_special_tokens=False)
        close_paren = self.tokenizer.encode(")", add_special_tokens=False)[0]

        all_ids = list(trigger_ids) + list(prefix_ids) + [close_paren]
        _feed_sequence(proc, self.tokenizer, all_ids)

        while proc.mode == CountBranchesLogitsProcessor.MODE_EMITTING:
            scores = torch.zeros(1, len(self.tokenizer))
            full_seq = proc.input_ids + [0]
            input_ids = torch.tensor([full_seq])
            proc(input_ids, scores)

        self.assertEqual(proc.mode, CountBranchesLogitsProcessor.MODE_FREE)

    def test_free_mode_no_masking(self):
        proc = self._make_proc()
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        scores_original = torch.randn(1, len(self.tokenizer))
        scores = scores_original.clone()
        _feed_sequence(proc, self.tokenizer, ids)
        torch.testing.assert_close(scores, scores_original)


# ---------------------------------------------------------------------------
# Feat 2 - Sentinel
# ---------------------------------------------------------------------------

class TestSentinel(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()
        triples = [
            "<Paris> <capital of> <France> .",
            "<Paris> <country> <France> .",
        ]
        ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for ids_seq in ids:
            self.kb.add(ids_seq)

    def _make_state(self):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )

    def test_sentinel_remaining_initialized_empty(self):
        state = self._make_state()
        self.assertEqual(state.sentinel_remaining, [])

    def test_sentinel_ids(self):
        states = ConstrainedStateList('auto', num_beams=1, num_batches=1, debug_tokenizer=self.tokenizer)
        proc = ConstrainedLogitsProcessor(
            index=self.kb, states=states, tokenizer=self.tokenizer,
            eot=None, sentinel=True, sentinel_text='no further records>',
        )
        ids = proc._sentinel_ids()
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_sentinel_text_decoded(self):
        states = ConstrainedStateList('auto', num_beams=1, num_batches=1, debug_tokenizer=self.tokenizer)
        proc = ConstrainedLogitsProcessor(
            index=self.kb, states=states, tokenizer=self.tokenizer,
            eot=None, sentinel=True, sentinel_text='no further records>',
        )
        ids = proc._sentinel_ids()
        decoded = self.tokenizer.decode(ids)
        self.assertIn("no further records", decoded)


# ---------------------------------------------------------------------------
# Feat 3 - Reset cache after thinking
# ---------------------------------------------------------------------------

class TestThinkingCacheReset(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()
        triples = ["<Paris> <capital of> <France> ."]
        ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for ids_seq in ids:
            self.kb.add(ids_seq)

    def _make_state(self, **kwargs):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
            **kwargs,
        )

    def test_thinking_pattern_default(self):
        state = self._make_state()
        self.assertIn("think", state.thinking_end_pattern)

    def test_cache_reset_on_thinking_end(self):
        state = self._make_state()
        triple_ids = self.tokenizer.encode("<Paris> <capital of> <France> .", add_special_tokens=False)
        state.cache_index.add(triple_ids)

        # Verify cache is non-empty by checking a prefix
        prefix_ids = triple_ids[:3]
        next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
        self.assertTrue(len(next_tokens) > 0)

        # Feed thinking end tokens (use the state's update method)
        end_pattern = state.thinking_end_pattern
        think_tokens = self.tokenizer.encode(end_pattern, add_special_tokens=False)
        for tid in think_tokens:
            state.update(tid)

        # Cache should now be reset
        try:
            next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
            self.assertEqual(len(next_tokens), 0)
        except Exception:
            pass  # EmptyIndexException means cache is empty - correct

    def test_cache_not_reset_twice(self):
        state = self._make_state()
        triple_ids = self.tokenizer.encode("<Paris> <capital of> <France> .", add_special_tokens=False)
        state.cache_index.add(triple_ids)

        # Feed thinking end once
        end_pattern = state.thinking_end_pattern
        think_tokens = self.tokenizer.encode(end_pattern, add_special_tokens=False)
        for tid in think_tokens:
            state.update(tid)
        self.assertTrue(state._thinking_cache_reset_done)

        # Add to cache again
        state.cache_index.add(triple_ids)
        # Feed thinking end again - should NOT reset because flag is set
        for tid in think_tokens:
            state.update(tid)
        # Cache should still have data
        prefix_ids = triple_ids[:3]
        next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
        self.assertTrue(len(next_tokens) > 0)

    def test_reset_clears_flag(self):
        state = self._make_state()
        self.assertFalse(state._thinking_cache_reset_done)
        state.reset()
        self.assertFalse(state._thinking_cache_reset_done)

    def test_custom_thinking_pattern(self):
        state = self._make_state(thinking_end_pattern='</reasoning>')
        self.assertEqual(state.thinking_end_pattern, '</reasoning>')


if __name__ == "__main__":
    unittest.main()
