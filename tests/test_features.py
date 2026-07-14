"""
Tests for:
  Architecture refactor: PatternConstrainedGeneration, FactGeneration, CountBranchesGeneration
  Feat 1: CountBranches tool call (standalone + generation class)
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
    PatternConfig,
    FactGeneration,
    CountBranchesGeneration,
)


def _feed_sequence(proc, tokenizer, token_ids):
    """Feed token_ids one-by-one, building up the full sequence like HF generate."""
    full_seq = []
    for tid in token_ids:
        full_seq.append(tid)
        input_ids = torch.tensor([full_seq])
        scores = torch.zeros(1, len(tokenizer))
        proc(input_ids, scores)


def _negative_inf_mask(batch_size, vocab_size):
    """Mask that blocks all tokens (used as the starting mask in production)."""
    return torch.full((batch_size, vocab_size), -math.inf)


# ---------------------------------------------------------------------------
# Architecture: PatternConfig
# ---------------------------------------------------------------------------

class TestPatternConfig(unittest.TestCase):
    def test_creation(self):
        cfg = PatternConfig(
            pattern="Fact:",
            generation_class=FactGeneration,
            config={"index": "dummy"},
        )
        self.assertEqual(cfg.pattern, "Fact:")
        self.assertIs(cfg.generation_class, FactGeneration)
        self.assertEqual(cfg.config, {"index": "dummy"})

    def test_default_config(self):
        cfg = PatternConfig(pattern="test", generation_class=int)
        self.assertEqual(cfg.config, {})


# ---------------------------------------------------------------------------
# Architecture: PatternConstrainedState
# ---------------------------------------------------------------------------

class TestPatternConstrainedState(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()

    def _make_state(self, **kwargs):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
            **kwargs,
        )

    def test_initial_state(self):
        state = self._make_state()
        self.assertEqual(state.state, 0)
        self.assertEqual(state.patterns, [])
        self.assertIsNone(state.active_generation)
        self.assertEqual(state.generation_history, [])

    def test_is_constrained_false_initially(self):
        state = self._make_state()
        self.assertFalse(state.is_constrained())

    def test_thinking_pattern_default(self):
        state = self._make_state()
        self.assertIn("think", state.thinking_end_pattern)

    def test_custom_thinking_pattern(self):
        state = self._make_state(thinking_end_pattern='</reasoning>')
        self.assertEqual(state.thinking_end_pattern, '</reasoning>')

    def test_cache_reset_on_thinking_end(self):
        state = self._make_state()
        triple_ids = self.tokenizer.encode("<Paris> <capital of> <France> .", add_special_tokens=False)
        state.cache_index.add(triple_ids)

        prefix_ids = triple_ids[:3]
        next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
        self.assertTrue(len(next_tokens) > 0)

        end_pattern = state.thinking_end_pattern
        think_tokens = self.tokenizer.encode(end_pattern, add_special_tokens=False)
        for tid in think_tokens:
            state.update(tid)

        try:
            next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
            self.assertEqual(len(next_tokens), 0)
        except Exception:
            pass

    def test_cache_not_reset_twice(self):
        state = self._make_state()
        triple_ids = self.tokenizer.encode("<Paris> <capital of> <France> .", add_special_tokens=False)
        state.cache_index.add(triple_ids)

        end_pattern = state.thinking_end_pattern
        think_tokens = self.tokenizer.encode(end_pattern, add_special_tokens=False)
        for tid in think_tokens:
            state.update(tid)
        self.assertTrue(state._thinking_cache_reset_done)

        state.cache_index.add(triple_ids)
        for tid in think_tokens:
            state.update(tid)
        prefix_ids = triple_ids[:3]
        next_tokens, _ = state.cache_index.next_tokens(prefix_ids)
        self.assertTrue(len(next_tokens) > 0)

    def test_reset_clears_flag(self):
        state = self._make_state()
        self.assertFalse(state._thinking_cache_reset_done)
        state.reset()
        self.assertFalse(state._thinking_cache_reset_done)

    def test_reset_clears_patterns_and_generation(self):
        state = self._make_state()
        cfg = PatternConfig(pattern="Fact:", generation_class=FactGeneration,
                            config={"index": self.kb, "sentinel": False, "eot": None})
        state.patterns.append(cfg)
        state.reset()
        self.assertEqual(state.patterns, [])
        self.assertIsNone(state.active_generation)


# ---------------------------------------------------------------------------
# Architecture: FactGeneration
# ---------------------------------------------------------------------------

class TestFactGeneration(unittest.TestCase):

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

    def _make_state(self):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )

    def test_constrain_masks_all_except_possible(self):
        state = self._make_state()
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=False, eot=None,
        )
        vocab_size = len(self.tokenizer)
        mask = _negative_inf_mask(1, vocab_size)
        gen.constrain([], mask, 0)
        allowed = (mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        self.assertGreater(len(allowed), 0)
        self.assertLess(len(allowed), vocab_size)

    def test_done_set_when_exhausted(self):
        state = self._make_state()
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=False, eot=None,
        )
        all_ids = self.tokenizer.encode(
            "<Paris> <capital of> <France> .", add_special_tokens=False)
        for i, tid in enumerate(all_ids):
            sequence = all_ids[:i + 1]
            gen.constrain(sequence, _negative_inf_mask(1, len(self.tokenizer)), 0)
        self.assertTrue(gen.done)


# ---------------------------------------------------------------------------
# Architecture: CountBranchesGeneration
# ---------------------------------------------------------------------------

class TestCountBranchesGeneration(unittest.TestCase):
    """Tests use BERT tokenizer with '.' as delimiter (BERT strips whitespace)."""

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
        self.delimiter_id = self.tokenizer.encode(".", add_special_tokens=False)[0]

    def _make_gen(self, **extra):
        state = PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )
        return CountBranchesGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            kb_index=self.kb, delimiter=".", **extra,
        )

    def test_starts_in_collecting_mode(self):
        gen = self._make_gen()
        self.assertEqual(gen.mode, CountBranchesGeneration.MODE_COLLECTING)
        self.assertEqual(gen.prefix_tokens, [])

    def test_collecting_appends_tokens(self):
        gen = self._make_gen()
        mask = _negative_inf_mask(1, len(self.tokenizer))
        prefix_ids = self.tokenizer.encode("<Paris> <capital of>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], mask, 0)
        self.assertEqual(len(gen.prefix_tokens), len(prefix_ids))

    def test_delimiter_triggers_count(self):
        gen = self._make_gen()
        mask = _negative_inf_mask(1, len(self.tokenizer))
        prefix_ids = self.tokenizer.encode("<Paris> <capital of>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], mask, 0)
        gen.constrain([self.delimiter_id], mask, 0)
        self.assertEqual(gen.mode, CountBranchesGeneration.MODE_EMITTING)
        self.assertEqual(len(gen.calls), 1)
        self.assertEqual(gen.calls[0][1], 1)

    def test_calls_recorded(self):
        gen = self._make_gen()
        mask = _negative_inf_mask(1, len(self.tokenizer))
        prefix_ids = self.tokenizer.encode("<Paris> <country>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], mask, 0)
        gen.constrain([self.delimiter_id], mask, 0)
        self.assertEqual(gen.calls[0][1], 1)  # 1 leaf: <Paris> <country> <France> .

    def test_emitting_forces_tokens(self):
        gen = self._make_gen()
        mask = _negative_inf_mask(1, len(self.tokenizer))
        prefix_ids = self.tokenizer.encode("<Paris> <country>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], mask, 0)
        gen.constrain([self.delimiter_id], mask, 0)
        allowed = (mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
        self.assertEqual(len(allowed), 1)

    def test_done_after_emitting_all(self):
        gen = self._make_gen()
        vocab_size = len(self.tokenizer)
        prefix_ids = self.tokenizer.encode("<Paris> <country>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], _negative_inf_mask(1, vocab_size), 0)
        gen.constrain([self.delimiter_id], _negative_inf_mask(1, vocab_size), 0)
        while not gen.done:
            gen.constrain([], _negative_inf_mask(1, vocab_size), 0)
        self.assertTrue(gen.done)

    def test_no_delimiter_stays_collecting(self):
        gen = self._make_gen()
        mask = _negative_inf_mask(1, len(self.tokenizer))
        prefix_ids = self.tokenizer.encode("<Paris> <capital of>", add_special_tokens=False)
        for tid in prefix_ids:
            gen.constrain([tid], mask, 0)
        self.assertEqual(gen.mode, CountBranchesGeneration.MODE_COLLECTING)
        self.assertEqual(len(gen.calls), 0)


# ---------------------------------------------------------------------------
# Feat 1 - CountBranches standalone (deprecated, kept for backward compat)
# ---------------------------------------------------------------------------

class TestCountBranchesStandalone(unittest.TestCase):

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

    def test_calls_persisted(self):
        proc = self._make_proc()
        self.assertEqual(proc.calls, [])
        trigger_ids = self.tokenizer.encode("count_branches(", add_special_tokens=False)
        prefix_ids = self.tokenizer.encode("<Paris> <capital of>", add_special_tokens=False)
        close_paren = self.tokenizer.encode(")", add_special_tokens=False)[0]
        _feed_sequence(proc, self.tokenizer, list(trigger_ids) + list(prefix_ids) + [close_paren])
        self.assertEqual(len(proc.calls), 1)
        prefix_text, count = proc.calls[0]
        self.assertEqual(count, 1)
        while proc.mode == CountBranchesLogitsProcessor.MODE_EMITTING:
            scores = torch.zeros(1, len(self.tokenizer))
            full_seq = proc.input_ids + [0]
            input_ids = torch.tensor([full_seq])
            proc(input_ids, scores)
        self.assertEqual(len(proc.calls), 1)


# ---------------------------------------------------------------------------
# Feat 2 - Sentinel (via FactGeneration)
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
        state = self._make_state()
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=True, sentinel_text='no further records>',
            eot=None,
        )
        ids = gen._sentinel_ids()
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_sentinel_text_decoded(self):
        state = self._make_state()
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=True, sentinel_text='no further records>',
            eot=None,
        )
        ids = gen._sentinel_ids()
        decoded = self.tokenizer.decode(ids)
        self.assertIn("no further records", decoded)


if __name__ == "__main__":
    unittest.main()
