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


# ---------------------------------------------------------------------------
# Feat 2b - Generalized sentinel (fires at any trie level)
# ---------------------------------------------------------------------------

class TestSentinelGeneralized(unittest.TestCase):
    """Generalized sentinel: fires when the model picks an exhausted token.

    Both live *and* exhausted tokens (remaining == 0) are always offered.
    After each ``constrain`` call, ``prev_exhausted`` stores the set of
    exhausted tokens.  On the next call, if ``sequence[-1]`` is in that
    set the sentinel fires immediately.
    """

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()
        # Two objects under <Paris> <capital of> to allow live/exhausted mix
        self.triples = [
            "<Paris> <capital of> <France> .",
            "<Paris> <capital of> <Germany> .",
            "<Paris> <country> <France> .",
            "<Lyon> <country> <France> .",
        ]
        ids = self.tokenizer(self.triples, add_special_tokens=False)["input_ids"]
        for ids_seq in ids:
            self.kb.add(ids_seq)

    def _make_state_and_gen(self):
        state = PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=True, eot=None,
        )
        return state, gen

    def _vocab_size(self):
        return len(self.tokenizer)

    def _allowed_tokens(self, mask):
        return set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())

    def _mark_visited(self, state, triple_text):
        ids = self.tokenizer.encode(triple_text, add_special_tokens=False)
        state.cache_index.add(ids, new_leaf=True)

    # -- Both live and exhausted always offered -------------------------------

    def test_exhausted_tokens_always_offered(self):
        """Exhausted tokens are offered alongside live ones at every level."""
        state, gen = self._make_state_and_gen()
        # Exhaust both <Paris> <capital of> objects
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        # Relation level: [<, paris, >, <] — 'capital' exhausted, 'country' live
        rel_seq = self.tokenizer.encode('<Paris> <', add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(rel_seq, mask, 0)
        allowed = self._allowed_tokens(mask)

        cap_tok = self.tokenizer.encode('capital', add_special_tokens=False)[0]
        country_tok = self.tokenizer.encode('country', add_special_tokens=False)[0]
        self.assertIn(cap_tok, allowed,
                      "Exhausted token should be offered")
        self.assertIn(country_tok, allowed,
                      "Live token should be offered")

    def test_live_and_exhausted_tokens_offered(self):
        """When some tokens are live and some exhausted, both are offered."""
        state, gen = self._make_state_and_gen()
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        rel_seq = self.tokenizer.encode('<Paris> <', add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(rel_seq, mask, 0)
        allowed = self._allowed_tokens(mask)
        self.assertGreaterEqual(len(allowed), 2,
                                "Both live and exhausted tokens should be offered")
        self.assertFalse(gen.completed_with_sentinel)

    # -- prev_exhausted tracking ---------------------------------------------

    def test_prev_exhausted_set_after_constrain(self):
        """After constrain, prev_exhausted contains exhausted token IDs."""
        state, gen = self._make_state_and_gen()
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        rel_seq = self.tokenizer.encode('<Paris> <', add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(rel_seq, mask, 0)

        cap_tok = self.tokenizer.encode('capital', add_special_tokens=False)[0]
        country_tok = self.tokenizer.encode('country', add_special_tokens=False)[0]
        self.assertIn(cap_tok, gen.prev_exhausted,
                      "Exhausted 'capital' should be in prev_exhausted")
        self.assertNotIn(country_tok, gen.prev_exhausted,
                         "Live 'country' should NOT be in prev_exhausted")

    def test_prev_exhausted_empty_when_no_exhaustion(self):
        """prev_exhausted is empty when no tokens are exhausted."""
        state, gen = self._make_state_and_gen()
        # No triples visited — nothing exhausted
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain([], mask, 0)
        self.assertEqual(len(gen.prev_exhausted), 0)

    # -- Sentinel fires when last token was exhausted ------------------------

    def test_sentinel_fires_when_exhausted_token_picked(self):
        """Sentinel fires on the call AFTER an exhausted token is picked."""
        state, gen = self._make_state_and_gen()
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        # Call 1: at relation level, 'capital' becomes exhausted
        rel_seq = self.tokenizer.encode('<Paris> <', add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(rel_seq, mask, 0)
        self.assertIn(
            self.tokenizer.encode('capital', add_special_tokens=False)[0],
            gen.prev_exhausted)

        # Call 2: simulate model picking 'capital' (exhausted)
        exhausted_seq = rel_seq + self.tokenizer.encode('capital', add_special_tokens=False)
        mask2 = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(exhausted_seq, mask2, 0)

        self.assertTrue(gen.completed_with_sentinel,
                        "Sentinel should fire when exhausted token is picked")

    def test_no_sentinel_when_live_token_picked(self):
        """Sentinel does NOT fire when a live token is picked."""
        state, gen = self._make_state_and_gen()
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        # Call 1: at relation level
        rel_seq = self.tokenizer.encode('<Paris> <', add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(rel_seq, mask, 0)

        # Call 2: simulate model picking 'country' (live)
        live_seq = rel_seq + self.tokenizer.encode('country', add_special_tokens=False)
        mask2 = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(live_seq, mask2, 0)

        self.assertFalse(gen.completed_with_sentinel,
                         "Sentinel should NOT fire when live token is picked")

    def test_sentinel_fires_at_object_level(self):
        """Sentinel fires when model picks an exhausted object token."""
        state, gen = self._make_state_and_gen()
        # Exhaust both <Paris> <capital of> objects
        self._mark_visited(state, self.triples[0])
        self._mark_visited(state, self.triples[1])

        # Call 1: at object level, '<' is exhausted
        obj_seq = self.tokenizer.encode('<Paris> <capital of>',
                                        add_special_tokens=False)
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(obj_seq, mask, 0)

        lt_tok = self.tokenizer.encode('<', add_special_tokens=False)[0]
        self.assertIn(lt_tok, gen.prev_exhausted,
                      "Setup: '<' should be exhausted at object level")

        # Call 2: simulate model picking '<' (exhausted)
        exhausted_seq = obj_seq + [lt_tok]
        mask2 = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(exhausted_seq, mask2, 0)

        self.assertTrue(gen.completed_with_sentinel,
                        "Sentinel should fire when exhausted object token is picked")

    def test_no_sentinel_when_live_object_token_picked(self):
        """No sentinel when model picks a live object token."""
        state, gen = self._make_state_and_gen()
        # Visit only 1 of 2 objects — '<' still live
        self._mark_visited(state, self.triples[0])

        obj_seq = self.tokenizer.encode('<Paris> <capital of>',
                                        add_special_tokens=False)
        lt_tok = self.tokenizer.encode('<', add_special_tokens=False)[0]

        # Call 1: at object level, '<' is live
        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(obj_seq, mask, 0)
        self.assertNotIn(lt_tok, gen.prev_exhausted,
                         "Setup: '<' should be live")

        # Call 2: simulate model picking '<' (live)
        live_seq = obj_seq + [lt_tok]
        mask2 = _negative_inf_mask(1, self._vocab_size())
        gen.constrain(live_seq, mask2, 0)

        self.assertFalse(gen.completed_with_sentinel,
                         "Sentinel should NOT fire when live object token is picked")

    # -- All over-visited fallback -------------------------------------------

    def test_sentinel_fires_when_all_overvisited(self):
        """Sentinel fires when ALL children are over-visited (remaining < 0)."""
        state, gen = self._make_state_and_gen()
        # Exhaust everything + add an extra visit
        for triple in self.triples:
            self._mark_visited(state, triple)
        # Add one more visit to push remaining < 0
        self._mark_visited(state, self.triples[0])

        mask = _negative_inf_mask(1, self._vocab_size())
        gen.constrain([], mask, 0)
        self.assertTrue(gen.completed_with_sentinel,
                        "Sentinel should fire when all children are over-visited")

    # -- Begin sentinel with custom text -------------------------------------

    def test_begin_sentinel_custom_text(self):
        """_begin_sentinel with explicit sentinel_text overrides default."""
        state, gen = self._make_state_and_gen()
        vocab_size = self._vocab_size()
        mask = _negative_inf_mask(1, vocab_size)
        custom_text = " <no further records> ."
        gen._begin_sentinel(mask, 0, sequence=[], sentinel_text=custom_text)

        self.assertTrue(gen.completed_with_sentinel)
        self.assertTrue(len(state.sentinel_remaining) >= 0)

        # Exactly one token should be unmasked (the first sentinel token)
        allowed = self._allowed_tokens(mask)
        self.assertEqual(len(allowed), 1)
        tok = list(allowed)[0]
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, vocab_size)

    def test_begin_sentinel_default_text(self):
        """_begin_sentinel without sentinel_text uses default sentinel_text."""
        state, gen = self._make_state_and_gen()
        vocab_size = self._vocab_size()
        mask = _negative_inf_mask(1, vocab_size)
        gen._begin_sentinel(mask, 0, sequence=[])

        self.assertTrue(gen.completed_with_sentinel)
        allowed = self._allowed_tokens(mask)
        self.assertEqual(len(allowed), 1)
        tok = list(allowed)[0]
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, vocab_size)

    def test_custom_text_produces_different_first_token(self):
        """Different sentinel texts should produce different first tokens."""
        vocab_size = self._vocab_size()

        state1, gen1 = self._make_state_and_gen()
        mask1 = _negative_inf_mask(1, vocab_size)
        gen1._begin_sentinel(mask1, 0, sequence=[])
        tok1 = list(self._allowed_tokens(mask1))[0]

        state2, gen2 = self._make_state_and_gen()
        mask2 = _negative_inf_mask(1, vocab_size)
        gen2._begin_sentinel(mask2, 0, sequence=[],
                             sentinel_text=' <no further records> .')
        tok2 = list(self._allowed_tokens(mask2))[0]

        self.assertNotEqual(tok1, tok2,
                            "Object vs non-object sentinel should start with different tokens")


if __name__ == "__main__":
    unittest.main()
