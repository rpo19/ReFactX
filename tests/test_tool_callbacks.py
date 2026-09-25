"""Tests for the exact decoder-side count/sentinel callbacks.

These require torch/transformers and are skipped when those are unavailable.
"""

import math
import unittest

try:
    import torch
    from transformers import AutoTokenizer

    from refactx.index import DictIndex
    from refactx.generate import (
        CountBranchesGeneration,
        FactGeneration,
        PatternConstrainedState,
    )

    HAVE_TORCH = True
except Exception:  # pragma: no cover - environment without torch
    HAVE_TORCH = False


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not available")
class TestDecoderToolCallbacks(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb = DictIndex()
        triples = ["<Paris> <capital of> <France> .", "<Paris> <country> <France> ."]
        ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for triple_ids in ids:
            self.kb.add(triple_ids)

    def _state(self):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )

    def _mask(self):
        return torch.full((1, len(self.tokenizer)), -math.inf)

    def test_count_callback_fires_once_per_call(self):
        state = self._state()
        calls = []
        state.on_count_generated = lambda prefix, count: calls.append((prefix, count))
        gen = CountBranchesGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0, kb_index=self.kb,
        )
        if gen.delimiter_token is None:
            self.skipTest("tokenizer produced no delimiter token for newline")
        # Simulate a completed `<count> <Paris> <country>` prefix at the newline.
        gen.prefix_tokens = self.tokenizer.encode(
            " <Paris> <country>", add_special_tokens=False)
        gen.constrain([gen.delimiter_token], self._mask(), 0)
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0][1], int)

    def test_sentinel_callback_fires(self):
        state = self._state()
        calls = []
        state.on_sentinel_generated = lambda: calls.append(True)
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=True, eot=None,
        )
        gen._begin_sentinel(self._mask(), 0, sequence=[1, 2, 3])
        self.assertEqual(len(calls), 1)

    def test_sentinel_callback_fires_only_once(self):
        state = self._state()
        calls = []
        state.on_sentinel_generated = lambda: calls.append(True)
        gen = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.kb, sentinel=True, eot=None,
        )
        gen._begin_sentinel(self._mask(), 0, sequence=[1, 2, 3])
        gen._begin_sentinel(self._mask(), 0, sequence=[1, 2, 3])
        self.assertEqual(len(calls), 1)

    def test_callbacks_absent_by_default(self):
        state = self._state()
        self.assertIsNone(state.on_count_generated)
        self.assertIsNone(state.on_sentinel_generated)
        self.assertIsNone(state.on_triple_generated)


if __name__ == "__main__":
    unittest.main()
