"""Tests for the decoder-side retrieval budget.

After ``max_tool_calls`` completed fact/count calls the processor must force
the answer section, and once the answer has started it must forbid new tool
calls.  This prevents the long retrieval loops that otherwise hit
``max_new_tokens`` on small models.
"""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.generate import (
    ConstrainedLogitsProcessor,
    ConstrainedStateList,
)


class TestRetrievalBudget(unittest.TestCase):
    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.states = ConstrainedStateList(
            'auto', num_beams=1, num_batches=1,
            debug_tokenizer=self.tokenizer,
        )
        self.proc = ConstrainedLogitsProcessor(
            self.states, self.tokenizer, max_tool_calls=2,
        )
        self.state = self.states[0, 0]

    def _step(self, token_ids):
        input_ids = torch.tensor([token_ids])
        scores = torch.zeros(1, len(self.tokenizer))
        return self.proc(input_ids, scores)

    def test_budget_forces_answer_prefix(self):
        # First call initializes the state input_ids.
        self._step([101])

        # No budget yet: every token is allowed.
        scores = self._step([101, 102])
        self.assertTrue(torch.isfinite(scores).all())

        # Simulate two completed tool calls.
        self.state.tool_calls = 2
        scores = self._step([101, 102, 103])

        finite = torch.isfinite(scores[0]).nonzero(as_tuple=False).flatten()
        expected = self.tokenizer.encode('\n<answer>\n', add_special_tokens=False)[0]
        self.assertEqual(finite.tolist(), [expected])

    def test_tool_calls_blocked_after_answer_started(self):
        self._step([101])
        self.state.tool_calls = 2
        self.state.answer_prefix_remaining = []
        self.state.answer_started = True

        scores = self._step([101, 102])
        blocked = self.proc._tool_block_ids()
        for token in blocked:
            self.assertEqual(scores[0, token].item(), -math.inf)
        # A punctuation token used inside a normal answer stays available.
        self.assertTrue(torch.isfinite(scores[0, self.tokenizer.encode('[')[0]]).item())

    def test_reset_clears_budget_state(self):
        self.state.tool_calls = 5
        self.state.answer_started = True
        self.state.answer_prefix_remaining = [1, 2]
        self.state.reset(clear_patterns=False)
        self.assertEqual(self.state.tool_calls, 0)
        self.assertFalse(self.state.answer_started)
        self.assertEqual(self.state.answer_prefix_remaining, [])


if __name__ == '__main__':
    unittest.main()
