"""
Tests for XgrammarConstrainedProcessor.

Tests the custom FSM logit processor that uses xgrammar for JSON constraint
and DictIndex for KB triple constraint.
"""
import unittest
import math
import torch
from transformers import AutoTokenizer

import xgrammar as xgr
from refactx.index import DictIndex
from refactx.xgrammar_matcher import XgrammarConstrainedProcessor


class TestFSMStateTransitions(unittest.TestCase):
    """Test the FSM state machine (free -> triple -> json)."""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb_index = DictIndex()

        # Populate KB with some triples
        triples = [
            "<Paris> <capital of> <France>",
            "<France> <continent> <Europe>",
            "<Mont Blanc> <elevation> <4808 meters>",
        ]
        triple_ids = self.tokenizer(
            triples, add_special_tokens=False, padding=False
        )["input_ids"]
        for ids in triple_ids:
            self.kb_index.add(ids)

        self.json_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

    def test_starts_in_free_mode(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index, json_schema=self.json_schema
        )
        self.assertEqual(processor.mode, "free")

    def test_transitions_to_triple_on_fact_pattern(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index
        )

        scores = torch.zeros(1, len(self.tokenizer))
        fact_ids = self.tokenizer.encode("Some text Fact:", add_special_tokens=False)
        input_ids = torch.tensor([fact_ids])

        for i in range(len(fact_ids)):
            input_ids = torch.tensor([fact_ids[: i + 1]])
            scores = torch.zeros(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "triple")

    def test_transitions_to_json_on_answer_pattern(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index, json_schema=self.json_schema
        )

        answer_ids = self.tokenizer.encode(
            "Some reasoning Answer:", add_special_tokens=False
        )
        input_ids = torch.tensor([answer_ids])
        scores = torch.zeros(1, len(self.tokenizer))

        for i in range(len(answer_ids)):
            input_ids = torch.tensor([answer_ids[: i + 1]])
            scores = torch.zeros(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "json")

    def test_free_mode_no_masking(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index
        )

        # In free mode, all scores should remain unchanged
        ids = self.tokenizer.encode("Hello world", add_special_tokens=False)
        input_ids = torch.tensor([ids])
        scores = torch.randn(1, len(self.tokenizer))
        original = scores.clone()

        for i in range(len(ids)):
            input_ids = torch.tensor([ids[: i + 1]])
            _ = processor(input_ids, scores)

        torch.testing.assert_close(scores, original)


class TestTripleConstraint(unittest.TestCase):
    """Test that triple mode constrains to valid KB triples."""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb_index = DictIndex()
        triples = ["<Paris> <capital of> <France> ."]
        triple_ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for ids in triple_ids:
            self.kb_index.add(ids)

    def test_triple_allows_kb_tokens(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index
        )

        # Generate "Fact:" to trigger triple mode
        fact_ids = self.tokenizer.encode("Fact:", add_special_tokens=False)
        scores = torch.zeros(1, len(self.tokenizer))
        for i in range(len(fact_ids)):
            input_ids = torch.tensor([fact_ids[: i + 1]])
            scores = torch.zeros(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "triple")

        allowed = torch.where(scores[0] != -math.inf)[0]
        self.assertGreater(len(allowed), 0, "Should allow at least one token")

        # After Fact:, the first token of "<Paris> <capital of> <France> ."
        # should be in the allowed set
        triple_start = self.tokenizer.encode("<", add_special_tokens=False)[0]
        self.assertIn(triple_start, allowed.tolist())

    def test_triple_resets_when_exhausted(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index
        )

        # Add the single valid triple sequence
        triple_ids = self.tokenizer.encode(
            "<Paris> <capital of> <France> .", add_special_tokens=False
        )

        # Trigger triple mode
        fact_ids = self.tokenizer.encode("Fact:", add_special_tokens=False)
        scores = torch.zeros(1, len(self.tokenizer))
        all_ids = fact_ids + triple_ids

        for i in range(len(all_ids)):
            input_ids = torch.tensor([all_ids[: i + 1]])
            scores = torch.zeros(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "free")
        self.assertGreater(len(processor.generated_triples), 0)


class TestJSONConstraint(unittest.TestCase):
    """Test that JSON mode constrains to valid JSON schema."""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb_index = DictIndex()
        self.json_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

    def test_json_constrains_to_valid_tokens(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index, json_schema=self.json_schema
        )

        # Generate "Answer:" to trigger JSON mode
        answer_ids = self.tokenizer.encode("Answer:", add_special_tokens=False)
        scores = torch.zeros(1, len(self.tokenizer))
        for i in range(len(answer_ids)):
            input_ids = torch.tensor([answer_ids[: i + 1]])
            scores = torch.zeros(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "json")

        allowed = torch.where(scores[0] != -math.inf)[0]
        self.assertGreater(len(allowed), 0)

        # With JSON schema {"answer": "string"}, the only valid next
        # token after "Answer:" should be "{" (open object)
        open_brace = self.tokenizer.encode("{", add_special_tokens=False)[0]
        self.assertIn(open_brace, allowed.tolist())

    def test_json_completes_and_returns_to_free(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index, json_schema=self.json_schema
        )

        json_str = (
            'Answer: {"answer": "test"}'
        )

        ids = self.tokenizer.encode(json_str, add_special_tokens=False)
        scores = torch.randn(1, len(self.tokenizer))

        for i in range(len(ids)):
            input_ids = torch.tensor([ids[: i + 1]])
            scores = torch.randn(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "free")


class TestComplexFSM(unittest.TestCase):
    """Test full FSM pipeline: free -> triple -> free -> json."""

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.kb_index = DictIndex()
        triples = ["<Paris> <capital of> <France> ."]
        triple_ids = self.tokenizer(triples, add_special_tokens=False)["input_ids"]
        for ids in triple_ids:
            self.kb_index.add(ids)

        self.json_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
        }

    def test_multiple_transitions(self):
        processor = XgrammarConstrainedProcessor(
            self.tokenizer, self.kb_index, json_schema=self.json_schema
        )

        text = "Let me find a fact. Fact: <Paris> <capital of> <France> . "
        text += "Now the answer. Answer: {\"answer\": \"Paris\"}"

        ids = self.tokenizer.encode(text, add_special_tokens=False)
        scores = torch.randn(1, len(self.tokenizer))

        for i in range(len(ids)):
            input_ids = torch.tensor([ids[: i + 1]])
            scores = torch.randn(1, len(self.tokenizer))
            scores = processor(input_ids, scores)

        self.assertEqual(processor.mode, "free")


if __name__ == "__main__":
    unittest.main()
