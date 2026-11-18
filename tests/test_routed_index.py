"""Tests for RoutedIndex — per-state KB table dispatch."""
import unittest
from types import SimpleNamespace

from refactx.index import Index, RoutedIndex


class FakeIndex(Index):
    def __init__(self, name, switch_parameter=7, rootkey=-100):
        super().__init__()
        self.name = name
        self.switch_parameter = switch_parameter
        self.rootkey = rootkey
        self.tokenizer_name = "fake-tokenizer"

    def next_tokens(self, sequence, **kwargs):
        return {ord(self.name[0]): len(sequence)}, {"index": self.name}


class TestRoutedIndex(unittest.TestCase):

    def make(self):
        return RoutedIndex(
            {"a": FakeIndex("a"), "b": FakeIndex("b")}, default="a"
        )

    def test_dispatches_by_state_table(self):
        routed = self.make()
        tokens, extra = routed.next_tokens([1, 2], state=SimpleNamespace(index_table="b"))
        self.assertEqual(tokens, {ord("b"): 2})
        self.assertEqual(extra["index"], "b")

    def test_uses_default_for_missing_or_unknown_table(self):
        routed = self.make()
        for state in (None, SimpleNamespace(), SimpleNamespace(index_table="missing")):
            tokens, extra = routed.next_tokens([1], state=state)
            self.assertEqual(tokens, {ord("a"): 1})
            self.assertEqual(extra["index"], "a")

    def test_set_tokenizer_reaches_all_indexes(self):
        routed = self.make()
        routed.set_tokenizer("tok")
        self.assertEqual(routed.tokenizer, "tok")
        for index in routed.indexes.values():
            self.assertEqual(index.tokenizer, "tok")

    def test_rejects_mismatched_switch_parameters(self):
        with self.assertRaises(ValueError):
            RoutedIndex(
                {"a": FakeIndex("a", switch_parameter=7),
                 "b": FakeIndex("b", switch_parameter=8)},
                default="a",
            )

    def test_rejects_unknown_default(self):
        with self.assertRaises(ValueError):
            RoutedIndex({"a": FakeIndex("a")}, default="b")


class TestRoutedGeneration(unittest.TestCase):
    """FactGeneration must read the table selected for its state."""

    def setUp(self):
        import torch
        from transformers import AutoTokenizer
        from refactx.index import DictIndex

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.paris = DictIndex()
        self.berlin = DictIndex()
        for index, triple in ((self.paris, "<Paris> <capital of> <France> ."),
                              (self.berlin, "<Berlin> <capital of> <Germany> .")):
            index.set_tokenizer(self.tokenizer)
            index.add(self.tokenizer(triple, add_special_tokens=False)["input_ids"])
        self.routed = RoutedIndex({"paris": self.paris, "berlin": self.berlin},
                                  default="paris")

    def _allowed(self, table):
        from refactx.generate import FactGeneration, PatternConstrainedState
        from refactx.index import DictIndex

        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex()
        )
        state.index_table = table
        generation = FactGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0,
            index=self.routed, sentinel=False,
        )
        mask = self.torch.full((1, len(self.tokenizer)), -float("inf"))
        # Both triples start with the same '<' token, so advance one token to
        # reach the table-specific subject branch.
        first = self.tokenizer("<Paris>", add_special_tokens=False)["input_ids"][0]
        generation.constrain([first], mask, 0)
        return set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())

    def test_fact_generation_uses_state_table(self):
        paris_allowed = self._allowed("paris")
        berlin_allowed = self._allowed("berlin")
        self.assertNotEqual(paris_allowed, berlin_allowed)
        self.assertTrue(paris_allowed)
        self.assertTrue(berlin_allowed)


if __name__ == "__main__":
    unittest.main()
