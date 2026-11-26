"""Tests for utils/kg_explorer.py (DictIndex prefix counting/enumeration)."""
import unittest

from refactx.index import DictIndex

from utils.kg_explorer import (
    enumerate_prefix,
    next_token_counts,
    parse_tokens,
    prefix_count,
    prefix_report,
)


def _index():
    idx = DictIndex()
    for seq in ([1, 2, 3, 9], [1, 2, 4, 9], [1, 5, 6, 9], [7, 8, 9]):
        idx.add(list(seq))
    return idx


class TestTrieHelpers(unittest.TestCase):

    def setUp(self):
        self.idx = _index()

    def test_root_count(self):
        self.assertEqual(prefix_count(self.idx, []), 4)

    def test_prefix_count(self):
        self.assertEqual(prefix_count(self.idx, [1]), 3)
        self.assertEqual(prefix_count(self.idx, [1, 2]), 2)
        self.assertEqual(prefix_count(self.idx, [1, 2, 3]), 1)
        self.assertEqual(prefix_count(self.idx, [7]), 1)

    def test_missing_prefix(self):
        self.assertEqual(prefix_count(self.idx, [99]), 0)

    def test_next_tokens(self):
        self.assertEqual(next_token_counts(self.idx, []), {1: 3, 7: 1})
        self.assertEqual(next_token_counts(self.idx, [1]), {2: 2, 5: 1})

    def test_enumerate_does_not_repeat_prefix(self):
        self.assertEqual(enumerate_prefix(self.idx, [1, 2]),
                         [[1, 2, 3, 9], [1, 2, 4, 9]])
        self.assertEqual(enumerate_prefix(self.idx, [7]), [[7, 8, 9]])
        self.assertEqual(enumerate_prefix(self.idx, []),
                         [[1, 2, 3, 9], [1, 2, 4, 9], [1, 5, 6, 9], [7, 8, 9]])

    def test_enumerate_limit(self):
        self.assertEqual(len(enumerate_prefix(self.idx, [], limit=2)), 2)

    def test_report(self):
        rep = prefix_report(self.idx, [1], tokenizer=None, limit=10)
        self.assertEqual(rep["count"], 3)
        self.assertEqual({n["token"]: n["count"] for n in rep["next"]}, {2: 2, 5: 1})
        self.assertEqual([s for s, _ in rep["triples"]],
                         [[1, 2, 3, 9], [1, 2, 4, 9], [1, 5, 6, 9]])
        self.assertFalse(rep["truncated"])

    def test_report_missing_prefix(self):
        rep = prefix_report(self.idx, [99], tokenizer=None)
        self.assertIsNone(rep["count"])
        self.assertIn("not found", rep["note"])


class TestParseTokens(unittest.TestCase):

    def test_token_list_wins(self):
        self.assertEqual(parse_tokens("ignored", "[1, 2, 3]", tokenizer=None), [1, 2, 3])

    def test_token_list_rejects_non_ints(self):
        with self.assertRaises(ValueError):
            parse_tokens("", '[1, "x"]', tokenizer=None)


if __name__ == "__main__":
    unittest.main()
