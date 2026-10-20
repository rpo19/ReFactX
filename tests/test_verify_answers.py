"""Tests for the dataset-agnostic verifier in utils/answer_normalization.py."""
import unittest

from utils.answer_normalization import verify_answer, verify_mintaka


class TestVerifyAnswerAcrossDatasets(unittest.TestCase):

    def test_mintaka_ambiguous_two_digit_year(self):
        result = verify_answer(["13-Jun-46"], "13 June 1946", dataset="mintaka")
        self.assertFalse(result.strict_match)
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "date")

    def test_2wiki_date_format(self):
        result = verify_answer(["19-Jun-81"], "June 19, 1981", dataset="2wiki")
        self.assertTrue(result.normalized_match)

    def test_webqsp_alias(self):
        result = verify_answer(["United States"], "USA", dataset="webqsp")
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "alias")

    def test_cwq_accent_folding(self):
        self.assertTrue(verify_answer(["Aarhus"], "Århus", dataset="cwq").normalized_match)

    def test_strict_match_mirrors_evaluator(self):
        # The official evaluator compares lowercased sets.
        self.assertTrue(verify_answer(["Paris"], ["paris"], dataset="webqsp").strict_match)

    def test_cardinality_mismatch_is_rejected(self):
        result = verify_answer(["a", "b"], ["a"], dataset="2wiki")
        self.assertFalse(result.normalized_match)
        self.assertEqual(result.rule, "cardinality")

    def test_quantity_conversion(self):
        result = verify_answer(["170 cm"], "5'7\"", dataset="mintaka")
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "quantity")

    def test_verify_mintaka_is_a_wrapper(self):
        a = verify_mintaka(["August 2, 1970"], "2-Aug-70")
        b = verify_answer(["August 2, 1970"], "2-Aug-70", dataset="mintaka")
        self.assertEqual(a.as_dict(), b.as_dict())


if __name__ == "__main__":
    unittest.main()
