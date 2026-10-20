"""Tests for utils/advanced_error_analysis.py — categorisation helpers."""
import unittest

from utils.advanced_error_analysis import (
    classify,
    is_boolean,
)


def _rec(full_prediction="", triples=None, prediction=None, gt_answer=None,
         correct=0, reached_max_tokens=False, dont_know=0, question="q",
         input_sample=None, new_tokens=100):
    return {
        "full_prediction": full_prediction,
        "triples": triples or [],
        "prediction": prediction,
        "gt_answer": gt_answer,
        "question": question,
        "input_sample": input_sample or {},
        "answer_complete": prediction is not None,
        "reached_max_tokens": reached_max_tokens,
        "new_tokens_generated": new_tokens,
        "evaluation": {"precision": 0, "recall": 0, "f1": 0, "correct": correct, "dont_know": dont_know},
    }


class TestIsBoolean(unittest.TestCase):

    def test_yes_no(self):
        self.assertTrue(is_boolean(["Yes"]))
        self.assertTrue(is_boolean(["no"]))
        self.assertFalse(is_boolean(["Paris"]))
        self.assertFalse(is_boolean([]))


class TestClassify(unittest.TestCase):

    def test_correct(self):
        rec = _rec(prediction=["Paris"], gt_answer="Paris", correct=1)
        self.assertEqual(classify(rec, "mintaka")["category"], "correct")

    def test_truncation_beats_no_answer(self):
        rec = _rec(prediction=None, gt_answer="Paris", reached_max_tokens=True, new_tokens=2048)
        self.assertEqual(classify(rec, "mintaka")["category"], "truncation")

    def test_abstention(self):
        rec = _rec(prediction=["I don't know"], gt_answer="Paris", dont_know=1)
        self.assertEqual(classify(rec, "mintaka")["category"], "abstention")

    def test_no_answer(self):
        rec = _rec(prediction=None, gt_answer="Paris")
        self.assertEqual(classify(rec, "mintaka")["category"], "no_answer")

    def test_normalization_via_ambiguous_two_digit_year(self):
        rec = _rec(prediction=["13-Jun-46"], gt_answer="13 June 1946")
        feat = classify(rec, "2wiki")
        self.assertEqual(feat["category"], "normalization")
        self.assertEqual(feat["rule"], "date")

    def test_wrong_reference_absent(self):
        rec = _rec(prediction=["London"], gt_answer="Paris",
                   triples=[" <X> <capital> <Berlin> ."])
        self.assertEqual(classify(rec, "mintaka")["category"], "wrong_reference_absent")

    def test_wrong_reference_present(self):
        rec = _rec(prediction=["London"], gt_answer="Paris",
                   triples=[" <France> <capital> <Paris> ."])
        self.assertEqual(classify(rec, "mintaka")["category"], "wrong_reference_present")

    def test_incomplete_calls_and_no_sentinel_hallucination(self):
        text = ('<fact> <France> <capital> <Paris> . </fact>'
                '<fact> <France> <capital> <no further records> . </fact>'
                '<fact> <France> <population> <67 million> .')
        rec = _rec(full_prediction=text, prediction=["Paris"], gt_answer="Paris",
                   triples=[" <France> <capital> <Paris> ."], correct=1)
        feat = classify(rec, "mintaka")
        self.assertEqual(feat["incomplete_calls"], 1)
        self.assertEqual(feat["n_hallucinated"], 0)  # sentinel + cut-off call discounted


if __name__ == "__main__":
    unittest.main()
