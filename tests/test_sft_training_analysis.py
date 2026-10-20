"""Tests for utils/sft_training_analysis.py helpers."""
import json
import unittest

from utils.sft_training_analysis import (
    answer_shape,
    categorize,
    gold_status,
    question_type,
    tool_pattern,
)


def _rec(full_prediction="", triples=None, prediction=None, gt_answer=None, correct=0,
         reached_max_tokens=False, input_sample=None, question="q"):
    return {
        "full_prediction": full_prediction,
        "triples": triples or [],
        "prediction": prediction,
        "gt_answer": gt_answer,
        "question": question,
        "input_sample": input_sample or {},
        "answer_complete": prediction is not None,
        "reached_max_tokens": reached_max_tokens,
        "new_tokens_generated": 100,
        "evaluation": {"precision": 0, "recall": 0, "f1": 0, "correct": correct, "dont_know": 0},
    }


class TestShapesAndTypes(unittest.TestCase):

    def test_answer_shapes(self):
        self.assertEqual(answer_shape(["Yes"]), "yes/no")
        self.assertEqual(answer_shape(["13 June 1946"]), "date")
        self.assertEqual(answer_shape(["11"]), "numeric")
        self.assertEqual(answer_shape(["a", "b"]), "list")
        self.assertEqual(answer_shape(["Paris"]), "entity")

    def test_question_types(self):
        self.assertEqual(question_type("mintaka", {"complexityType": "count"}), "count")
        self.assertEqual(question_type("2wiki", {"type": "compositional"}), "compositional")
        self.assertEqual(question_type("webqsp", {}), "generic")
        self.assertEqual(question_type("cwq", {"q_entity": ["a", "b"]}), "multi-entity")
        self.assertEqual(question_type("cwq", {"q_entity": ["a"]}), "single-entity")

    def test_tool_patterns(self):
        self.assertEqual(tool_pattern({"full_prediction": "<fact> x </fact>"})[0], "fact-only")
        self.assertEqual(tool_pattern({"full_prediction": "<count> x\n</count>"})[0], "count")
        self.assertEqual(tool_pattern({"full_prediction": "no further records"})[0], "sentinel")
        self.assertEqual(tool_pattern({"full_prediction": "<count> x <no further records>"})[0], "count+sentinel")
        self.assertEqual(tool_pattern({"full_prediction": "plain"})[0], "none")


class TestGoldStatus(unittest.TestCase):

    def test_2wiki_evidence_retrieved(self):
        inp = {"evidences": json.dumps([["Move (1970 film)", "director", "Stuart Rosenberg"]])}
        rec = _rec(triples=[" <Move (1970 film)> <director> <Stuart Rosenberg> ."])
        status = gold_status("2wiki", inp, ["Stuart Rosenberg"], rec)
        self.assertTrue(status["gold_retrieved"])
        self.assertEqual(status["evidence_recall"], 1.0)

    def test_2wiki_evidence_missing(self):
        inp = {"evidences": json.dumps([["Move (1970 film)", "director", "Stuart Rosenberg"]])}
        status = gold_status("2wiki", inp, ["Stuart Rosenberg"], _rec(triples=[]))
        self.assertFalse(status["gold_retrieved"])
        self.assertEqual(status["evidence_recall"], 0.0)

    def test_webqsp_graph_availability(self):
        inp = {"graph": [["France", "capital", "Paris"]]}
        status = gold_status("webqsp", inp, ["Paris"], _rec(triples=[]))
        self.assertFalse(status["gold_retrieved"])
        self.assertTrue(status["gold_available"])


class TestCategorize(unittest.TestCase):

    def test_correct(self):
        rec = _rec(prediction=["Paris"], gt_answer=["Paris"], correct=1,
                   triples=[" <France> <capital> <Paris> ."])
        self.assertEqual(categorize(rec, "webqsp")["category"], "correct")

    def test_truncation(self):
        rec = _rec(prediction=None, gt_answer=["Paris"], reached_max_tokens=True)
        self.assertEqual(categorize(rec, "webqsp")["category"], "truncation")

    def test_misevaluated_date(self):
        rec = _rec(prediction=["13-Jun-46"], gt_answer=["13 June 1946"])
        self.assertEqual(categorize(rec, "2wiki")["category"], "misevaluated")

    def test_reasoning_error(self):
        rec = _rec(prediction=["London"], gt_answer=["Paris"],
                   triples=[" <France> <capital> <Paris> ."])
        self.assertEqual(categorize(rec, "mintaka")["category"], "reasoning_error")

    def test_retrieval_failure(self):
        inp = {"graph": [["France", "capital", "Paris"]]}
        rec = _rec(prediction=["London"], gt_answer=["Paris"], input_sample=inp)
        self.assertEqual(categorize(rec, "webqsp")["category"], "retrieval_failure")

    def test_missing_or_derived(self):
        rec = _rec(prediction=["19"], gt_answer=["20"])
        self.assertEqual(categorize(rec, "mintaka")["category"], "missing_or_derived")


if __name__ == "__main__":
    unittest.main()
