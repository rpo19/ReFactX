"""Tests for utils/behaviors.py."""
import unittest

from utils.behaviors import (
    answer_shape,
    build_questions,
    build_state,
    calculate_metrics,
    heuristic_behavior,
    parse_answer,
    parse_jev_response,
    question_type,
)


def _record(**kw):
    rec = {
        "question": "What is the capital of France?",
        "gt_answer": "Paris",
        "prediction": ["Paris"],
        "triples": [" <France> <capital> <Paris> ."],
        "full_prediction": "<fact> <France> <capital> <Paris> . </fact>\n<answer>\n[\"Paris\"]\n</answer>",
        "input_sample": {"complexityType": "generic"},
        "evaluation": {"correct": 1, "precision": 1, "recall": 1, "f1": 1, "dont_know": 0},
        "reached_max_tokens": False,
    }
    rec.update(kw)
    return rec


class TestParsingAndMetrics(unittest.TestCase):

    def test_parse_answer_multiline(self):
        self.assertEqual(parse_answer('<answer>\n["a", "b"]\n</answer>'), ["a", "b"])

    def test_parse_answer_inline(self):
        self.assertEqual(parse_answer('<answer>["x"]</answer>'), ["x"])

    def test_parse_answer_missing(self):
        self.assertIsNone(parse_answer("no answer here"))

    def test_metrics_match_eval(self):
        metrics = calculate_metrics(["Paris"], ["paris"])
        self.assertEqual(metrics["correct"], 1)
        self.assertEqual(metrics["f1"], 1.0)

    def test_types_and_shapes(self):
        self.assertEqual(question_type("mintaka", {"complexityType": "count"}), "count")
        self.assertEqual(question_type("webqsp", {}), "generic")
        self.assertEqual(answer_shape(["Yes"]), "yes/no")
        self.assertEqual(answer_shape(["13 June 1946"]), "date")
        self.assertEqual(answer_shape(["11"]), "numeric")
        self.assertEqual(answer_shape(["a", "b"]), "list")


class TestJevContract(unittest.TestCase):

    def test_questions_have_four_labels_and_nouls(self):
        questions = build_questions()
        self.assertEqual(set(questions["behavior"]["criteria"]),
                         {"best", "acceptable", "bad", "error"})
        self.assertEqual(questions["answer_supported_by_facts"]["type"], "noul")
        self.assertEqual(questions["derived_from_facts"]["type"], "noul")

    def test_state_uses_verified_correctness(self):
        state = build_state(_record(evaluation={"correct": 0}), "mintaka")
        # The strict evaluator said wrong, but the answers are identical.
        self.assertTrue(state["answer_is_correct"])
        self.assertFalse(state["evaluator_correct"])
        self.assertIn("retrieved_facts", state)
        self.assertIn("model_trace", state)

    def test_parse_response(self):
        response = {
            "model": "jev-1.14.0",
            "answers": {
                "behavior": {"type": "choice", "choice": "best", "confidence": 0.9,
                             "probabilities": {"best": 0.9, "acceptable": 0.1}},
                "answer_supported_by_facts": {"type": "noul", "noul": 0.95},
                "derived_from_facts": {"type": "noul", "noul": 0.9},
            },
            "usage": {"input_tokens": 100, "output_tokens": 0},
        }
        parsed = parse_jev_response(response)
        self.assertEqual(parsed["label"], "best")
        self.assertEqual(parsed["diagnostic"], "best")
        self.assertAlmostEqual(parsed["supported_probability"], 0.95)

    def test_wrong_with_facts_is_flagged(self):
        response = {"answers": {
            "behavior": {"type": "choice", "choice": "error", "confidence": 0.8, "probabilities": {}},
            "answer_supported_by_facts": {"type": "noul", "noul": 0.9},
            "derived_from_facts": {"type": "noul", "noul": 0.1},
        }}
        self.assertEqual(parse_jev_response(response)["diagnostic"], "wrong_with_facts")


class TestHeuristic(unittest.TestCase):

    def test_grounded_correct_is_best(self):
        self.assertEqual(heuristic_behavior(_record(), "mintaka")["label"], "best")

    def test_correct_without_support_is_bad(self):
        rec = _record(gt_answer="Berlin", prediction=["Berlin"],
                      triples=[" <France> <capital> <Paris> ."],
                      full_prediction="Berlin")
        self.assertEqual(heuristic_behavior(rec, "mintaka")["label"], "bad")

    def test_wrong_is_error(self):
        rec = _record(gt_answer="London", prediction=["Paris"],
                      evaluation={"correct": 0})
        self.assertEqual(heuristic_behavior(rec, "mintaka")["label"], "error")


if __name__ == "__main__":
    unittest.main()
