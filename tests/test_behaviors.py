"""Tests for utils/behaviors.py (answer-support classifier)."""
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


class TestSupportContract(unittest.TestCase):

    def test_question_is_a_support_noul(self):
        questions = build_questions()
        self.assertEqual(set(questions), {"answer_supported"})
        question = questions["answer_supported"]
        self.assertEqual(question["type"], "noul")
        self.assertIn("supported by these facts", question["instructions"])
        self.assertIn("Do not use additional knowledge", question["instructions"])

    def test_state_uses_the_log_triples_field(self):
        triples = [" <A> <r> <B> .", " <B> <r2> <C> ."]
        state = build_state(_record(triples=triples), "mintaka")
        self.assertEqual(state["facts"], triples)  # verbatim, no regex
        self.assertEqual(state["answer"], ["Paris"])
        self.assertEqual(state["question"], "What is the capital of France?")
        self.assertNotIn("model_trace", state)

    def test_parse_supported(self):
        response = {"model": "jev-1.14.0",
                    "answers": {"answer_supported": {"type": "noul", "noul": 0.93}},
                    "usage": {"input_tokens": 100, "output_tokens": 0}}
        parsed = parse_jev_response(response)
        self.assertEqual(parsed["label"], "supported")
        self.assertAlmostEqual(parsed["probability"], 0.93)
        self.assertAlmostEqual(parsed["confidence"], 0.93)

    def test_parse_unsupported_and_threshold(self):
        response = {"answers": {"answer_supported": {"type": "noul", "noul": 0.2}}}
        self.assertEqual(parse_jev_response(response)["label"], "unsupported")
        self.assertEqual(parse_jev_response(response, threshold=0.1)["label"], "supported")


class TestHeuristic(unittest.TestCase):

    def test_answer_in_facts_is_supported(self):
        self.assertEqual(heuristic_behavior(_record(), "mintaka")["label"], "supported")

    def test_answer_not_in_facts_is_unsupported(self):
        rec = _record(prediction=["Berlin"])
        self.assertEqual(heuristic_behavior(rec, "mintaka")["label"], "unsupported")

    def test_missing_answer_is_unsupported(self):
        self.assertEqual(heuristic_behavior(_record(prediction=None), "mintaka")["label"],
                         "unsupported")


if __name__ == "__main__":
    unittest.main()
