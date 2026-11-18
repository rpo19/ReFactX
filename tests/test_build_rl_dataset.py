"""Tests for the mixed RL dataset builder helpers (no network)."""
import json
import tempfile
import unittest
from pathlib import Path

from utils.build_rl_dataset import (
    infer_question_type,
    load_excluded_questions,
    parse_answer_values,
)


class TestAnswerParsing(unittest.TestCase):

    def test_plain_strings(self):
        self.assertEqual(parse_answer_values("yes"), ["yes"])
        self.assertEqual(parse_answer_values("Mount Lucania"), ["Mount Lucania"])

    def test_numeric_string(self):
        self.assertEqual(parse_answer_values("1965"), ["1965"])

    def test_rog_list_string(self):
        self.assertEqual(parse_answer_values("['A', 'B']"), ["A", "B"])

    def test_native_list(self):
        self.assertEqual(parse_answer_values(["A", None, "B"]), ["A", "B"])

    def test_empty(self):
        self.assertEqual(parse_answer_values(None), [])
        self.assertEqual(parse_answer_values(""), [])


class TestQuestionType(unittest.TestCase):

    def test_mintaka(self):
        self.assertEqual(
            infer_question_type("mintaka", {"complexityType": "count"}), "count"
        )

    def test_2wiki(self):
        self.assertEqual(
            infer_question_type("2wiki", {"type": "bridge_comparison"}),
            "bridge_comparison",
        )

    def test_cwq_entity_count(self):
        self.assertEqual(
            infer_question_type("cwq", {"q_entity": "['A', 'B']"}), "multi-entity"
        )
        self.assertEqual(infer_question_type("cwq", {"q_entity": "['A']"}), "single-entity")


class TestExclusions(unittest.TestCase):

    def test_reads_question_and_input_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exclude.jsonl"
            path.write_text(
                json.dumps({"question": "What is X?"}) + "\n"
                + json.dumps({"_report": True}) + "\n"
                + json.dumps({"input_sample": {"question": "Who is Y?"}}) + "\n"
            )
            excluded = load_excluded_questions([str(path)])
            self.assertIn("what is x", excluded)
            self.assertIn("who is y", excluded)
            self.assertEqual(len(excluded), 2)


if __name__ == "__main__":
    unittest.main()
