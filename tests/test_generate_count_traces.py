"""Tests for utils/generate_count_traces.py."""
import json
import tempfile
import unittest
from pathlib import Path

from utils.generate_count_traces import (
    GoldCountSource,
    build_record,
    inject_count_result,
    mock_generate,
    parse_count_calls,
    read_candidates,
    validate_trace,
)

TRACE = (
    "<count> <France> <shares border with>\n"
    "= 8\n"
    "</count>\n"
    "The count tool returned 8.\n\n"
    "<answer>\n[\"8\"]\n</answer>"
)


class TestCountParsing(unittest.TestCase):

    def test_closed_call(self):
        calls = parse_count_calls(TRACE)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["subject"], "France")
        self.assertEqual(calls[0]["relation"], "shares border with")
        self.assertEqual(calls[0]["value"], 8.0)
        self.assertTrue(calls[0]["closed"])

    def test_unclosed_call(self):
        calls = parse_count_calls("<count> <France> <shares border with>\n")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["value"])
        self.assertFalse(calls[0]["closed"])

    def test_inject_result_into_unclosed_call(self):
        text = "<count> <France> <shares border with>\n"
        call = parse_count_calls(text)[-1]
        injected = inject_count_result(text, call, 8.0)
        self.assertIn("= 8\n</count>", injected)
        self.assertEqual(parse_count_calls(injected)[-1]["value"], 8.0)

    def test_validate_good_and_bad(self):
        ok, reasons = validate_trace(TRACE, 8.0)
        self.assertTrue(ok)
        self.assertEqual(reasons, [])
        ok, reasons = validate_trace(TRACE, 9.0)
        self.assertFalse(ok)
        self.assertTrue(any("!=" in r for r in reasons))

    def test_validate_missing_count(self):
        ok, reasons = validate_trace('<answer>\n["8"]\n</answer>', 8.0)
        self.assertFalse(ok)
        self.assertTrue(any("no <count>" in r for r in reasons))


class TestMockGeneration(unittest.TestCase):

    def test_mock_and_build_record(self):
        record = {"question": "How many countries border France?",
                  "gt_answer": "8", "input_sample": {}, "_dataset": "mintaka"}
        result = mock_generate(record, GoldCountSource())
        self.assertTrue(result["ok"])
        sample = build_record(record, record["question"], result, "system",
                              {"type": "mock", "model": "mock"})
        self.assertEqual(sample["prediction"], ["8"])
        self.assertTrue(sample["synthetic"])
        self.assertEqual(sample["evaluation"]["correct"], 1)
        self.assertTrue(sample["verification"]["ok"])


class TestCandidateSelection(unittest.TestCase):

    def _write_log(self, path):
        header = {"dataset": "rpozzi/mintaka", "dataset_split": "train"}
        records = [
            {"question": "q1", "input_sample": {"complexityType": "count"},
             "gt_answer": "8", "evaluation": {"correct": 1},
             "full_prediction": "<count> a b\n= 8\n</count>"},
            {"question": "q2", "input_sample": {"complexityType": "count"},
             "gt_answer": "3", "evaluation": {"correct": 0},
             "full_prediction": "no tool"},
            {"question": "q3", "input_sample": {"complexityType": "generic"},
             "gt_answer": "x", "evaluation": {"correct": 0}, "full_prediction": ""},
        ]
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_selects_count_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            self._write_log(path)
            all_count = read_candidates(path, select="all")
            self.assertEqual([r["question"] for r in all_count], ["q1", "q2"])
            wrong = read_candidates(path, select="teacher-wrong-or-no-count")
            self.assertEqual([r["question"] for r in wrong], ["q2"])


if __name__ == "__main__":
    unittest.main()
