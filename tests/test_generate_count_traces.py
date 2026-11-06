"""Tests for utils/generate_count_traces.py."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.generate_count_traces import (
    DEFAULT_PROMPT,
    TASK_PROMPT,
    build_generation_message,
    build_record,
    check_sample,
    load_generator_prompt,
    main,
    read_candidates,
)

REPO = Path(__file__).resolve().parents[1]

TRACE = (
    "I need the exact count for this question, so I will use the count tool.\n"
    "<count> <France> <shares border with>\n"
    "= 8\n"
    "</count>\n"
    "The count tool returned 8, which answers the question.\n\n"
    "<answer>\n[\"8\"]\n</answer>"
)

RECORD = {
    "question": "How many countries share a border with France?",
    "gt_answer": "8",
    "input_sample": {"complexityType": "count"},
    "evaluation": {"correct": 0},
    "full_prediction": "no tool",
}


class FakeClient:
    """Stand-in for DeepSeekClient so the CLI can be tested without an API key."""

    def __init__(self, *args, **kwargs):
        pass

    def chat(self, messages, *, temperature=0.7, max_tokens=2048):
        return {"choices": [{"message": {"content": TRACE}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10}}

    def content(self, response):
        choice = response["choices"][0]
        return choice["message"]["content"], choice["finish_reason"], response.get("usage") or {}

    def usage_summary(self):
        return {}


class TestPrompts(unittest.TestCase):

    def test_wrapper_includes_task_prompt(self):
        prompt = load_generator_prompt(REPO / DEFAULT_PROMPT, REPO / TASK_PROMPT)
        self.assertIn("Synthetic training-example generator", prompt)
        self.assertIn("Real task prompt", prompt)
        self.assertIn("Fact-Grounded Question Answering", prompt)
        self.assertNotIn("{{TASK_PROMPT}}", prompt)

    def test_generation_message_is_gold_conditioned(self):
        message = build_generation_message("How many?", "8")
        self.assertIn("Reference answer: 8", message)
        self.assertIn("<count>", message)


class TestChecks(unittest.TestCase):

    def test_check_sample(self):
        ok, reasons = check_sample(TRACE)
        self.assertTrue(ok, reasons)
        ok, reasons = check_sample('<answer>\n["8"]\n</answer>')
        self.assertFalse(ok)
        self.assertTrue(any("<count>" in reason for reason in reasons))
        ok, reasons = check_sample("<count> <France> <border>")
        self.assertFalse(ok)
        self.assertTrue(any("<answer>" in reason for reason in reasons))


class TestRecords(unittest.TestCase):

    def test_build_record(self):
        result = {"trace": TRACE, "finish_reasons": ["stop"], "usages": [], "ok": True, "reason": ""}
        sample = build_record(RECORD, RECORD["question"], result,
                              task_prompt="task", generator_meta={"type": "deepseek"})
        self.assertEqual(sample["prediction"], ["8"])
        self.assertTrue(sample["synthetic"])
        self.assertTrue(sample["verification"]["count_called"])
        self.assertEqual(sample["evaluation"]["correct"], 1)
        # The SFT prompt is the task prompt + question, with no gold answer leak.
        self.assertEqual(sample["prompt"], "task\n\nQuestion: " + RECORD["question"])
        self.assertNotIn("Reference answer", sample["prompt"])


class TestCandidateSelection(unittest.TestCase):

    def write_log(self, path):
        header = {"dataset": "rpozzi/mintaka", "dataset_split": "train"}
        records = [
            {"question": "q1", "input_sample": {"complexityType": "count"},
             "gt_answer": "8", "evaluation": {"correct": 0},
             "full_prediction": "no tool"},
            {"question": "q2", "input_sample": {"complexityType": "count"},
             "gt_answer": "3", "evaluation": {"correct": 1},
             "full_prediction": "<count> a b\n= 3\n</count>"},
            {"question": "q3", "input_sample": {"complexityType": "generic"},
             "gt_answer": "x", "evaluation": {"correct": 0}, "full_prediction": ""},
        ]
        with path.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for record in records:
                f.write(json.dumps(record) + "\n")
            f.write(json.dumps({"macro_precision": 1.0}) + "\n")

    def test_selects_count_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            self.write_log(path)
            self.assertEqual([r["question"] for r in read_candidates(path, select="all")],
                             ["q1", "q2"])
            self.assertEqual(
                [r["question"] for r in read_candidates(path, select="teacher-wrong-or-no-count")],
                ["q1"])


class TestCLI(unittest.TestCase):

    def test_run_with_stub_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            output = Path(tmp) / "counts.jsonl"
            TestCandidateSelection().write_log(path)
            with patch("utils.generate_count_traces.DeepSeekClient", FakeClient):
                main(["--input", str(path), "--output", str(output), "--n", "1", "--quiet"])
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            header, records, summary = lines[0], lines[1:-1], lines[-1]
            self.assertTrue(header["synthetic"])
            self.assertEqual(header["generator"]["type"], "deepseek")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["prediction"], ["8"])
            self.assertTrue(records[0]["verification"]["count_called"])
            self.assertTrue(summary["synthetic_count"])

    def test_continue_skips_existing_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.out"
            output = Path(tmp) / "counts.jsonl"
            TestCandidateSelection().write_log(path)
            with patch("utils.generate_count_traces.DeepSeekClient", FakeClient):
                main(["--input", str(path), "--output", str(output), "--n", "1",
                      "--select", "all", "--quiet"])
                main(["--input", str(path), "--output", str(output), "--n", "2",
                      "--select", "all", "--continue", "--keep-incorrect", "--quiet"])
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            records = [obj for obj in lines if "evaluation" in obj]
            self.assertEqual(len(records), 2)
            self.assertEqual(len({r["question"] for r in records}), 2)
            self.assertEqual(lines[-1]["resumed_from"], 1)
            self.assertEqual(lines[-1]["accepted"], 2)
            self.assertEqual(lines[-1]["attempted_this_run"], 1)


if __name__ == "__main__":
    unittest.main()
