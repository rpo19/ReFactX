"""Tests for utils/classify_behaviors.py (offline mock client)."""
import json
import tempfile
import unittest
from pathlib import Path

from utils.classify_behaviors import _mock_response, classify_log
from utils.jev_client import MockJevClient


def _record(question, gt, pred, triples, trace, correct):
    return {
        "question": question,
        "gt_answer": gt,
        "prediction": pred,
        "triples": triples,
        "full_prediction": trace,
        "input_sample": {"complexityType": "generic"},
        "answer_complete": True,
        "reached_max_tokens": False,
        "new_tokens_generated": 10,
        "evaluation": {"precision": 0, "recall": 0, "f1": 0, "correct": correct, "dont_know": 0},
    }


class TestClassifyLog(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.input = self.dir / "log.out.0"
        header = {"dataset": "rpozzi/mintaka", "dataset_split": "train", "model_name": "x"}
        records = [
            _record("cap of France?", "Paris", ["Paris"],
                    [" <France> <capital> <Paris> ."],
                    '<fact> <France> <capital> <Paris> . </fact>\n<answer>\n["Paris"]\n</answer>', 1),
            _record("capital of Germany?", "Berlin", ["Berlin"],
                    [" <France> <capital> <Paris> ."], "Berlin", 1),
            _record("capital of UK?", "London", ["London"],
                    [" <France> <capital> <Paris> ."], "Paris", 0),
        ]
        with self.input.open("w") as f:
            f.write(json.dumps(header) + "\n")
            for record in records:
                f.write(json.dumps(record) + "\n")
            f.write(json.dumps({"macro_precision": 0.5, "macro_correct_accuracy": 0.5}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kwargs):
        output = self.dir / "out.behaviors"
        client = MockJevClient(_mock_response)
        result = classify_log(self.input, output, client, workers=2, quiet=True, **kwargs)
        return output, result

    def test_labels_and_summary(self):
        output, result = self._run()
        lines = [json.loads(line) for line in output.read_text().splitlines()]
        header, records, summary = lines[0], lines[1:-1], lines[-1]
        self.assertIn("behavior_metadata", header)
        self.assertEqual([r["behavior"]["label"] for r in records],
                         ["supported", "unsupported", "unsupported"])
        self.assertEqual(summary["behavior_counts"], {"supported": 1, "unsupported": 2})
        self.assertEqual(result["written"], 3)

    def test_limit(self):
        output, result = self._run(limit=2)
        self.assertEqual(result["written"], 2)

    def test_start_skips_records(self):
        output, result = self._run(start=1)
        self.assertEqual(result["written"], 2)
        first = json.loads(output.read_text().splitlines()[1])
        self.assertEqual(first["behavior"]["label"], "unsupported")


if __name__ == "__main__":
    unittest.main()
