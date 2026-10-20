"""Tests for utils/build_sft_dataset.py."""
import json
import tempfile
import unittest
from pathlib import Path

from utils.build_sft_dataset import main


def _record(question, behavior, correct=1, triples=None):
    return {
        "question": question,
        "gt_answer": "a",
        "prediction": ["a"],
        "triples": triples if triples is not None else [" <x> <r> <a> ."],
        "full_prediction": '<fact> <x> <r> <a> . </fact>\n<answer>\n["a"]\n</answer>',
        "prompt": "prompt",
        "input_sample": {"complexityType": "generic"},
        "reached_max_tokens": False,
        "evaluation": {"correct": correct, "precision": 1, "recall": 1, "f1": 1, "dont_know": 0},
        "behavior": {"label": behavior, "confidence": 0.9},
    }


def _synthetic(question, dataset="rpozzi/mintaka"):
    record = _record(question, "best")
    record["synthetic"] = True
    record["dataset"] = dataset
    return record


def _write(path, header, records):
    with path.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for record in records:
            f.write(json.dumps(record) + "\n")


class TestBuilder(unittest.TestCase):

    def _setup(self, tmp):
        tmp = Path(tmp)
        mintaka = tmp / "mintaka.behaviors"
        _write(mintaka, {"dataset": "rpozzi/mintaka", "dataset_split": "train"}, [
            _record("m-best", "best"),
            _record("m-acceptable", "acceptable"),
            _record("m-bad", "bad"),
            _record("m-error", "error"),
            _record("m-wrong", "best", correct=0),
        ])
        twowiki_records = (
            [_record(f"w-best-{i}", "best") for i in range(10)]
            + [_record(f"w-acceptable-{i}", "acceptable") for i in range(10)]
        )
        twowiki = tmp / "2wiki.behaviors"
        _write(twowiki, {"dataset": "xanhho/2WikiMultihopQA", "dataset_split": "train"},
               twowiki_records)
        synthetic = tmp / "counts.jsonl"
        _write(synthetic, {"dataset": "rpozzi/mintaka", "synthetic": True},
               [_synthetic("s-1"), _synthetic("s-2")])
        return mintaka, twowiki, synthetic

    def _run(self, tmp, *extra):
        mintaka, twowiki, synthetic = self._setup(tmp)
        output = Path(tmp) / "mix.jsonl"
        main(["--natural", str(mintaka), str(twowiki),
              "--synthetic", str(synthetic),
              "--output", str(output), "--two-wiki-cap", "4", "--seed", "0", "--quiet", *extra])
        lines = [json.loads(line) for line in output.read_text().splitlines()]
        return lines[0], lines[1:]

    def test_selection_and_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, samples = self._run(tmp)
            self.assertEqual(len(samples), 8)  # 2 mintaka + 4 2wiki + 2 synthetic
            by_dataset = {}
            for sample in samples:
                by_dataset[sample["dataset"]] = by_dataset.get(sample["dataset"], 0) + 1
            self.assertEqual(by_dataset["2wiki"], 4)
            # 2Wiki ratio: 3 best + 1 acceptable.
            wiki = [s for s in samples if s["dataset"] == "2wiki"]
            self.assertEqual(sum(s["behavior"] == "best" for s in wiki), 3)
            self.assertEqual(sum(s["behavior"] == "acceptable" for s in wiki), 1)
            self.assertEqual(report["counts"]["by_provenance"]["synthetic"], 2)

    def test_include_bad(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, samples = self._run(tmp, "--include-bad")
            self.assertEqual(len(samples), 9)
            self.assertEqual(sum(s["behavior"] == "bad" for s in samples), 1)

    def test_require_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, _ = self._run(tmp, "--require-behavior")
            # All test records carry a label, so nothing is dropped for this reason.
            self.assertEqual(report["skipped"].get("missing_behavior", 0), 0)


if __name__ == "__main__":
    unittest.main()
