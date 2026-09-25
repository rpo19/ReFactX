import importlib.util
import unittest
from pathlib import Path


def _load_module(upto="def exact_reward("):
    """Load a slice of run_trl.py without importing trl/torch.

    The module-level imports pull in dotenv/torch, which are absent here, so the
    pure helpers are exec'd directly.  ``upto`` selects how far to slice.
    """
    path = Path(__file__).resolve().parents[1] / "utils" / "run_trl.py"
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _text("))
    end = next(i for i, l in enumerate(lines) if l.startswith(upto))
    snippet = "\n".join(lines[start:end])
    namespace = {}
    exec(
        "from typing import Any, Iterable\nimport json, re, os\n" + snippet,
        namespace,
    )
    return namespace


class TestEvaluatePolicyWithoutEvalSplit(unittest.TestCase):
    """The custom metrics callback is registered without an eval split, so
    evaluate_policy must no-op instead of importing torch and generating."""

    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module(upto="class CustomMetricsCallback")

    def test_none_dataset_returns_empty(self):
        result = self.ns["evaluate_policy"](
            None, None, None, 128, "<fact>", "<answer>",
        )
        self.assertEqual(result, {})

    def test_empty_dataset_returns_empty(self):
        result = self.ns["evaluate_policy"](
            None, None, [], 128, "<fact>", "<answer>",
        )
        self.assertEqual(result, {})


class TestToolUsageStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_counts_count_and_sentinel(self):
        stats = self.ns["_tool_usage_stats"]([
            "reason <count> <A> <b>\n= 3\n</count> then <fact> <A> <r> <B> . </fact>",
            "<fact> <A> <r> <no further records> . </fact>",
            "no tool use here",
        ])
        self.assertEqual(stats["count_uses"], 1)
        self.assertEqual(stats["count_examples"], 1)
        self.assertEqual(stats["sentinel_uses"], 1)
        self.assertEqual(stats["sentinel_examples"], 1)

    def test_multiple_occurrences_in_one_completion(self):
        stats = self.ns["_tool_usage_stats"]([
            "<count> <A> <b></count> ... <count> <C> <d></count>",
        ])
        self.assertEqual(stats["count_uses"], 2)
        self.assertEqual(stats["count_examples"], 1)

    def test_accumulate_sums_across_batches(self):
        ns = self.ns
        counter = {}
        ns["_accumulate_tool_usage"](counter, ["<count> <A> <b></count>"])
        ns["_accumulate_tool_usage"](counter, ["<count> <C> <d></count>", "plain"])
        self.assertEqual(counter["count_uses"], 2)
        self.assertEqual(counter["count_examples"], 2)
        self.assertEqual(counter["sentinel_uses"], 0)

    def test_case_insensitive_sentinel(self):
        stats = self.ns["_tool_usage_stats"](["<No Further Records>"])
        self.assertEqual(stats["sentinel_uses"], 1)

    def test_no_prompt_contamination(self):
        # A completion that merely lacks markers must not register usage, even
        # though the real prompt contains the marker text.
        stats = self.ns["_tool_usage_stats"](["I have enough facts to answer."])
        self.assertEqual(stats["count_uses"], 0)
        self.assertEqual(stats["sentinel_uses"], 0)


if __name__ == "__main__":
    unittest.main()
