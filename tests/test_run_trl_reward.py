import json
import unittest
from pathlib import Path


def _load_module():
    """Load the pure reward helpers from run_trl.py without importing trl/torch."""
    path = Path(__file__).resolve().parents[1] / "utils" / "run_trl.py"
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def _text("))
    end = next(i for i, l in enumerate(lines) if l.startswith("class CustomMetricsCallback"))
    snippet = "\n".join(lines[start:end])
    namespace = {}
    exec(
        "from typing import Any, Iterable\nimport json, re, os\n" + snippet,
        namespace,
    )
    return namespace


def _completion(answer, facts=0, extra="", count=None, sentinel=False):
    parts = []
    if count is not None:
        parts.append(f"<count> <A> <b>\n= {count}\n</count>")
    parts.extend(["<fact> <A> <r> <B> . </fact>"] * facts)
    parts.append(extra)
    parts.append(f"<answer>{json.dumps(answer)}</answer>")
    return "\n".join(parts)


class TestExactRewardComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()
        cls.reward = staticmethod(cls.ns["exact_reward"])

    def reward_one(self, completion, reference, **kwargs):
        return self.ns["exact_reward"]([completion], [reference], **kwargs)[0]

    def test_correct_answer_scores_full(self):
        score = self.reward_one(_completion(["Paris"], facts=1), ["Paris"])
        # 0.3 fact + 0.3 answer block + 0.5 JSON + 1.0 answer credit.
        self.assertAlmostEqual(score, 2.1)

    def test_malformed_json_keeps_only_structure(self):
        completion = "<fact> <A> <r> <B> . </fact>\n<answer>not json</answer>"
        score = self.reward_one(completion, ["Paris"])
        self.assertAlmostEqual(score, 0.6)

    def test_partial_answer_gets_iou_credit(self):
        score = self.reward_one(_completion(["Paris", "Lyon"], facts=1), ["Paris"])
        self.assertAlmostEqual(score, 0.6 + 0.5 + 0.5)


class TestFactCountPenalty(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def reward_one(self, completion, reference, **kwargs):
        return self.ns["exact_reward"]([completion], [reference], **kwargs)[0]

    def test_penalty_applies_beyond_soft_cap(self):
        score = self.reward_one(
            _completion(["Paris"], facts=30), ["Paris"],
            fact_soft_cap=20, fact_penalty=0.01, max_fact_penalty=0.2,
        )
        # 10 extra fact calls * 0.01 = 0.1 penalty.
        self.assertAlmostEqual(score, 2.1 - 0.1)

    def test_penalty_is_bounded(self):
        score = self.reward_one(
            _completion(["Paris"], facts=100), ["Paris"],
            fact_soft_cap=20, fact_penalty=0.01, max_fact_penalty=0.2,
        )
        self.assertAlmostEqual(score, 2.1 - 0.2)

    def test_no_penalty_under_cap(self):
        score = self.reward_one(_completion(["Paris"], facts=5), ["Paris"])
        self.assertAlmostEqual(score, 2.1)


class TestNormalizedCredit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def reward_one(self, completion, reference, **kwargs):
        return self.ns["exact_reward"]([completion], [reference], **kwargs)[0]

    def test_semantically_equal_date_gets_full_credit(self):
        completion = _completion(["August 2, 1970"], facts=1)
        score = self.reward_one(completion, ["2-Aug-70"], dataset="mintaka")
        self.assertAlmostEqual(score, 2.1)

    def test_normalization_can_be_disabled(self):
        completion = _completion(["August 2, 1970"], facts=1)
        score = self.reward_one(
            completion, ["2-Aug-70"], dataset="mintaka", use_normalized=False
        )
        # Strict comparison misses the date, so only structure + JSON remain.
        self.assertAlmostEqual(score, 1.1)


class TestToolGating(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def reward_one(self, completion, reference, **kwargs):
        return self.ns["exact_reward"]([completion], [reference], **kwargs)[0]

    def test_count_multiplier_paid_on_correct_count_question(self):
        completion = _completion(["8"], facts=1, count=8)
        score = self.reward_one(
            completion, ["8"], question_type=["count"], alpha=0.1, beta=0.0
        )
        self.assertAlmostEqual(score, 2.1 * 1.1)

    def test_count_multiplier_uses_question_text_when_no_type(self):
        completion = _completion(["8"], facts=1, count=8)
        score = self.reward_one(
            completion,
            ["8"],
            question=["How many countries border France?"],
            alpha=0.1,
            beta=0.0,
        )
        self.assertAlmostEqual(score, 2.1 * 1.1)

    def test_count_multiplier_withheld_for_non_count_type(self):
        completion = _completion(["8"], facts=1, count=8)
        score = self.reward_one(
            completion, ["8"], question_type=["generic"], alpha=0.1, beta=0.0
        )
        self.assertAlmostEqual(score, 2.1)

    def test_count_multiplier_withheld_when_answer_wrong(self):
        completion = _completion(["9"], facts=1, count=8)
        score = self.reward_one(
            completion, ["8"], question_type=["count"], alpha=0.1, beta=0.0
        )
        # No correct-answer credit and no tool bonus.
        self.assertAlmostEqual(score, 1.1)

    def test_sentinel_multiplier_paid_when_required(self):
        completion = _completion(
            ["Paris"], facts=1, extra="<no further records>", sentinel=True
        )
        score = self.reward_one(
            completion, ["Paris"], requires_enumeration=[True], alpha=0.0, beta=0.05
        )
        self.assertAlmostEqual(score, 2.1 * 1.05)

    def test_sentinel_multiplier_withheld_when_not_required(self):
        completion = _completion(["Paris"], facts=1, extra="<no further records>")
        score = self.reward_one(
            completion, ["Paris"], requires_enumeration=[False], alpha=0.0, beta=0.05
        )
        self.assertAlmostEqual(score, 2.1)


class TestRewardStats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_stats_are_accumulated(self):
        stats = {}
        self.ns["exact_reward"](
            [_completion(["Paris"], facts=25), _completion(["Lyon"], facts=1)],
            [["Paris"], ["Paris"]],
            fact_soft_cap=20,
            fact_penalty=0.01,
            max_fact_penalty=0.2,
            stats=stats,
        )
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["correct"], 1)
        self.assertEqual(stats["fact_calls"], 26)
        # First example: 5 extra fact calls -> 0.05 penalty.
        self.assertAlmostEqual(stats["fact_penalty"], 0.05)
        self.assertAlmostEqual(stats["total"], (2.1 - 0.05) + 1.1)


class TestProcessorForwarding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_sentinel_and_count_pattern_are_forwarded(self):
        kwargs = self.ns["_processor_kwargs"]("<fact>", "<count>", True)
        self.assertIs(kwargs["sentinel"], True)
        self.assertEqual(kwargs["count_pattern"], "<count>")
        self.assertEqual(kwargs["fact_pattern"], "<fact>")

    def test_sentinel_can_be_disabled(self):
        kwargs = self.ns["_processor_kwargs"]("<fact>", "<count>", False)
        self.assertIs(kwargs["sentinel"], False)


class TestToolCounterGaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_gap_is_text_minus_decoder(self):
        gaps = self.ns["_tool_counter_gaps"](
            {"count_uses": 7, "sentinel_uses": 2},
            {"count_calls": 5, "sentinel_calls": 3},
        )
        self.assertEqual(gaps["count"], 2)
        self.assertEqual(gaps["sentinel"], -1)

    def test_missing_keys_default_to_zero(self):
        gaps = self.ns["_tool_counter_gaps"]({}, {})
        self.assertEqual(gaps, {"count": 0, "sentinel": 0})


class TestSchemaDerivations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_question_type_top_level_and_nested(self):
        derive = self.ns["_derive_question_type"]
        self.assertEqual(derive({"complexityType": "count"}), "count")
        self.assertEqual(derive({"type": "compositional"}), "compositional")
        self.assertEqual(
            derive({"input_sample": {"complexityType": "multihop"}}), "multihop"
        )
        self.assertIsNone(derive({}))

    def test_enumeration_only_for_configured_datasets(self):
        derive = self.ns["_derive_requires_enumeration"]
        self.assertTrue(derive(["a", "b"], "rmanluo/RoG-webqsp", ["webqsp"]))
        self.assertFalse(derive(["a"], "rmanluo/RoG-webqsp", ["webqsp"]))
        self.assertFalse(derive(["a", "b"], "rpozzi/mintaka", ["webqsp"]))
        self.assertFalse(derive(["a", "b"], "rmanluo/RoG-webqsp", []))


class TestGrpoConfigsSentinel(unittest.TestCase):
    def test_sentinel_is_true_where_declared(self):
        configs = Path(__file__).resolve().parents[1] / "configs"
        paths = sorted(configs.glob("*grpo*.json"))
        self.assertTrue(paths, "expected GRPO configs to exist")
        for path in paths:
            config = json.loads(path.read_text())
            if "sentinel" in config:
                self.assertIs(config["sentinel"], True, f"{path.name} disables sentinel")


class TestDetailsOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_details_capture_correctness_and_tool_use(self):
        details = []
        completion = _completion(["8"], facts=1, count=8)
        self.ns["exact_reward"](
            [completion],
            [["8"]],
            question_type=["count"],
            count_pattern="<count>",
            alpha=0.1,
            beta=0.0,
            details=details,
        )
        self.assertEqual(len(details), 1)
        info = details[0]
        self.assertTrue(info["correct"])
        self.assertTrue(info["is_count_question"])
        self.assertTrue(info["used_count"])
        self.assertTrue(info["answered"])
        self.assertFalse(info["dont_know"])
        self.assertEqual(info["question_type"], "count")
        self.assertEqual(info["fact_calls"], 1)
        self.assertAlmostEqual(info["answer_credit"], 1.0)

    def test_details_mark_dont_know(self):
        details = []
        completion = _completion(["I don't know."], facts=1)
        self.ns["exact_reward"]([completion], [["Paris"]], details=details)
        self.assertTrue(details[0]["dont_know"])
        self.assertFalse(details[0]["correct"])


class TestWindowHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def test_stats_delta(self):
        delta = self.ns["_stats_delta"](
            {"n": 5, "total": 10.0}, {"n": 2, "total": 4.0}
        )
        self.assertEqual(delta["n"], 3)
        self.assertAlmostEqual(delta["total"], 6.0)

    def test_percentile(self):
        perc = self.ns["_percentile"]
        self.assertAlmostEqual(perc([0, 10, 20, 30], 0.5), 15.0)
        self.assertAlmostEqual(perc([0, 10, 20, 30], 0.0), 0.0)
        self.assertAlmostEqual(perc([0, 10, 20, 30], 1.0), 30.0)
        self.assertEqual(perc([], 0.5), 0.0)


class TestGroupAndBehaviorMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = _load_module()

    def _detail(self, correct, used_count, **overrides):
        detail = {
            "correct": correct,
            "used_count": used_count,
            "used_sentinel": False,
            "answered": True,
            "dont_know": False,
            "normalized_match": False,
            "answer_credit": 1.0 if correct else 0.0,
            "structure": 0.6,
            "json": 0.5,
            "is_count_question": False,
        }
        detail.update(overrides)
        return detail

    def test_group_diagnostics(self):
        stats = {}
        details = [
            self._detail(False, False),
            self._detail(True, True),
            self._detail(False, False),
            self._detail(False, False),
        ]
        self.ns["_accumulate_group_metrics"]([1.0, 2.0, 0.0, 0.0], details, 2, stats)
        metrics = self.ns["_group_metrics"](stats)
        self.assertEqual(metrics["train/group/groups"], 2)
        self.assertAlmostEqual(metrics["train/group/zero_advantage_frac"], 0.5)
        self.assertAlmostEqual(metrics["train/group/reward_std"], 0.25)
        self.assertAlmostEqual(metrics["train/group/all_wrong_frac"], 0.5)
        self.assertAlmostEqual(metrics["train/group/all_correct_frac"], 0.0)
        self.assertAlmostEqual(metrics["train/group/pass_at_k"], 0.5)
        self.assertAlmostEqual(metrics["train/group/tool_diverse_frac"], 0.5)

    def test_behavior_and_conditional_tool_metrics(self):
        stats = {}
        details = [
            self._detail(True, True, is_count_question=True),
            self._detail(False, False, is_count_question=True),
        ]
        self.ns["_accumulate_group_metrics"]([2.1, 1.1], details, 2, stats)
        metrics = self.ns["_behavior_metrics"](stats)
        self.assertAlmostEqual(metrics["train/tool/count_type_rate"], 1.0)
        self.assertAlmostEqual(metrics["train/tool/count_usage_on_count_type"], 0.5)
        self.assertAlmostEqual(metrics["train/tool/count_accuracy_used"], 1.0)
        self.assertAlmostEqual(metrics["train/tool/count_accuracy_unused"], 0.0)
        self.assertAlmostEqual(metrics["train/window/confident_wrong_rate"], 0.5)
        self.assertAlmostEqual(metrics["train/window/format_saturation"], 1.0)
        self.assertAlmostEqual(metrics["train/window/answer_vs_correct_gap"], 0.0)

    def test_tool_counter_metrics_include_gaps(self):
        metrics = self.ns["_tool_counter_metrics"](
            {"count_uses": 3, "sentinel_uses": 1},
            {"count_calls": 2, "sentinel_calls": 1},
        )
        self.assertEqual(metrics["train/tool/count_uses"], 3)
        self.assertEqual(metrics["train/tool/count_calls"], 2)
        self.assertEqual(metrics["train/tool/count_text_decoder_gap"], 1)
        self.assertEqual(metrics["train/tool/sentinel_text_decoder_gap"], 0)


if __name__ == "__main__":
    unittest.main()
