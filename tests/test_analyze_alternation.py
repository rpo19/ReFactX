import unittest

from utils.analyze_alternation import abstention_profile, run_lengths


class TestRunLengths(unittest.TestCase):
    def test_no_gaps_is_a_single_run(self):
        self.assertEqual(run_lengths([]), 1)

    def test_adjacent_calls_form_one_run(self):
        # Three fact calls, zero words between each pair.
        self.assertEqual(run_lengths([0, 0], gap_threshold=1), 3)

    def test_reasoning_breaks_the_run(self):
        # fact, 10 words, fact, fact  -> longest run is 2
        self.assertEqual(run_lengths([10, 0], gap_threshold=1), 2)

    def test_all_separated(self):
        self.assertEqual(run_lengths([5, 5, 5], gap_threshold=1), 1)

    def test_threshold_boundary(self):
        # Gap of exactly the threshold does not join runs.
        self.assertEqual(run_lengths([1, 1], gap_threshold=1), 1)
        self.assertEqual(run_lengths([1, 1], gap_threshold=2), 3)


class TestAbstentionProfile(unittest.TestCase):
    def _row(self, prediction, correct, triples=None):
        return {
            "prediction": prediction,
            "evaluation": {"correct": correct},
            "triples": triples if triples is not None else [["t"]],
        }

    def test_counts_abstention_and_confident_wrong(self):
        rows = [
            self._row(["I don't know."], 0),
            self._row(["Paris"], 1),
            self._row(["Berlin"], 0),
            self._row(None, 0),
        ]
        profile = abstention_profile(rows)
        self.assertAlmostEqual(profile["abstain_rate"], 0.25)
        self.assertEqual(profile["abstain_n"], 1)
        self.assertAlmostEqual(profile["answered_rate"], 0.75)
        # Two of three answered are not abstention; one is wrong.
        self.assertAlmostEqual(profile["confident_wrong_rate"], 0.25)
        self.assertEqual(profile["confident_wrong_n"], 1)
        self.assertAlmostEqual(profile["empty_rate"], 0.25)

    def test_answered_with_zero_triples(self):
        rows = [
            self._row(["Paris"], 1, triples=[]),
            self._row(["Lyon"], 0, triples=[]),
        ]
        profile = abstention_profile(rows)
        self.assertEqual(profile["answered_zero_triples_n"], 2)
        self.assertAlmostEqual(profile["answered_zero_triples_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
