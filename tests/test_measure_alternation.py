import unittest

from utils.measure_alternation import measure_completion


class TestMeasureAlternation(unittest.TestCase):
    def test_alternating_reasoning_between_calls(self):
        text = (
            "I need to find the director first.\n\n"
            "<fact> <A> <director> <B> . </fact>\n"
            "B directed it, so now I need B's birth date to answer.\n\n"
            "<fact> <B> <date of birth> <1970-01-01> . </fact>\n"
            "I have enough evidence now.\n\n"
            "<answer>\n[\"1970\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertEqual(result["tool_blocks"], 2)
        self.assertEqual(result["free_spans"], 1)
        self.assertGreaterEqual(result["min_free_words"], 4)

    def test_adjacent_calls_have_no_reasoning(self):
        text = (
            "Looking up facts now.\n"
            "<fact> <A> <r> <B> . </fact>\n"
            "<fact> <A> <r> <C> . </fact>\n"
            "<answer>\n[\"B\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertEqual(result["tool_blocks"], 2)
        self.assertEqual(result["min_free_words"], 0)

    def test_answer_block_is_excluded_from_reasoning(self):
        text = (
            "<fact> <A> <r> <B> . </fact>\n"
            "<answer>\n[\"B\"]\n</answer>"
        )
        result = measure_completion(text)
        # Only one tool call, so there is no between-call span to measure.
        self.assertEqual(result["tool_blocks"], 1)
        self.assertEqual(result["free_spans"], 0)
        self.assertEqual(result["min_free_words"], 0)

    def test_count_block_is_treated_as_a_tool_call(self):
        text = (
            "<fact> <A> <r> <B> . </fact>\n"
            "Let me count how many children there are for completeness.\n"
            "<count> <A> <child>\n</count>\n"
            "The count confirms the answer.\n"
            "<answer>\n[\"3\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertEqual(result["tool_blocks"], 2)
        self.assertGreaterEqual(result["min_free_words"], 4)

    def test_tool_call_counts_by_kind(self):
        text = (
            "Lead in words here.\n"
            "<fact> <A> <r> <B> . </fact>\n"
            "Some reasoning sentence goes here now.\n"
            "<count> <A> <child>\n</count>\n"
            "<fact> <A> <r> <C> . </fact>\n"
            "Trailing sentence before the answer.\n"
            "<answer>\n[\"B\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertEqual(result["fact_calls"], 2)
        self.assertEqual(result["count_calls"], 1)
        self.assertEqual(result["sentinel_calls"], 0)
        self.assertGreater(result["constrained_words"], 0)

    def test_prefix_and_suffix_are_reported_separately(self):
        text = (
            "Prefix reasoning words before the first call.\n"
            "<fact> <A> <r> <B> . </fact>\n"
            "Between call reasoning words here now.\n"
            "<fact> <A> <r> <C> . </fact>\n"
            "Suffix reasoning words right before answer.\n"
            "<answer>\n[\"B\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertGreater(result["prefix_words"], 0)
        self.assertGreater(result["suffix_words"], 0)
        self.assertGreater(result["free_words"], 0)

    def test_sentinel_counts_as_a_tool_call(self):
        text = (
            "<fact> <A> <r> <B> . </fact>\n"
            "The relation looks exhausted now, so let me verify.\n"
            "<fact> <A> <r> <no further records> . </fact>\n"
            "<answer>\n[\"B\"]\n</answer>"
        )
        result = measure_completion(text)
        self.assertEqual(result["tool_blocks"], 2)


if __name__ == "__main__":
    unittest.main()
