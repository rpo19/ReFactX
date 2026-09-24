import unittest

from utils.answer_normalization import compare_answers


class TestAnswerNormalization(unittest.TestCase):
    def test_mintaka_date_formats_match(self):
        result = compare_answers(
            ["August 2, 1970"],
            ["2-Aug-70"],
            dataset="mintaka",
            question="When was the director born?",
        )
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "date")

    def test_year_reference_accepts_more_precise_date(self):
        result = compare_answers(
            ["July 9, 1981"],
            ["1981"],
            dataset="mintaka",
            question="In which year was the film released?",
        )
        self.assertTrue(result.normalized_match)
        self.assertFalse(result.partial_match)

    def test_full_date_reference_does_not_accept_year_as_full_match(self):
        result = compare_answers(
            ["1981"],
            ["July 9, 1981"],
            dataset="mintaka",
            question="When was the person born?",
        )
        self.assertFalse(result.normalized_match)
        self.assertTrue(result.partial_match)

    def test_2wiki_date_formats_match(self):
        result = compare_answers(
            ["June 12, 1516"],
            ["12 June 1516"],
            dataset="2wiki",
            question="When did he die?",
        )
        self.assertTrue(result.normalized_match)

    def test_height_conversion(self):
        result = compare_answers(
            ["170 cm"],
            ["5'7\""],
            dataset="mintaka",
            question="How tall is the actor?",
        )
        self.assertTrue(result.normalized_match)

    def test_numeric_substring_is_not_a_match(self):
        result = compare_answers(
            ["14"],
            ["4"],
            dataset="mintaka",
            question="How many films were released?",
        )
        self.assertFalse(result.normalized_match)

    def test_mintaka_entity_label_alias(self):
        result = compare_answers(
            ["Monsters Inc"],
            ["Monsters, Inc."],
            dataset="mintaka",
            question="Which film was released?",
            input_sample={
                "answerEntity": [{"name": "Q187726", "label": "Monsters, Inc."}]
            },
        )
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "mintaka_answer_entity_label")

    def test_mintaka_entity_alias_not_used_for_count(self):
        result = compare_answers(
            ["5"],
            ["Donald Trump Jr."],
            dataset="mintaka",
            question="How many children are there?",
            input_sample={
                "answerEntity": [{"name": "Q1", "label": "Donald Trump Jr."}]
            },
        )
        self.assertFalse(result.normalized_match)


if __name__ == "__main__":
    unittest.main()
