"""Verification tests using real Mintaka examples from the evaluation logs.

Expected values below are taken from
``logs/22_sept_2026/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0``
and the large baseline logs.
"""

import unittest

from utils.answer_normalization import (
    canonical_value,
    compare_answers,
    verify_mintaka,
)


class TestVerifyMintakaRealExamples(unittest.TestCase):
    def assert_normalized(self, prediction, reference, question, rule=None):
        result = verify_mintaka(prediction, reference, question=question)
        self.assertTrue(result.normalized_match, f"{prediction!r} vs {reference!r}")
        if rule is not None:
            self.assertEqual(result.rule, rule)
        return result

    def assert_rejected(self, prediction, reference, question):
        result = verify_mintaka(prediction, reference, question=question)
        self.assertFalse(result.normalized_match, f"{prediction!r} vs {reference!r}")
        return result

    # -- dates -------------------------------------------------------------

    def test_year_reference_vs_full_date(self):
        # "When was the first Donkey Kong arcade game released?" gt=1981
        self.assert_normalized(
            ["July 9, 1981"], "1981",
            "When was the first Donkey Kong arcade game released?", rule="date",
        )

    def test_mintaka_abbreviated_gold_dates(self):
        cases = [
            (["August 2, 1970"], "2-Aug-70", "When was the director of Chasing Amy born?"),
            (["November 17, 1942"], "17-Nov-42", "When was the director of Taxi Driver born?"),
            (["August 24, 1988"], "24-Aug-88", "When was the actor who played Ron Weasley born?"),
            (["August 16, 1954"], "16-Aug-54", "When was the director of Titanic born?"),
            (["July 23, 1968"], "23-Jul-68", "When was the director of Free Guy born?"),
        ]
        for prediction, reference, question in cases:
            with self.subTest(reference=reference):
                self.assert_normalized(prediction, reference, question, rule="date")

    def test_two_digit_year_is_not_assumed_to_be_20xx(self):
        # gt=4-Oct-23 refers to 1923 (Charlton Heston); the century is encoded
        # in the full date the model produced, so modulo comparison must hold.
        self.assert_normalized(
            ["October 4, 1923"], "4-Oct-23",
            "When was the lead actor of the original Planet of the Apes born?",
            rule="date",
        )

    def test_wrong_full_date_is_rejected(self):
        self.assert_rejected(
            ["August 3, 1970"], "2-Aug-70", "When was the director of Chasing Amy born?",
        )

    def test_year_only_prediction_is_partial_for_full_date_reference(self):
        result = verify_mintaka(
            ["1970"], "2-Aug-70", question="When was the director of Chasing Amy born?"
        )
        self.assertFalse(result.normalized_match)
        self.assertTrue(result.partial_match)
        self.assertEqual(result.rule, "date_precision")

    # -- quantities --------------------------------------------------------

    def test_height_cm_vs_feet_inches(self):
        self.assert_normalized(
            ["170 cm"], "5'7\"",
            "How tall is the actor who played Jerry Maguire in Jerry Maguire?",
            rule="quantity",
        )

    def test_height_metres_vs_feet_inches(self):
        self.assert_normalized(
            ["1.65 m"], "5'5\"",
            "How tall is the actor who played the title role in the Harry Potter movies?",
            rule="quantity",
        )

    def test_different_heights_are_rejected(self):
        self.assert_rejected(
            ["1.80 m"], "5'5\"",
            "How tall is the actor who played the title role in the Harry Potter movies?",
        )

    # -- entities and aliases ---------------------------------------------

    def test_country_alias_with_trailing_period(self):
        # gt='USA.' (note the period) vs prediction 'United States'
        self.assert_normalized(
            ["United States"], "USA.",
            "Where is the actor who played Pirates of the Caribbean from?",
            rule="alias",
        )

    def test_accent_and_spelling_variants(self):
        self.assertTrue(
            compare_answers(["Aarhus"], ["\u00c5rhus"], dataset="mintaka").normalized_match
        )

    def test_answer_entity_label_variant(self):
        result = verify_mintaka(
            ["Monsters Inc"], "Monsters, Inc.",
            question="Which film was released?",
            input_sample={"answerEntity": [{"name": "Q187726", "label": "Monsters, Inc."}]},
        )
        self.assertTrue(result.normalized_match)
        self.assertEqual(result.rule, "mintaka_answer_entity_label")

    def test_dropped_qualifier_is_partial_not_correct(self):
        # The linked entity is 'Chester' but the reference keeps the country.
        for prediction, reference, question in [
            (["Chester"], "Chester, United Kingdom",
             "Where is the actor who played James Bond in Skyfall from?"),
            (["Vienna"], "Vienna, Austria",
             "Where was the director of the film born?"),
        ]:
            with self.subTest(reference=reference):
                result = verify_mintaka(
                    prediction, reference, question=question,
                    input_sample={"answerEntity": [{"name": "Q1", "label": prediction[0]}]},
                )
                self.assertFalse(result.normalized_match)
                self.assertTrue(result.partial_match)
                self.assertEqual(result.rule, "granularity")

    def test_punctuation_variant_keeps_full_match(self):
        # Same entity, no qualifier dropped: the granularity guard must not fire.
        result = verify_mintaka(
            ["Monsters, Inc."], "Monsters Inc",
            question="Which film was released?",
            input_sample={"answerEntity": [{"name": "Q187726", "label": "Monsters Inc"}]},
        )
        self.assertTrue(result.normalized_match)
        self.assertFalse(result.partial_match)
        self.assertEqual(result.rule, "mintaka_answer_entity_label")

    def test_identical_answer_is_a_strict_match(self):
        result = verify_mintaka(["Monsters Inc"], "Monsters Inc", question="Which film?")
        self.assertTrue(result.strict_match)
        self.assertTrue(result.normalized_match)

    def test_strict_match_mirrors_evaluator_semantics(self):
        # The official evaluator lowercases but does not strip punctuation, so
        # "USA" vs "USA." is a normalisation gain, not a strict match.
        result = verify_mintaka(["USA"], "USA.", question="Where is he from?")
        self.assertFalse(result.strict_match)
        self.assertTrue(result.normalized_match)
        # Trailing punctuation is handled by text normalisation, so this is a
        # 'text' gain rather than an alias lookup.
        self.assertEqual(result.rule, "text")

    # -- genuine errors must stay errors ----------------------------------

    def test_free_text_justification_is_not_an_answer(self):
        # The model hedged with a justification; gt is a single entity.
        self.assert_rejected(
            ["They are the same height (180 cm)"], "Brad Pitt",
            "Who is taller, George Clooney or Brad Pitt?",
        )

    def test_abstention_is_not_correct(self):
        self.assert_rejected(
            ["I don't know."], "No", "Is Ben Platt taller than Lin-Manuel Miranda?",
        )

    def test_count_mismatch_stays_wrong(self):
        self.assert_rejected(
            ["2"], "3", "How many kids does the lead actress of Pulp Fiction have?",
        )

    def test_numeric_substring_is_not_a_match(self):
        self.assert_rejected(["14"], "4", "How many films did he direct?")

    # -- list handling -----------------------------------------------------

    def test_multi_answer_alias_matching(self):
        result = compare_answers(
            ["United States", "Canada"],
            ["USA", "Canada"],
            dataset="mintaka",
            question="Which countries?",
        )
        self.assertTrue(result.normalized_match)

    def test_extra_answer_is_rejected(self):
        result = compare_answers(
            ["United States", "Canada", "Mexico"],
            ["USA", "Canada"],
            dataset="mintaka",
            question="Which countries?",
        )
        self.assertFalse(result.normalized_match)
        self.assertEqual(result.rule, "cardinality")

    # -- canonical reporting forms ----------------------------------------

    def test_canonical_date_forms(self):
        self.assertEqual(canonical_value("2-Aug-70"), "2-Aug-70")
        self.assertEqual(canonical_value("August 2, 1970"), "2-Aug-1970")
        self.assertEqual(canonical_value("1981"), "1981")

    def test_canonical_quantity_is_in_canonical_unit(self):
        self.assertEqual(canonical_value("170 cm"), "170.00 cm")
        self.assertEqual(canonical_value("1.65 m"), "165.00 cm")
        self.assertEqual(canonical_value("5'7\""), "170.18 cm")

    def test_canonical_alias_resolution(self):
        self.assertEqual(canonical_value("USA."), "united states")


if __name__ == "__main__":
    unittest.main()
