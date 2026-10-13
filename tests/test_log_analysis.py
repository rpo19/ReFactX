"""Tests for utils/log_analysis.py — triple extraction and error analysis."""
import unittest

from utils.log_analysis import (
    extract_triples,
    hallucinated_triples,
    normalize_triple,
    record_triples,
    text_triples,
    triple_diff,
    records_frame,
    analyze,
)


def _record(full_prediction, triples, **kw):
    rec = {
        "full_prediction": full_prediction,
        "triples": triples,
        "evaluation": {"precision": 1, "recall": 1, "f1": 1, "correct": 1, "dont_know": 0},
        "prediction": kw.pop("prediction", ["yes"]),
        "question": kw.pop("question", "q"),
        "kind": kw.pop("kind", "ask"),
        "gt_answer": kw.pop("gt_answer", ["yes"]),
    }
    rec.update(kw)
    return rec


class TestTripleExtraction(unittest.TestCase):

    def test_basic_triple(self):
        text = 'reasoning <kg> <Alice> <owns> <Bob> . </kg> done'
        self.assertEqual(extract_triples(text), ['<Alice> <owns> <Bob> .'])

    def test_object_with_spaces(self):
        text = '<kg> <Alice> <owns> <no further records> . </kg>'
        self.assertEqual(extract_triples(text), ['<Alice> <owns> <no further records> .'])

    def test_skips_kg_tag(self):
        # The <kg> tag itself must not be counted as a triple.
        text = '<kg> <A> <r> <B> . </kg>'
        self.assertEqual(len(extract_triples(text)), 1)

    def test_incomplete_triple_not_matched(self):
        self.assertEqual(extract_triples('<A> <r> <B>'), [])

    def test_normalize_strips_period_and_spaces(self):
        self.assertEqual(normalize_triple(' <A>  <r>   <B> . '), '<A> <r> <B>')


class TestRecordTriples(unittest.TestCase):

    def test_strips_kg_suffix_from_key_triples(self):
        rec = _record('', [' <A> <r> <B> . </kg>'])
        self.assertEqual(record_triples(rec), ['<A> <r> <B>'])

    def test_text_and_key_agree(self):
        text = '<kg> <A> <r> <B> . </kg>'
        rec = _record(text, [' <A> <r> <B> . </kg>'])
        self.assertEqual(text_triples(rec), record_triples(rec))


class TestTripleDiff(unittest.TestCase):

    def test_no_diff_when_consistent(self):
        text = '<kg> <A> <r> <B> . </kg><kg> <C> <r> <D> . </kg>'
        rec = _record(text, [' <A> <r> <B> . </kg>', ' <C> <r> <D> . </kg>'])
        extra, missing = triple_diff(rec)
        self.assertEqual(sum(extra.values()), 0)
        self.assertEqual(sum(missing.values()), 0)

    def test_detects_hallucinated_triple(self):
        # A triple in free text that the constrained generator never made.
        text = ('<kg> <A> <r> <B> . </kg>'
                ' but actually <Eve> <owns> <Mallory> .')
        rec = _record(text, [' <A> <r> <B> . </kg>'])
        extra, missing = triple_diff(rec)
        self.assertEqual(sum(extra.values()), 1)
        self.assertEqual(sum(missing.values()), 0)
        self.assertIn('<Eve> <owns> <Mallory>', extra)
        self.assertEqual(hallucinated_triples(rec), ['<Eve> <owns> <Mallory>'])

    def test_requoted_triple_is_not_a_hallucination(self):
        # The model restates a generated triple: count mismatch, but not a
        # hallucination, because the triple *was* generated.
        text = '<kg> <A> <r> <B> . </kg> I got <A> <r> <B> .'
        rec = _record(text, [' <A> <r> <B> . </kg>'])
        extra, _ = triple_diff(rec)
        self.assertEqual(sum(extra.values()), 1)
        self.assertEqual(hallucinated_triples(rec), [])

    def test_incomplete_triple_is_not_a_hallucination(self):
        # The last lookup was cut off before being recorded: ``incomplete_triples``
        # accounts for it, so it is not a hallucination.
        text = ('<kg> <A> <r> <B> . </kg>'
                '<kg> <Cut Off> <is_influence> <Some Group> . </kg>')
        rec = _record(text, [' <A> <r> <B> . </kg>'], incomplete_triples=1)
        self.assertEqual(hallucinated_triples(rec), [])

    def test_incomplete_only_discounts_the_last_lookup(self):
        # A genuine hallucination *before* the truncated lookup is still caught.
        text = ('<kg> <A> <r> <B> . </kg>'
                ' made up <X> <y> <Z> .'
                '<kg> <Cut Off> <is_influence> <Group> . </kg>')
        rec = _record(text, [' <A> <r> <B> . </kg>'], incomplete_triples=1)
        self.assertEqual(hallucinated_triples(rec), ['<X> <y> <Z>'])

    def test_unexplained_triple_is_a_hallucination(self):
        text = '<kg> <A> <r> <B> . </kg> and note <X> <y> <Z> .'
        rec = _record(text, [' <A> <r> <B> . </kg>'])
        self.assertEqual(hallucinated_triples(rec), ['<X> <y> <Z>'])

    def test_detects_missing_triple(self):
        rec = _record('', [' <A> <r> <B> . </kg>'])
        extra, missing = triple_diff(rec)
        self.assertEqual(sum(extra.values()), 0)
        self.assertEqual(sum(missing.values()), 1)


class TestRecordsFrame(unittest.TestCase):

    def test_columns_and_hit_max_estimate(self):
        recs = [
            _record('<kg> <A> <r> <B> . </kg>', [' <A> <r> <B> . </kg>']),
            _record('<kg> <A> <r> <B> . </kg> but <X> <y> <Z> .',
                    [' <A> <r> <B> . </kg>']),
        ]
        df = records_frame(recs, metadata={"max_new_tokens": 512})
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["n_triples"]), [1, 1])
        self.assertEqual(list(df["n_triples_regex"]), [1, 2])
        self.assertEqual(list(df["triple_mismatch"]), [0, 1])
        self.assertEqual(list(df["extra_triples"]), [0, 1])
        self.assertEqual(list(df["unknown_triples"]), [0, 1])
        # No recorded token flag -> estimate path used.
        self.assertTrue(df["hit_max_is_estimate"].all())

    def test_recorded_hit_max_tokens_used(self):
        rec = _record('x', [], n_new_tokens=512, hit_max_tokens=1)
        df = records_frame([rec], metadata={"max_new_tokens": 512})
        self.assertEqual(int(df["hit_max_tokens"].iloc[0]), 1)
        self.assertFalse(bool(df["hit_max_is_estimate"].iloc[0]))

    def test_analyze_stats(self):
        recs = [
            _record('<kg> <A> <r> <B> . </kg>', [' <A> <r> <B> . </kg>'], kind="ask"),
            _record('<kg> <C> <r> <no further records> . </kg>',
                    [' <C> <r> <no further records> . </kg>'], kind="count"),
        ]
        df, stats = analyze(recs, metadata={"max_new_tokens": 512})
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["sentinel_rate"], 0.5)
        self.assertEqual(stats["triple_mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
