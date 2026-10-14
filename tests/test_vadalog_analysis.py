"""Tests for Vadalog symbol-table / safety analysis (Phase 3)."""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.vadalog_analysis import analyze, parse_program, tokenize


class TestParser(unittest.TestCase):

    def test_tokenize(self):
        kinds = [k for k, _ in tokenize('p(X, "a") :- q(X).')]
        self.assertEqual(kinds, ["ident", "punct", "var", "punct", "string",
                                 "punct", "impl", "ident", "punct", "var",
                                 "punct", "punct"])

    def test_parse_atoms(self):
        rules = parse_program("path(X, Y) :- edge(X, Y), ~blocked(Y).")
        self.assertEqual(len(rules), 1)
        r = rules[0]
        self.assertEqual(r["head"]["pred"], "path")
        self.assertTrue(r["has_body"])
        self.assertEqual([lit["pred"] for lit in r["body"]], ["edge", "blocked"])
        self.assertFalse(r["body"][0]["negated"])
        self.assertTrue(r["body"][1]["negated"])


class TestAnalyze(unittest.TestCase):

    def test_safe_transitive(self):
        a = analyze("control(X, Z) :- control(X, Y), control(Y, Z).")
        self.assertTrue(a["safe"])
        self.assertEqual(a["rules"][0]["head_vars"], ["X", "Z"])
        self.assertEqual(a["rules"][0]["body_only_vars"], ["Y"])

    def test_unsafe_head_variable(self):
        a = analyze("p(X, Y) :- q(X).")
        self.assertFalse(a["safe"])
        self.assertIn("Y", a["violations"][0])

    def test_fact_with_variable(self):
        a = analyze("p(X).")
        self.assertFalse(a["safe"])
        self.assertIn("variable", a["violations"][0])

    def test_ground_fact_is_safe(self):
        a = analyze('p("Acme Corp").')
        self.assertTrue(a["safe"])

    def test_arity_inconsistency(self):
        a = analyze("p(X) :- q(X). p(X, Y) :- q(X, Y).")
        self.assertFalse(a["safe"])
        self.assertTrue(any("arity" in v for v in a["violations"]))

    def test_multiple_rules(self):
        a = analyze("path(X, Y) :- edge(X, Y). path(X, Y) :- edge(X, Z), path(Z, Y).")
        self.assertEqual(a["n_rules"], 2)
        self.assertTrue(a["safe"])

    def test_partial_output_is_best_effort(self):
        # malformed / truncated must not raise
        for text in ["", "p(", "p(X) :- ", ":- q(X).", 'p("unterminated']:
            analyze(text)  # should not raise


class TestGenerationAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)

    def _run(self, program, **kw):
        from refactx.generate import PatternConstrainedState
        from refactx.index import DictIndex
        from refactx.vadalog import VadalogGeneration, VadalogEngine

        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        gen = VadalogGeneration(state, self.tokenizer, 0,
                                engine=VadalogEngine(runner=lambda code: "R"),
                                grammar=False, **kw)
        seq = self.tokenizer.encode(program, add_special_tokens=False) \
            + self.tokenizer.encode("</vadalog>", add_special_tokens=False)
        gen.constrain(seq, torch.full((1, self.vocab), float("-inf")), 0)
        self.assertTrue(gen.done)
        return state.vadalog_blocks[0]

    def test_block_records_analysis(self):
        block = self._run(" control(X, Z) :- control(X, Y), control(Y, Z). ")
        self.assertTrue(block["analysis"]["safe"])
        self.assertEqual(block["analysis"]["rules"][0]["body_only_vars"], ["Y"])
        self.assertEqual(block["result"], {"ok": True, "result": "R", "error": None})

    def test_require_safe_skips_execution(self):
        block = self._run(" p(X, Y) :- q(X). ", require_safe=True)
        self.assertFalse(block["analysis"]["safe"])
        self.assertFalse(block["result"]["ok"])
        self.assertIn("unsafe", block["result"]["error"])


if __name__ == "__main__":
    unittest.main()
