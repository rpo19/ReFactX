"""Offline tests for the compiled Vadalog grammar (Phase 1).

No model is needed: the grammar is compiled once and exercised via
``GrammarMatcher.accept_string`` (string level) and ``accept_token`` +
``fill_next_token_bitmask`` (token level).
"""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.vadalog_grammar import VadalogGrammar, build_ebnf, content_ebnf

OPEN = "<vadalog>"
CLOSE = "</vadalog>"

VALID = [
    " path(X, Y) :- edge(X, Y). ",
    "\nultimate_control(X, Z) :- control(X, Y), control(Y, Z).\n",
    " c(X) :- own(X, \"Acme Corp\"). ",
    " c(X) :- ~own(X, Y). ",
    " p(X) :- q(X). p(X) :- r(X). ",
    " related(X, Y) :- edge(X, Y). related(X, Y) :- edge(X, Z), related(Z, Y). ",
]

INVALID = [
    " path(X Y) :- edge(X, Y). ",      # missing comma
    " path(X, Y) :- . ",               # empty body
    " path(X, Y) :- edge(X, Y) ",      # missing final dot
    " 123(X) :- edge(X, Y). ",         # bad predicate name
]


class TestVadalogGrammar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)
        cls.grammar = VadalogGrammar(cls.tokenizer, OPEN, CLOSE)

    def _accepts(self, program):
        matcher = self.grammar.new_matcher()
        ok = matcher.accept_string(program + CLOSE)
        return bool(ok and matcher.is_completed())

    def test_ebnf_assembly(self):
        self.assertIn("program", content_ebnf())
        self.assertTrue(build_ebnf(OPEN, CLOSE).startswith(f'root ::= "{OPEN}"'))

    def test_accepts_valid_programs(self):
        for program in VALID:
            self.assertTrue(self._accepts(program), f"rejected valid: {program!r}")

    def test_rejects_invalid_programs(self):
        for program in INVALID:
            self.assertFalse(self._accepts(program), f"accepted invalid: {program!r}")

    def test_token_level_allows_a_valid_program(self):
        for program in VALID:
            matcher = self.grammar.new_matcher()
            bitmask = self.grammar.allocate_bitmask()
            mask = torch.full((1, self.vocab), -math.inf)
            for token in self.tokenizer.encode(program + CLOSE, add_special_tokens=False):
                self.grammar.apply_mask(mask, 0, matcher, bitmask)
                allowed = set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())
                self.assertIn(token, allowed,
                              f"token {token!r}={self.tokenizer.decode([token])!r} "
                              f"not offered while feeding {program!r}")
                self.assertTrue(matcher.accept_token(token))
            self.assertTrue(matcher.is_completed())

    def test_not_completed_before_closing_tag(self):
        matcher = self.grammar.new_matcher()
        self.assertTrue(matcher.accept_string(" path(X, Y) :- edge(X, Y). "))
        self.assertFalse(matcher.is_completed())  # still owes the </vadalog>


if __name__ == "__main__":
    unittest.main()
