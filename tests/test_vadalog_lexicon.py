"""Tests for lexicon grounding of <vadalog> (Phase 2 / Challenge 2a)."""
import unittest

import torch
import xgrammar as xgr
from transformers import AutoTokenizer

from refactx.vadalog_lexicon import (
    build_grounded_ebnf,
    collect_relations,
    render_content_ebnf,
)

OPEN = "<vadalog>"
CLOSE = "</vadalog>"

LEXICON = dict(
    body_relations=["own", "is_own", "control", "is_ultimate_control"],
    head_relations=["path", "control", "ultimate_control"],
    variables=["X", "Y", "Z"],
    entities=["Acme Corp", "Globex"],
)

VALID = [
    " path(X, Y) :- own(X, Y). ",
    " control(X, Z) :- control(X, Y), control(Y, Z). ",
    " path(X) :- own(X, \"Acme Corp\"). ",
    " ultimate_control(X, Z) :- control(X, Y), is_own(Y, Z). ",
]

INVALID = [
    " path(X) :- foo(X). ",                 # unknown body predicate
    " path(X) :- own(X, \"Unknown\"). ",      # entity not in lexicon
    " path(A) :- own(A, X). ",              # A is not a variable
    " own(X, Y). ",                         # EDB predicate as head
]


class TestRender(unittest.TestCase):

    def test_terminals_substituted(self):
        ebnf = render_content_ebnf(**LEXICON)
        self.assertIn('"own"', ebnf)          # body predicate
        self.assertIn('"path"', ebnf)         # head predicate
        self.assertIn('"X" | "Y" | "Z"', ebnf)  # variables
        self.assertIn('"Acme Corp"', ebnf)    # entity
        self.assertNotIn("[a-z] [a-zA-Z0-9_]*", ebnf)

    def test_generic_when_unset(self):
        ebnf = render_content_ebnf()
        self.assertIn("[a-z] [a-zA-Z0-9_]*", ebnf)
        self.assertIn("[A-Z] [A-Za-z0-9_]*", ebnf)

    def test_entity_cap(self):
        with self.assertRaises(ValueError):
            render_content_ebnf(entities=[f"e{i}" for i in range(10)], max_entities=5)


class TestCollectRelations(unittest.TestCase):

    def test_union(self):
        class FakeApi:
            def entity_index(self):
                return ["a", "b"]

            def property_strings(self, entity):
                return {"a": ["own", "control"], "b": ["own", "is_own"]}[entity]

        self.assertEqual(collect_relations(FakeApi()), ["control", "is_own", "own"])
        self.assertEqual(collect_relations(FakeApi(), sample=1), ["control", "own"])


class TestGroundedGrammar(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        info = xgr.TokenizerInfo.from_huggingface(cls.tokenizer)
        ebnf = build_grounded_ebnf(OPEN, CLOSE, **LEXICON)
        cls.compiled = xgr.GrammarCompiler(info).compile_grammar(ebnf)

    def _accepts(self, program):
        matcher = xgr.GrammarMatcher(self.compiled)
        matcher.accept_string(OPEN)
        ok = matcher.accept_string(program + CLOSE)
        return bool(ok and matcher.is_completed())

    def test_accepts_valid(self):
        for program in VALID:
            self.assertTrue(self._accepts(program), f"rejected valid: {program!r}")

    def test_rejects_invalid(self):
        for program in INVALID:
            self.assertFalse(self._accepts(program), f"accepted invalid: {program!r}")


class TestGenerationWithLexicon(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def test_lexicon_constrains_and_completes(self):
        from refactx.generate import PatternConstrainedState
        from refactx.index import DictIndex
        from refactx.vadalog import VadalogGeneration

        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        gen = VadalogGeneration(state, self.tokenizer, 0, lexicon=LEXICON)
        seq = []
        tokens = self._encode(" path(X, Y) :- own(X, Y). ") + self._encode(CLOSE)
        for token in tokens:
            mask = torch.full((1, self.vocab), float("-inf"))
            gen.constrain(seq, mask, 0)
            allowed = set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())
            self.assertIn(token, allowed)
            # a lexicon-restricted predicate position is much smaller than the vocab
            self.assertLess(len(allowed), self.vocab)
            seq.append(token)
        gen.constrain(seq, torch.full((1, self.vocab), float("-inf")), 0)
        self.assertTrue(gen.done)
        self.assertIn("own(X, Y)", state.vadalog_blocks[0]["code"])


if __name__ == "__main__":
    unittest.main()
