"""Phase 0 tests for the <vadalog> command plumbing."""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.generate import PatternConstrainedState
from refactx.index import DictIndex
from refactx.vadalog import VadalogEngine, VadalogGeneration


def _state(tok):
    return PatternConstrainedState(
        tokenizer=tok, cache_index=DictIndex(), subtree_cache=DictIndex())


class TestVadalogEngine(unittest.TestCase):

    def test_custom_runner(self):
        engine = VadalogEngine(runner=lambda code: ["fact"])
        self.assertTrue(engine.available)
        self.assertEqual(engine.run("p(X):-q(X)."),
                         {"ok": True, "result": ["fact"], "error": None})

    def test_failure_is_structured(self):
        engine = VadalogEngine(runner=lambda code: (_ for _ in ()).throw(ValueError("bad")))
        out = engine.run("p(")
        self.assertFalse(out["ok"])
        self.assertIn("bad", out["error"])


class TestVadalogGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _mask(self):
        return torch.full((1, self.vocab), -math.inf)

    def _allowed(self, mask):
        return (mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist()

    def _make(self, state, **kw):
        return VadalogGeneration(state, self.tokenizer, 0, **kw)

    def test_grammar_constrains_inside_block(self):
        state = _state(self.tokenizer)
        gen = self._make(state, engine=VadalogEngine(runner=lambda code: "ok"))
        seq = self._encode(" path(X, Y) :- edge(X, Y). ")
        mask = self._mask()
        gen.constrain(seq, mask, 0)
        allowed = set(self._allowed(mask))
        # constrained: not every token is allowed, but valid Vadalog tokens are
        self.assertLess(len(allowed), self.vocab)
        # a complete program: the closing tag may start, '#' may not appear
        self.assertIn(self._encode("<")[0], allowed)
        self.assertNotIn(self._encode("#")[0], allowed)
        self.assertFalse(gen.done)

    def test_grammar_drives_a_valid_program_to_completion(self):
        state = _state(self.tokenizer)
        gen = self._make(state, engine=VadalogEngine(runner=lambda code: "R"))
        seq = []
        tokens = self._encode(" path(X, Y) :- edge(X, Y). ") + self._encode("</vadalog>")
        for token in tokens:
            mask = self._mask()
            gen.constrain(seq, mask, 0)
            self.assertIn(token, self._allowed(mask))
            seq.append(token)
        gen.constrain(seq, self._mask(), 0)  # close tag fed -> completes
        self.assertTrue(gen.done)
        self.assertEqual(len(state.vadalog_blocks), 1)
        self.assertIn("edge(X, Y)", state.vadalog_blocks[0]["code"])
        self.assertNotIn("</vadalog>", state.vadalog_blocks[0]["code"])

    def test_unconstrained_when_grammar_disabled(self):
        state = _state(self.tokenizer)
        gen = self._make(state, grammar=False, engine=VadalogEngine(runner=lambda code: "ok"))
        seq = self._encode(" path(X, Y) :- edge(X, Y). ")
        mask = self._mask()
        gen.constrain(seq, mask, 0)
        self.assertEqual(len(self._allowed(mask)), self.vocab)
        self.assertFalse(gen.done)

    def test_model_close_ends_block_and_runs_engine(self):
        state = _state(self.tokenizer)
        gen = self._make(state, engine=VadalogEngine(runner=lambda code: "42"))
        code = " path(X, Y) :- edge(X, Y). "
        seq = self._encode(code) + self._encode("</vadalog>")
        mask = self._mask()
        gen.constrain(seq, mask, 0)
        self.assertTrue(gen.done)
        self.assertEqual(len(state.vadalog_blocks), 1)
        block = state.vadalog_blocks[0]
        self.assertIn("edge(X, Y)", block["code"])
        self.assertNotIn("</vadalog>", block["code"])
        self.assertEqual(block["result"], {"ok": True, "result": "42", "error": None})

    def test_execute_disabled(self):
        state = _state(self.tokenizer)
        gen = self._make(state, execute=False)
        seq = self._encode(" p(X) :- q(X). ") + self._encode("</vadalog>")
        gen.constrain(seq, self._mask(), 0)
        self.assertTrue(gen.done)
        self.assertIsNone(state.vadalog_blocks[0]["result"])

    def test_forced_close_when_too_long(self):
        state = _state(self.tokenizer)
        gen = self._make(state, engine=VadalogEngine(runner=lambda code: None),
                         max_code_tokens=3)
        seq = self._encode(" path(X, Y) :- edge(X, Y). ")[:3]
        # Drive generation: forced-close enforces exactly one allowed token.
        for _ in range(16):
            if gen.done:
                break
            mask = self._mask()
            gen.constrain(seq, mask, 0)
            allowed = self._allowed(mask)
            self.assertEqual(len(allowed), 1, f"expected forced token, got {allowed}")
            seq.append(allowed[0])
        self.assertTrue(gen.done)
        emitted = self.tokenizer.decode(seq)
        self.assertIn("</vadalog>", emitted)
        # the stored code excludes the closing tag
        self.assertNotIn("</vadalog>", state.vadalog_blocks[0]["code"])

    def test_engine_unavailable_is_graceful(self):
        state = _state(self.tokenizer)
        gen = self._make(state, engine=VadalogEngine())  # `vadalog` not installed
        seq = self._encode(" p(X) :- q(X). ") + self._encode("</vadalog>")
        gen.constrain(seq, self._mask(), 0)
        self.assertTrue(gen.done)
        result = state.vadalog_blocks[0]["result"]
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
