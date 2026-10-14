"""Tests for dynamic entity grounding of <vadalog> (Phase 2b)."""
import unittest

import torch
from transformers import AutoTokenizer

from refactx.vadalog_entities import EntityGrounder

ENTITIES = ["Acme Corp", "Acme GmbH", "Globex", "Initech Limited"]


class TestInString(unittest.TestCase):

    def test_outside(self):
        self.assertEqual(EntityGrounder.in_string(" path(X) :- own(X, "), (False, ""))

    def test_inside(self):
        self.assertEqual(EntityGrounder.in_string(' a "Acme C'), (True, "Acme C"))

    def test_closed(self):
        self.assertEqual(EntityGrounder.in_string(' a "Acme" '), (False, ""))

    def test_escape(self):
        self.assertEqual(EntityGrounder.in_string(r' a "Ac\"me'), (True, r"Ac\"me"))


class TestSimulate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.g = EntityGrounder(cls.tokenizer, ENTITIES)

    def test_valid_opening(self):
        self.assertIsNotNone(self.g._simulate(False, "", '"Acme'))
        self.assertIsNotNone(self.g._simulate(False, "", '"Acme Corp"'))

    def test_invalid_opening(self):
        self.assertIsNone(self.g._simulate(False, "", '"Unknown'))
        self.assertIsNone(self.g._simulate(False, "", '")'))  # empty string

    def test_continue_content(self):
        self.assertIsNotNone(self.g._simulate(True, "Acme", " Corp"))
        self.assertIsNone(self.g._simulate(True, "Acme", " Xyz"))

    def test_close_only_when_complete(self):
        self.assertIsNotNone(self.g._simulate(True, "Acme Corp", '"'))
        self.assertIsNone(self.g._simulate(True, "Acme", '"'))  # not a full name


class TestRestrict(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)
        cls.g = EntityGrounder(cls.tokenizer, ENTITIES)

    def _allowed(self, text):
        mask = torch.zeros(1, self.vocab)
        self.g.restrict(mask, 0, text)
        return set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())

    def test_outside_removes_bad_quote_tokens(self):
        allowed = self._allowed(" path(X) :- own(X, ")
        # some quote tokens are pruned, most tokens remain
        self.assertLess(len(allowed), self.vocab)
        self.assertGreater(len(allowed), self.vocab - len(self.g.quote_items))

    def test_inside_restricts_to_prefix(self):
        allowed = self._allowed(' own(X, "Ac')
        # a valid continuation of "Ac" (towards Acme) is offered
        self.assertIn(self.tokenizer.encode("me", add_special_tokens=False)[0], allowed)
        # a diverging continuation is not
        self.assertNotIn(self.tokenizer.encode("zz", add_special_tokens=False)[0], allowed)

    def test_close_allowed_when_complete(self):
        allowed = self._allowed(' own(X, "Globex')
        self.assertIn(self.tokenizer.encode('"', add_special_tokens=False)[0], allowed)

    def test_close_not_allowed_when_incomplete(self):
        allowed = self._allowed(' own(X, "Glob')
        self.assertNotIn(self.tokenizer.encode('"', add_special_tokens=False)[0], allowed)


class TestGenerationWithEntities(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        cls.vocab = len(cls.tokenizer)

    def test_grounded_entity_completes(self):
        from refactx.generate import PatternConstrainedState
        from refactx.index import DictIndex
        from refactx.vadalog import VadalogGeneration

        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        gen = VadalogGeneration(state, self.tokenizer, 0, entities=ENTITIES)
        seq = []
        tokens = (self.tokenizer.encode(' p(X) :- own(X, "Acme Corp"). ',
                                        add_special_tokens=False)
                  + self.tokenizer.encode("</vadalog>", add_special_tokens=False))
        for token in tokens:
            mask = torch.full((1, self.vocab), float("-inf"))
            gen.constrain(seq, mask, 0)
            allowed = set((mask[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())
            self.assertIn(token, allowed, f"token {self.tokenizer.decode([token])!r} not offered")
            seq.append(token)
        gen.constrain(seq, torch.full((1, self.vocab), float("-inf")), 0)
        self.assertTrue(gen.done)
        self.assertIn("Acme Corp", state.vadalog_blocks[0]["code"])


if __name__ == "__main__":
    unittest.main()
