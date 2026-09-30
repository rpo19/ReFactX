"""Tests for KnowledgeGraphGeneration, in particular the long_chains stop behaviour:
with long_chains=True the model must be allowed to stop at a hop boundary even
though the graph could always provide another relation.
"""
import math
import unittest

import torch
from transformers import AutoTokenizer

from refactx.index import DictIndex
from refactx.generate import KnowledgeGraphGeneration, PatternConstrainedState


GRAPH = {
    'Paris': {'capital of': ['France'], 'country': ['France']},
    'France': {'continent': ['Europe'], 'currency': ['Euro']},
    'Europe': {'part of': ['European Union']},
}

EOT = ' </kg>\n'


def _allowed(mask, idx=0):
    return set((mask[idx] == 0).nonzero(as_tuple=False).squeeze(-1).tolist())


class TestKnowledgeGraphLongChains(unittest.TestCase):

    def setUp(self):
        # A Qwen tokenizer decodes names cleanly (no added spaces), matching
        # how KnowledgeGraphGeneration is used in practice.
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        self.vocab = len(self.tokenizer)

    # -- helpers ----------------------------------------------------------
    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _make_gen(self, long_chains):
        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        index = DictIndex()
        index.add(self._encode(' <Paris>'))
        return KnowledgeGraphGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0, index=index,
            get_relations=lambda e: list(GRAPH.get(e, {}).keys()),
            get_objects=lambda s, r: GRAPH.get(s, {}).get(r, []),
            long_chains=long_chains, eot=EOT,
        )

    def _constrain(self, gen, seq):
        mask = torch.full((1, self.vocab), -math.inf)
        gen.constrain(seq, mask, 0)
        return _allowed(mask)

    def _feed(self, gen, seq, ids):
        for tok in ids:
            allowed = self._constrain(gen, seq)
            self.assertIn(tok, allowed,
                          f'token {tok} not offered while feeding {ids} (seq={seq})')
            seq.append(tok)

    def _finish_forced_terminal(self, gen, seq):
        """Emit whatever the generator queues deterministically until done."""
        for _ in range(32):
            if gen.done:
                return
            allowed = self._constrain(gen, seq)
            self.assertEqual(len(allowed), 1, f'expected a forced token, got {allowed}')
            seq.append(next(iter(allowed)))
        self.fail('generator did not finish')

    def _reach_first_hop_boundary(self, gen):
        """Feed Paris -> capital of -> France, returning the sequence."""
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <capital of>'))
        self._feed(gen, seq, self._encode(' <France>'))
        # One more constrain triggers the OBJECT->RELATION advance.
        self._constrain(gen, seq)
        return seq

    @staticmethod
    def _triples(path):
        return [(e['entity'], e['relation'], e['object']) for e in path]

    # -- tests ------------------------------------------------------------
    def test_stop_not_offered_before_first_hop(self):
        gen = self._make_gen(long_chains=True)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        # First relation phase: the model still owes at least one triple.
        self.assertIsNone(gen._stop_ids)

    def test_stop_offered_at_hop_boundary(self):
        gen = self._make_gen(long_chains=True)
        seq = self._reach_first_hop_boundary(gen)
        self.assertIsNotNone(gen._stop_ids)
        allowed = self._constrain(gen, seq)
        # Both stopping and continuing must be possible.
        self.assertIn(gen._stop_ids[0], allowed)
        self.assertGreater(len(allowed), 1)

    def test_stop_terminates_path_without_dangling_subject(self):
        gen = self._make_gen(long_chains=True)
        seq = self._reach_first_hop_boundary(gen)
        self._feed(gen, seq, list(gen._stop_ids))
        self._constrain(gen, seq)  # consume the terminal -> completes
        self.assertTrue(gen.done)
        self.assertEqual(len(gen.generated_path_metadata), 1)
        self.assertEqual(self._triples(gen.generated_path_metadata[0]),
                         [('Paris', 'capital of', 'France')])

    def test_continue_then_stop_two_hops(self):
        gen = self._make_gen(long_chains=True)
        seq = self._reach_first_hop_boundary(gen)
        # Continue to a second hop instead of stopping.
        self._feed(gen, seq, self._encode(' <continent>'))
        self._feed(gen, seq, self._encode(' <Europe>'))
        self._constrain(gen, seq)  # advance to the next hop boundary
        self.assertIsNotNone(gen._stop_ids)
        self._feed(gen, seq, list(gen._stop_ids))
        self._constrain(gen, seq)
        self.assertTrue(gen.done)
        self.assertEqual(self._triples(gen.generated_path_metadata[0]),
                         [('Paris', 'capital of', 'France'),
                          ('France', 'continent', 'Europe')])

    def test_long_chains_false_stops_after_one_hop(self):
        gen = self._make_gen(long_chains=False)
        seq = self._reach_first_hop_boundary(gen)
        self.assertIsNone(gen._stop_ids)
        self._finish_forced_terminal(gen, seq)
        self.assertTrue(gen.done)
        self.assertEqual(self._triples(gen.generated_path_metadata[0]),
                         [('Paris', 'capital of', 'France')])


# A small directed graph with a 2-cycle (b -r-> a) and a chain b -r-> c -r-> d.
REVISIT_GRAPH = {
    'a': {'r': ['b']},
    'b': {'r': ['a', 'c']},
    'c': {'r': ['d']},
    'd': {},
}


class TestKnowledgeGraphRevisits(unittest.TestCase):

    def setUp(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        self.vocab = len(self.tokenizer)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _make_gen(self, forbid_revisits):
        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        index = DictIndex()
        index.add(self._encode(' <a>'))
        return KnowledgeGraphGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0, index=index,
            get_relations=lambda e: list(REVISIT_GRAPH.get(e, {}).keys()),
            get_objects=lambda s, r: REVISIT_GRAPH.get(s, {}).get(r, []),
            long_chains=True, eot=' </kg>\n', forbid_revisits=forbid_revisits,
        )

    def _constrain(self, gen, seq):
        mask = torch.full((1, self.vocab), -math.inf)
        gen.constrain(seq, mask, 0)
        return _allowed(mask)

    def _feed(self, gen, seq, ids):
        for tok in ids:
            allowed = self._constrain(gen, seq)
            self.assertIn(tok, allowed,
                          f'token {tok} not offered while feeding {ids} (seq={seq})')
            seq.append(tok)

    def _object_names(self, gen):
        return {value[0] for value in gen.phase_names.values()}

    def _reach_b_objects(self, gen):
        """Path a -> b, then relation r from b; returns (seq, object names)."""
        seq = []
        self._feed(gen, seq, self._encode(' <a>'))
        self._feed(gen, seq, self._encode(' <r>'))
        self._feed(gen, seq, self._encode(' <b>'))
        self._constrain(gen, seq)                       # hop boundary at b
        self._feed(gen, seq, self._encode(' <r>'))      # b's only relation
        self._constrain(gen, seq)                       # begin objects for (b, r)
        return seq, self._object_names(gen)

    def test_revisited_entity_is_not_offered(self):
        gen = self._make_gen(forbid_revisits=True)
        _, objects = self._reach_b_objects(gen)
        # 'a' is already on the path (and closes the 2-cycle) -> excluded.
        self.assertEqual(objects, {'c'})

    def test_revisits_allowed_when_disabled(self):
        gen = self._make_gen(forbid_revisits=False)
        _, objects = self._reach_b_objects(gen)
        self.assertEqual(objects, {'a', 'c'})

    def test_chain_still_extends_to_new_entities(self):
        gen = self._make_gen(forbid_revisits=True)
        seq, objects = self._reach_b_objects(gen)
        self.assertIn('c', objects)
        # Continue c -> d (both new, no reversal).
        self._feed(gen, seq, self._encode(' <c>'))
        self._constrain(gen, seq)                       # hop boundary at c
        self._feed(gen, seq, self._encode(' <r>'))
        self._constrain(gen, seq)                       # objects for (c, r)
        self.assertEqual(self._object_names(gen), {'d'})


if __name__ == '__main__':
    unittest.main()
