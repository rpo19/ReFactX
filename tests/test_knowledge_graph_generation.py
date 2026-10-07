"""Tests for KnowledgeGraphGeneration.

Covers three behaviours:

* ``long_chains`` stop: with long chains the model must be allowed to stop at a
  hop boundary even though the graph could always provide another relation.
* ``forbid_revisits``: the walk stays acyclic (no entity revisited, no edge
  traversed backwards).
* ``avoid_duplicates``: the same *triple* is never generated twice, while the
  same *subject* may be used to start several distinct triples.  When
  ``sentinel`` is enabled an exhausted branch ends with an explicit
  ``<no further records>`` object.
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


class _KGTestMixin:
    @classmethod
    def setUpClass(cls):
        # A Qwen tokenizer decodes names cleanly (no added spaces), matching
        # how KnowledgeGraphGeneration is used in practice.
        cls.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
        cls.vocab = len(cls.tokenizer)

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

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

    def _drain(self, gen, seq):
        """Emit whatever the generator queues until it is done."""
        emitted = []
        for _ in range(64):
            if gen.done:
                return emitted
            allowed = self._constrain(gen, seq)
            self.assertTrue(allowed, 'generation stalled with no token offered')
            tok = next(iter(sorted(allowed)))
            seq.append(tok)
            emitted.append(tok)
        self.fail('generator did not finish')

    def _make_gen(self, state, index, long_chains=False, avoid_duplicates=True,
                  forbid_revisits=True, sentinel=False, graph=GRAPH):
        return KnowledgeGraphGeneration(
            state=state, tokenizer=self.tokenizer, start_idx=0, index=index,
            get_relations=lambda e: list(graph.get(e, {}).keys()),
            get_objects=lambda s, r: graph.get(s, {}).get(r, []),
            long_chains=long_chains, eot=EOT, avoid_duplicates=avoid_duplicates,
            forbid_revisits=forbid_revisits, sentinel=sentinel,
        )

    def _new_state_and_index(self, entity=' <Paris>'):
        state = PatternConstrainedState(
            tokenizer=self.tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
        index = DictIndex()
        index.add(self._encode(entity))
        return state, index


class TestKnowledgeGraphLongChains(_KGTestMixin, unittest.TestCase):

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

    def test_stop_not_offered_before_first_hop(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, long_chains=True)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        # First relation phase: the model still owes at least one triple.
        self.assertIsNone(gen._stop_ids)

    def test_stop_offered_at_hop_boundary(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, long_chains=True)
        seq = self._reach_first_hop_boundary(gen)
        self.assertIsNotNone(gen._stop_ids)
        allowed = self._constrain(gen, seq)
        # Both stopping and continuing must be possible.
        self.assertIn(gen._stop_ids[0], allowed)
        self.assertGreater(len(allowed), 1)

    def test_stop_terminates_path_without_dangling_subject(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, long_chains=True)
        seq = self._reach_first_hop_boundary(gen)
        self._feed(gen, seq, list(gen._stop_ids))
        self._constrain(gen, seq)  # consume the terminal -> completes
        self.assertTrue(gen.done)
        self.assertEqual(len(gen.generated_path_metadata), 1)
        self.assertEqual(self._triples(gen.generated_path_metadata[0]),
                         [('Paris', 'capital of', 'France')])

    def test_continue_then_stop_two_hops(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, long_chains=True)
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
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, long_chains=False)
        seq = self._reach_first_hop_boundary(gen)
        self.assertIsNone(gen._stop_ids)
        self._drain(gen, seq)
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


class TestKnowledgeGraphRevisits(_KGTestMixin, unittest.TestCase):

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
        state, index = self._new_state_and_index(' <a>')
        gen = self._make_gen(state, index, long_chains=True,
                             forbid_revisits=True, graph=REVISIT_GRAPH)
        _, objects = self._reach_b_objects(gen)
        # 'a' is already on the path (and closes the 2-cycle) -> excluded.
        self.assertEqual(objects, {'c'})

    def test_revisits_allowed_when_disabled(self):
        state, index = self._new_state_and_index(' <a>')
        gen = self._make_gen(state, index, long_chains=True,
                             forbid_revisits=False, graph=REVISIT_GRAPH)
        _, objects = self._reach_b_objects(gen)
        self.assertEqual(objects, {'a', 'c'})

    def test_chain_still_extends_to_new_entities(self):
        state, index = self._new_state_and_index(' <a>')
        gen = self._make_gen(state, index, long_chains=True,
                             forbid_revisits=True, graph=REVISIT_GRAPH)
        seq, objects = self._reach_b_objects(gen)
        self.assertIn('c', objects)
        # Continue c -> d (both new, no reversal).
        self._feed(gen, seq, self._encode(' <c>'))
        self._constrain(gen, seq)                       # hop boundary at c
        self._feed(gen, seq, self._encode(' <r>'))
        self._constrain(gen, seq)                       # objects for (c, r)
        self.assertEqual(self._object_names(gen), {'d'})


class TestKnowledgeGraphAvoidDuplicates(_KGTestMixin, unittest.TestCase):
    """``avoid_duplicates`` must forbid repeated triples, not repeated subjects."""

    def _complete_triple(self, gen, seq, subject, relation, obj):
        self._feed(gen, seq, self._encode(f' <{subject}>'))
        self._feed(gen, seq, self._encode(f' <{relation}>'))
        self._feed(gen, seq, self._encode(f' <{obj}>'))
        self._drain(gen, seq)

    def test_same_subject_can_start_a_second_generation(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index)
        seq = []
        self._complete_triple(gen, seq, 'Paris', 'capital of', 'France')

        # A fresh generation in the same state must still be able to pick the
        # same subject (this used to be wrongly forbidden).
        gen2 = self._make_gen(state, index)
        allowed = self._constrain(gen2, [])
        self.assertIn(self._encode(' <Paris>')[0], allowed)

    def test_duplicate_triple_object_is_not_offered(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index)
        seq = []
        self._complete_triple(gen, seq, 'Paris', 'capital of', 'France')

        # (Paris, capital of) has a single object (France) which was already
        # generated, so at the object phase there is no live token left.
        gen2 = self._make_gen(state, index)
        seq2 = []
        self._feed(gen2, seq2, self._encode(' <Paris>'))
        self._feed(gen2, seq2, self._encode(' <capital of>'))
        allowed = self._constrain(gen2, seq2)
        self.assertNotIn(self._encode(' <France>')[0], allowed)

    def test_same_subject_relation_different_object_allowed(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index)
        seq = []
        self._complete_triple(gen, seq, 'Paris', 'capital of', 'France')

        # Re-using the subject and choosing a different relation stays allowed.
        gen2 = self._make_gen(state, index)
        seq2 = []
        self._feed(gen2, seq2, self._encode(' <Paris>'))
        allowed = self._constrain(gen2, seq2)
        self.assertIn(self._encode(' <country>')[0], allowed)
        self.assertIn(self._encode(' <capital of>')[0], allowed)

    def test_pruning_only_applies_to_object_phase(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index)
        # Prime the duplicate cache with a whole path starting at the shared
        # first token; selecting a subject must not be pruned by it.
        gen.state.cache_index.add(self._encode(' <Paris> <capital of> <France>'),
                                  new_leaf=True)
        self.assertEqual(gen.phase, KnowledgeGraphGeneration.ENTITY)
        allowed = self._constrain(gen, [])
        self.assertIn(self._encode(' <Paris>')[0], allowed)


class TestKnowledgeGraphSentinel(_KGTestMixin, unittest.TestCase):
    """The sentinel marks an exhausted branch with ``<no further records>``."""

    def _prime_cache_with_triple(self, state, subject, relation, obj):
        ids = []
        for text in (f' <{subject}>', f' <{relation}>', f' <{obj}>'):
            ids += self._encode(text)
        state.cache_index.add(ids, new_leaf=True)

    def test_sentinel_emitted_when_all_objects_duplicated(self):
        state, index = self._new_state_and_index()
        self._prime_cache_with_triple(state, 'Paris', 'capital of', 'France')
        gen = self._make_gen(state, index, sentinel=True)

        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <capital of>'))
        emitted = self._drain(gen, seq)
        text = self.tokenizer.decode(emitted)
        self.assertIn('no further records', text)
        self.assertTrue(text.rstrip().endswith('</kg>'))

    def test_no_sentinel_when_disabled(self):
        state, index = self._new_state_and_index()
        self._prime_cache_with_triple(state, 'Paris', 'capital of', 'France')
        gen = self._make_gen(state, index, sentinel=False)

        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <capital of>'))
        emitted = self._drain(gen, seq)
        text = self.tokenizer.decode(emitted)
        self.assertNotIn('no further records', text)
        self.assertTrue(text.rstrip().endswith('</kg>'))

    def test_no_sentinel_after_successful_triple(self):
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True)

        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <capital of>'))
        self._feed(gen, seq, self._encode(' <France>'))
        emitted = self._drain(gen, seq)
        text = self.tokenizer.decode(emitted)
        self.assertNotIn('no further records', text)

    def test_sentinel_emitted_when_relation_has_no_objects(self):
        empty_graph = {'Paris': {'capital of': [], 'country': ['France']}}
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True, graph=empty_graph)

        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <capital of>'))
        emitted = self._drain(gen, seq)
        text = self.tokenizer.decode(emitted)
        self.assertIn('no further records', text)


class TestKnowledgeGraphDeadBranches(_KGTestMixin, unittest.TestCase):
    """Once a branch is known to hold no new fact it must stop being offered.

    The sentinel is shown once (so the model learns the branch is empty) and the
    prefix is then forbidden, which breaks the "<S> <R> <no further records>"
    repetition loop.
    """

    def _dead_relations(self, state):
        return state.kg_memory.get('dead_relations', set())

    def _dead_entities(self, state):
        return state.kg_memory.get('dead_entities', set())

    def test_relation_without_objects_is_marked_dead(self):
        graph = {'Paris': {'capital of': ['France'], 'ghost': []}}
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True, graph=graph)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <ghost>'))
        self._drain(gen, seq)
        self.assertIn(('Paris', 'ghost'), self._dead_relations(state))

    def test_dead_relation_is_not_offered_again(self):
        graph = {'Paris': {'capital of': ['France'], 'ghost': []}}
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True, graph=graph)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <ghost>'))
        self._drain(gen, seq)

        gen2 = self._make_gen(state, index, sentinel=True, graph=graph)
        seq2 = []
        self._feed(gen2, seq2, self._encode(' <Paris>'))
        self._constrain(gen2, seq2)  # advance to the relation index
        names = {name for name, _ in gen2.phase_names.values()}
        self.assertNotIn('ghost', names)
        self.assertIn('capital of', names)

    def test_entity_without_live_relations_is_marked_dead(self):
        graph = {'Paris': {'ghost': []}}
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True, graph=graph)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <ghost>'))
        self._drain(gen, seq)
        # <ghost> has no object, and it is Paris' only relation, so Paris is
        # immediately remembered as dead.
        self.assertIn(('Paris', 'ghost'), self._dead_relations(state))
        self.assertIn('Paris', self._dead_entities(state))

    def test_dead_entity_prefix_not_offered_again(self):
        graph = {'Paris': {'ghost': []}}
        state, index = self._new_state_and_index()
        gen = self._make_gen(state, index, sentinel=True, graph=graph)
        seq = []
        self._feed(gen, seq, self._encode(' <Paris>'))
        self._feed(gen, seq, self._encode(' <ghost>'))
        self._drain(gen, seq)

        # A later pass may still show the sentinel once for the dead prefix,
        # but must never emit the entity name again.
        gen2 = self._make_gen(state, index, sentinel=True, graph=graph)
        seq2 = []
        emitted = self._drain(gen2, seq2)
        text = self.tokenizer.decode(emitted)
        self.assertNotIn('Paris', text)

        gen3 = self._make_gen(state, index, sentinel=True, graph=graph)
        seq3 = []
        emitted3 = self._drain(gen3, seq3)
        self.assertNotIn('Paris', self.tokenizer.decode(emitted3))

    def test_is_fully_dead_only_covers_dead_subtree(self):
        state, index = self._new_state_and_index(' <Paris>')
        index.add(self._encode(' <Lyon>'))
        gen = self._make_gen(state, index)
        gen._mark_entity_dead('Paris', self._encode(' <Paris>'))
        self.assertTrue(gen._is_fully_dead(self._encode(' <Paris>')))
        self.assertFalse(gen._is_fully_dead(self._encode(' <Lyon>')))
        self.assertFalse(gen._is_fully_dead([]))  # Lyon is still live

    def test_partial_prefix_gets_sentinel(self):
        state, index = self._new_state_and_index(' <Paris>')
        index.add(self._encode(' <Lyon>'))
        gen = self._make_gen(state, index, sentinel=True)
        gen._mark_entity_dead('Paris', self._encode(' <Paris>'))
        # Typing the dead entity's name must end in a sentinel, without ever
        # completing the entity.
        seq = []
        for tok in self._encode(' <Paris'):
            allowed = self._constrain(gen, seq)
            if tok not in allowed:
                break
            seq.append(tok)
        emitted = self._drain(gen, seq)
        text = self.tokenizer.decode(emitted)
        self.assertIn('no further records', text)


if __name__ == '__main__':
    unittest.main()
