"""Tests for the vLLM V1 logits-processor adapter.

The tests exercise batch bookkeeping and masking without starting a vLLM
engine.  They are skipped when ``vllm`` cannot be imported.
"""

import math
import types
import unittest

import torch
from transformers import AutoTokenizer

try:
    # Import the adapter first: it applies the transformers/vLLM compatibility
    # shim before vLLM imports transformers.configuration_utils.
    from refactx.vllm_proc import RefactxLogitsProcessor, _ReqState
    from refactx.generate import activate_constrained_fact
    from vllm.v1.sample.logits_processor import BatchUpdate, MoveDirectionality
    VLLM_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    VLLM_AVAILABLE = False


class FakeIndex:
    """Maps a partial-fact token tuple to ``{allowed_token: leaf_count}``."""

    def __init__(self, allowed):
        self.allowed = allowed

    def next_tokens(self, sequence, state=None):
        return dict(self.allowed.get(tuple(sequence), {})), None


def _make_proc(tokenizer, index):
    proc = RefactxLogitsProcessor.__new__(RefactxLogitsProcessor)
    proc.device = torch.device("cpu")
    proc.tokenizer = tokenizer
    proc.index = index
    proc.fact_pattern = "<fact>"
    proc.count_pattern = "<count>"
    proc.eot = " </fact>\n"
    proc.sentinel = False
    proc.force_constrained = False
    proc._mask_buffer = None
    proc._req = {}
    proc._facts = {}
    return proc


def _added(index, enabled=True, corr=None, out=None, table=None):
    extra = {}
    if enabled:
        extra["refactx"] = True
    if corr is not None:
        extra["refactx_id"] = corr
    if table is not None:
        extra["index_table"] = table
    params = types.SimpleNamespace(extra_args=extra)
    return (index, params, None, [] if out is None else out)


def _batch(batch_size=4, removed=(), added=(), moved=()):
    return BatchUpdate(
        batch_size=batch_size, removed=list(removed),
        added=list(added), moved=list(moved),
    )


@unittest.skipUnless(VLLM_AVAILABLE, "vllm is not importable")
class TestRefactxVLLMProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def test_only_enabled_requests_are_tracked(self):
        proc = _make_proc(self.tokenizer, FakeIndex({}))
        proc.update_state(_batch(added=[
            _added(0, enabled=True), _added(1, enabled=False),
        ]))
        self.assertEqual(set(proc._req), {0})

    def test_remove_and_move_bookkeeping(self):
        proc = _make_proc(self.tokenizer, FakeIndex({}))
        proc.update_state(_batch(added=[_added(0), _added(1)]))
        state0, state1 = proc._req[0], proc._req[1]

        proc.update_state(_batch(moved=[(0, 1, MoveDirectionality.SWAP)]))
        self.assertIs(proc._req[0], state1)
        self.assertIs(proc._req[1], state0)

        proc.update_state(_batch(moved=[(1, 0, MoveDirectionality.UNIDIRECTIONAL)]))
        self.assertIs(proc._req[0], state0)
        self.assertNotIn(1, proc._req)

        proc.update_state(_batch(removed=[0]))
        self.assertEqual(proc._req, {})

    def test_apply_masks_to_allowed_tokens_and_keeps_scores(self):
        proc = _make_proc(self.tokenizer, FakeIndex({(): {3: 1, 7: 1}}))
        proc.update_state(_batch(batch_size=1, added=[_added(0)]))
        req = proc._req[0]
        activate_constrained_fact(
            req.state, proc.index, sentinel=False, eot=proc.eot,
        )

        logits = torch.randn(1, 10)
        original = logits.clone()
        out = proc.apply(logits)

        self.assertIs(out, logits)
        torch.testing.assert_close(out[0, [3, 7]], original[0, [3, 7]])
        blocked = [i for i in range(10) if i not in (3, 7)]
        self.assertTrue(torch.isinf(out[0, blocked]).all())
        self.assertTrue((out[0, blocked] < 0).all())

    def test_unconstrained_rows_are_untouched(self):
        proc = _make_proc(self.tokenizer, FakeIndex({(): {3: 1}}))
        proc.update_state(_batch(added=[_added(0), _added(1, enabled=False)]))
        activate_constrained_fact(
            proc._req[0].state, proc.index, sentinel=False, eot=proc.eot,
        )

        logits = torch.randn(2, 10)
        original = logits.clone()
        proc.apply(logits)

        torch.testing.assert_close(logits[1], original[1])
        self.assertTrue(torch.isfinite(logits[0, 3]))
        self.assertTrue(torch.isinf(logits[0, 0]))

    def test_trigger_switches_to_constrained(self):
        proc = _make_proc(self.tokenizer, FakeIndex({(): {5: 1}}))
        out_ids = self.tokenizer.encode("The answer is Paris. <fact>")
        proc.update_state(_batch(batch_size=1, added=[_added(0, out=out_ids)]))
        proc.apply(torch.zeros(1, len(self.tokenizer)))
        self.assertTrue(proc._req[0].state.is_constrained())

    def test_facts_stashed_for_correlation_id(self):
        proc = _make_proc(self.tokenizer, FakeIndex({}))
        proc.update_state(_batch(batch_size=1, added=[_added(0, corr="q0")]))
        triple = self.tokenizer.encode(" Paris capital France .")
        proc._req[0].state.generated_triples.append(triple)
        proc.update_state(_batch(batch_size=0, removed=[0]))

        facts = proc.pop_facts("q0")
        self.assertEqual(len(facts), 1)
        self.assertIn("paris", facts[0].lower())
        self.assertEqual(proc.pop_facts("q0"), [])

    def test_validate_params(self):
        with self.assertRaises(ValueError):
            RefactxLogitsProcessor.validate_params(
                types.SimpleNamespace(extra_args={"refactx": "yes"}))
        with self.assertRaises(ValueError):
            RefactxLogitsProcessor.validate_params(
                types.SimpleNamespace(extra_args={"refactx": True,
                                                  "index_table": 3}))
        RefactxLogitsProcessor.validate_params(
            types.SimpleNamespace(extra_args={"refactx": True,
                                              "index_table": "qwen36"}))


if __name__ == "__main__":
    unittest.main()
