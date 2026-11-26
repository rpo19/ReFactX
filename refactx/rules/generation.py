"""``<rule>`` mode: constrained generation of one Vadalog rule.

Registered on the ReFactX pattern processor next to ``<kg>``::

    processor.add_pattern('<rule>', RuleGeneration, kg=kg)

When the model writes ``<rule>`` the processor activates this class.  Every
following token is masked by a :class:`VadalogRuleConstraint` built from the
*current* KG schema (stored relations + derived relations + rules added
earlier in the same answer), so the rule is syntactically valid Vadalog and
only uses existing predicates, arities, types and constants.  As soon as the
rule is complete the class

1. sends it to Vadalog (``kg.add_rule``): evaluated on the spot and, if it
   succeeds, registered so ``<kg>`` and later ``<rule>`` blocks can use the
   new relation;
2. forces the closing tag and a ``<result> ... </result>`` block with what
   Vadalog derived -- the model reads it like a tool answer and goes on in
   free generation.
"""
from __future__ import annotations

import math
import time

from refactx.generate import PatternConstrainedGeneration
from refactx.rules.token_mask import SortedVocab, raw_tokenizer


class RuleGeneration(PatternConstrainedGeneration):
    def __init__(self, state, tokenizer, start_idx, kg, close_text=" </rule>\n",
                 result_open="<result> ", result_close=" </result>\n",
                 show_results=20, register=True, execute=True, vocab=None,
                 constraint_kwargs=None):
        super().__init__(state, tokenizer, start_idx)
        self.kg = kg
        self.close_text = close_text
        self.result_open, self.result_close = result_open, result_close
        self.show_results = show_results
        self.register = register
        self.execute = execute
        self.constraint = kg.rule_constraint(**(constraint_kwargs or {}))
        self.vocab = vocab or SortedVocab.for_tokenizer(tokenizer)
        self._cc = {}
        self._forced = []
        self.rule_text = None
        self.result = None
        self.steps = []            # per generated token: (#allowed, seconds)
        self.mask_seconds = 0.0
        self.error = None
        if not hasattr(state, "rule_blocks"):
            state.rule_blocks = []

    def __deepcopy__(self, memo):  # beams share the heavy, read-only parts
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone._forced = list(self._forced)
        clone.steps = list(self.steps)
        return clone

    # ------------------------------------------------------------ helpers
    def _encode(self, text):
        return list(raw_tokenizer(self.tokenizer).encode(text, add_special_tokens=False))

    def _decode(self, ids):
        return raw_tokenizer(self.tokenizer).decode(ids, clean_up_tokenization_spaces=False)

    def _emit_forced(self, mask, mask_idx):
        token = self._forced.pop(0)
        mask[mask_idx, :] = -math.inf
        if 0 <= token < mask.shape[-1]:
            mask[mask_idx, token] = 0
        if not self._forced:
            self.done = True

    def _finish(self, rule_text):
        """Run the rule in Vadalog and queue the forced tail."""
        self.rule_text = rule_text.strip()
        tail = self.close_text
        if self.execute:
            self.result = self.kg.evaluate(self.rule_text, register=self.register)
            tail += self.result_open + self.result.summary(self.show_results) + self.result_close
        self.state.rule_blocks.append(self)
        self._forced = self._encode(tail)

    # ------------------------------------------------------------ main hook
    def constrain(self, sequence, mask, mask_idx):
        if self._forced:
            self._emit_forced(mask, mask_idx)
            return
        t0 = time.perf_counter()
        text = self._decode(sequence)
        subset = self.constraint.consume(self.constraint.initial, text)
        if subset and self.constraint.accepting(subset):
            self._finish(text)
            self._emit_forced(mask, mask_idx)
            return
        allowed = self.vocab.allowed(self.constraint, subset, self._cc,
                                     limit=mask.shape[-1]) if subset else []
        if not allowed:
            # Should not happen (every reachable prefix is completable); close
            # the block instead of letting the model run away.
            self.error = "dead prefix: %r" % text
            self.rule_text = text.strip()
            self.state.rule_blocks.append(self)
            self._forced = self._encode(self.close_text + self.result_open
                                        + "error: incomplete rule" + self.result_close)
            self._emit_forced(mask, mask_idx)
            return
        mask[mask_idx, allowed] = 0
        dt = time.perf_counter() - t0
        self.mask_seconds += dt
        self.steps.append((len(allowed), dt))
