"""Apply a rule constraint to the generated suffix of a single sequence."""
import time
from collections import defaultdict

import torch
from transformers.generation.logits_process import LogitsProcessor


class RuleDecodingError(RuntimeError):
    def __init__(self, message, prefix, n_tokens):
        super().__init__(message)
        self.prefix = prefix
        self.n_tokens = n_tokens


class RuleTokenIndex:
    """Reusable token spellings and candidate IDs for one tokenizer."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        special_ids = set(tokenizer.all_special_ids)
        self.strings = tokenizer.batch_decode(
            [[token_id] for token_id in range(len(tokenizer))],
            clean_up_tokenization_spaces=False,
        )
        self.by_first = defaultdict(list)
        for token_id, spelling in enumerate(self.strings):
            if spelling and token_id not in special_ids:
                self.by_first[spelling[0]].append(token_id)


class RuleLogitsProcessor(LogitsProcessor):
    def __init__(self, token_index, constraint, generation_start):
        self.token_index = token_index
        self.constraint = constraint
        self.generation_start = generation_start
        self.mask_steps = 0
        self.mask_s = 0.0
        self._cc_cache = {}

    def _allowed(self, state, vocab_size):
        keep = []
        for ch in sorted(self.constraint.next_chars(state)):
            for token_id in self.token_index.by_first.get(ch, ()):
                if token_id >= vocab_size:
                    continue
                nxt = self.constraint.consume(state, self.token_index.strings[token_id])
                if nxt and self.constraint.can_complete(nxt, self._cc_cache):
                    keep.append(token_id)
        return keep

    def __call__(self, input_ids, scores):
        t0 = time.perf_counter()
        try:
            ids = input_ids[0] if input_ids.ndim == 2 else input_ids
            ids = ids[self.generation_start:].tolist()
            tokenizer = self.token_index.tokenizer
            text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            state = self.constraint.consume(self.constraint.initial, text)
            if not state:
                raise RuleDecodingError("generated prefix left the rule grammar", text, len(ids))
            if self.constraint.accepting(state):
                eos = tokenizer.eos_token_id
                keep = [eos] if eos is not None and eos < scores.shape[-1] else []
            else:
                keep = self._allowed(state, scores.shape[-1])
            if not keep:
                raise RuleDecodingError("no token can continue the rule", text, len(ids))
            mask = torch.ones_like(scores, dtype=torch.bool)
            mask[..., keep] = False
            self.mask_steps += 1
            return scores.masked_fill(mask, -float("inf"))
        finally:
            self.mask_s += time.perf_counter() - t0
