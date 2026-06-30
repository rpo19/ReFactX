"""
Custom FSM logit processor using xgrammar.

Architecture:
- xgrammar for bitmask infrastructure (allocate, apply, reset)
- xgrammar GrammarMatcher for JSON schema constraint (Answer: mode)
- DictIndex for KB trie lookup (Fact: mode)
- Custom FSM for pattern-triggered state transitions

States:
  free    - unconstrained generation
  triple  - constrained to valid KB triples (trigger: "Fact:")
  json    - constrained to JSON schema    (trigger: "Answer:")
"""
import math
from typing import Dict, List, Optional, Union

import torch
from transformers import LogitsProcessor

import xgrammar as xgr
from refactx.index import DictIndex, EmptyIndexException, TripleNotFoundException


class XgrammarConstrainedProcessor(LogitsProcessor):
    MODE_FREE = "free"
    MODE_TRIPLE = "triple"
    MODE_JSON = "json"

    def __init__(
        self,
        tokenizer,
        kb_index: DictIndex,
        json_schema: Optional[Union[str, Dict]] = None,
        fact_pattern: str = "Fact:",
        answer_pattern: str = "Answer:",
        eot: str = "\n",
        avoid_duplicates: bool = True,
        regex_window: int = 20,
    ):
        self.tokenizer = tokenizer
        self.kb_index = kb_index
        self.fact_pattern = fact_pattern.lower()
        self.answer_pattern = answer_pattern.lower()
        self.eot = eot
        self.avoid_duplicates = avoid_duplicates
        self.regex_window = regex_window
        self._visited_index = DictIndex()

        self.vocab_size = len(tokenizer)
        self._make_eot_token()

        self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size)

        self._json_matcher = None
        if json_schema is not None:
            tok_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            compiler = xgr.GrammarCompiler(tok_info)
            compiled = compiler.compile_json_schema(json_schema)
            self._json_matcher = xgr.GrammarMatcher(
                compiled, terminate_without_stop_token=True
            )

        self.reset()

    def _make_eot_token(self):
        self.eot_token = None
        if self.eot is not None:
            enc = self.tokenizer.encode(self.eot, add_special_tokens=False)
            self.eot_token = enc[0] if enc else None

    def reset(self):
        self.mode = self.MODE_FREE
        self.input_ids = []
        self.triple_seq = []
        self.generated_triples = []
        self._first_call = True
        self._entering_mode = False
        self._visited_index = DictIndex()
        if self._json_matcher is not None:
            self._json_matcher.reset()

    def _get_text_window(self) -> str:
        text = self.tokenizer.decode(self.input_ids[-self.regex_window:])
        return text

    def _check_pattern(self) -> Optional[str]:
        text = self._get_text_window()
        if self.mode == self.MODE_FREE:
            normalized = "".join(text.split())
            if normalized.lower().endswith(self.fact_pattern):
                return self.fact_pattern
            if normalized.lower().endswith(self.answer_pattern):
                return self.answer_pattern
        return None

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            seq = input_ids[i].tolist()

            if self._first_call:
                self._first_call = False
                self.input_ids = seq
                continue

            last_token = seq[-1]

            if seq[:-1] != self.input_ids:
                self.reset()
                self.input_ids = seq
                continue

            self.input_ids = seq

            if self.mode == self.MODE_FREE:
                detected = self._check_pattern()
                if detected == self.fact_pattern:
                    self.mode = self.MODE_TRIPLE
                    self.triple_seq = []
                    self._entering_mode = True
                elif detected == self.answer_pattern:
                    self.mode = self.MODE_JSON
                    self._entering_mode = True
                    if self._json_matcher is not None:
                        self._json_matcher.reset()

            if self.mode == self.MODE_TRIPLE:
                self._mask_for_triple(scores, i)
            elif self.mode == self.MODE_JSON:
                self._mask_for_json(scores, i)

        return scores

    def _allow_tokens(self, scores, batch_idx, token_ids):
        scores[batch_idx] = -math.inf
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                scores[batch_idx, tid] = 0

    def _mask_for_triple(self, scores, batch_idx):
        if self._entering_mode:
            self._entering_mode = False
            return
        self.triple_seq.append(self.input_ids[-1])
        triple_prefix = self.triple_seq[:]

        try:
            possible, _ = self.kb_index.next_tokens(triple_prefix)
        except (EmptyIndexException, TripleNotFoundException):
            self.mode = self.MODE_FREE
            self.triple_seq = []
            if self.eot_token is not None:
                self._allow_tokens(scores, batch_idx, [self.eot_token])
            return

        natural_end = (len(possible) == 0)

        if self.avoid_duplicates:
            try:
                visited, _ = self._visited_index.next_tokens(triple_prefix)
            except (EmptyIndexException, TripleNotFoundException):
                visited = {}
            for tid in list(possible.keys()):
                if tid in visited:
                    diff = possible[tid] - visited[tid]
                    if diff <= 0:
                        del possible[tid]

        if isinstance(possible, dict):
            allowed = list(possible.keys())
        else:
            allowed = list(possible) if possible else []

        if len(allowed) == 0:
            if natural_end and self.triple_seq:
                self.generated_triples.append(self.triple_seq[:])
                if self.avoid_duplicates:
                    triple_text = self.tokenizer.decode(self.triple_seq).lstrip()
                    for prefix in ("", " "):
                        alt_ids = self.tokenizer.encode(
                            prefix + triple_text, add_special_tokens=False
                        )
                        self._visited_index.add(alt_ids)
            self.mode = self.MODE_FREE
            self.triple_seq = []
            if self.eot_token is not None:
                allowed = [self.eot_token]

        self._allow_tokens(scores, batch_idx, allowed)

    def _mask_for_json(self, scores, batch_idx):
        if self._json_matcher is None:
            return
        if self._entering_mode:
            self._entering_mode = False
            return
        self._json_matcher.accept_token(self.input_ids[-1])
        self._json_matcher.fill_next_token_bitmask(self.bitmask, 0)
        xgr.apply_token_bitmask_inplace(scores, self.bitmask)

        if self._json_matcher.is_terminated():
            self.mode = self.MODE_FREE
