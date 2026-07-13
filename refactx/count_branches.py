"""
Count branches LogitsProcessor.

Detects the ``count_branches(<prefix>)`` tool call in the generated text,
traverses the KB prefix tree to count leaves under that prefix, and forces
emission of `` = <count>`` so the model receives the answer inline.

States:
  COLLECTING_PREFIX – between the opening ``(`` and closing ``)``
  EMITTING_COUNT    – forcing the `` = <count>`` tokens one by one
"""
import math
from typing import Optional

import torch
from transformers import LogitsProcessor

from refactx.index import DictIndex


class CountBranchesLogitsProcessor(LogitsProcessor):
    """A ``LogitsProcessor`` that handles ``count_branches(<prefix>)`` tool calls.

    The processor monitors the generated token stream for the pattern
    ``count_branches(``.  Once detected it collects tokens until the closing
    ``)``, counts the leaves at the matched prefix in the KB trie, and then
    forces the output of `` = <count>`` tokens.

    Parameters
    ----------
    tokenizer : tokenizer
        HuggingFace tokenizer (or processor) used for encode/decode.
    kb_index : DictIndex
        The knowledge-base prefix tree used for counting.
    trigger : str
        The trigger pattern that activates collection (default ``"count_branches("``).
    regex_window : int
        Number of recent tokens to inspect when looking for the trigger.
    ignore_case : bool
        Case-insensitive matching for the trigger.
    """

    # FSM states
    MODE_FREE = 0
    MODE_COLLECTING = 1
    MODE_EMITTING = 2

    def __init__(
        self,
        tokenizer,
        kb_index: DictIndex,
        trigger: str = "count_branches(",
        regex_window: int = 20,
        ignore_case: bool = True,
    ):
        self.tokenizer = tokenizer
        self.kb_index = kb_index
        self.trigger = trigger.lower() if ignore_case else trigger
        self.regex_window = regex_window
        self.ignore_case = ignore_case

        self.mode = self.MODE_FREE
        self.prefix_tokens: list[int] = []
        self.count_tokens: list[int] = []
        self.input_ids: list[int] = []
        self._first_call = True

        # Persistent log of all calls: list of (prefix_text, count)
        self.calls: list[tuple[str, int]] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids)

    def _count_at_prefix(self, prefix_token_ids: list[int]) -> int:
        """Return the number of leaves under *prefix_token_ids* in the trie."""
        try:
            next_tokens, _ = self.kb_index.next_tokens(prefix_token_ids)
        except Exception:
            return 0
        if not next_tokens:
            return 0
        return sum(next_tokens.values())

    # ------------------------------------------------------------------
    # core
    # ------------------------------------------------------------------

    def _check_trigger(self) -> bool:
        text = self._decode(self.input_ids[-self.regex_window:])
        if self.ignore_case:
            text = text.lower()
        normalized = "".join(text.split())
        return normalized.rstrip().endswith(self.trigger)

    def _begin_collecting(self):
        self.mode = self.MODE_COLLECTING
        self.prefix_tokens = []

    def _finish_collecting(self):
        count = self._count_at_prefix(self.prefix_tokens)
        prefix_text = self._decode(self.prefix_tokens).strip()
        self.calls.append((prefix_text, count))
        count_str = f" = {count}"
        self.count_tokens = self._encode(count_str)
        self.mode = self.MODE_EMITTING

    def _emit_count(self, scores, batch_idx):
        if self.count_tokens:
            nxt = self.count_tokens.pop(0)
            scores[batch_idx] = -math.inf
            if 0 <= nxt < scores.shape[-1]:
                scores[batch_idx, nxt] = 0
            if not self.count_tokens:
                self.mode = self.MODE_FREE
        else:
            self.mode = self.MODE_FREE

    # ------------------------------------------------------------------
    # LogitsProcessor interface
    # ------------------------------------------------------------------

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
                # sequence diverged (e.g. beam search reordering) – reset
                self.mode = self.MODE_FREE
                self.prefix_tokens = []
                self.count_tokens = []
                self.input_ids = seq
                continue

            self.input_ids = seq

            if self.mode == self.MODE_FREE:
                if self._check_trigger():
                    self._begin_collecting()

            elif self.mode == self.MODE_COLLECTING:
                if last_token == self._encode(")")[0]:
                    self._finish_collecting()
                    self._emit_count(scores, i)
                else:
                    self.prefix_tokens.append(last_token)

            elif self.mode == self.MODE_EMITTING:
                self._emit_count(scores, i)

        return scores
