"""From a character-level constraint to the set of allowed *tokens*.

The vocabulary is decoded once and sorted, so every token that starts with a
given prefix lives in one contiguous range.  ``allowed`` walks the constraint
and the sorted vocabulary together (an implicit trie over token spellings):
a character that leaves the constraint prunes, in one step, every token that
starts with that prefix.  A token is allowed when its whole spelling is
consumed and the rule is still completable.
"""
from __future__ import annotations

from bisect import bisect_left

_CACHE = {}


def raw_tokenizer(tokenizer):
    return getattr(tokenizer, "tokenizer", tokenizer)


class SortedVocab:
    def __init__(self, tokenizer):
        tok = raw_tokenizer(tokenizer)
        special = set(tok.all_special_ids)
        spellings = tok.batch_decode([[i] for i in range(len(tok))],
                                     clean_up_tokenization_spaces=False)
        pairs = sorted((s, i) for i, s in enumerate(spellings)
                       if s and i not in special and "�" not in s)
        self.strings = [s for s, _ in pairs]
        self.ids = [i for _, i in pairs]
        self.spelling = dict(enumerate(spellings))

    @classmethod
    def for_tokenizer(cls, tokenizer):
        tok = raw_tokenizer(tokenizer)
        key = (getattr(tok, "name_or_path", None), len(tok), id(tok))
        if key not in _CACHE:
            _CACHE[key] = cls(tokenizer)
        return _CACHE[key]

    def allowed(self, constraint, subset, cc_cache=None, limit=None):
        """Token ids whose spelling keeps ``subset`` completable."""
        out = []
        cc_cache = {} if cc_cache is None else cc_cache
        self._walk(constraint, subset, 0, len(self.strings), "", out, cc_cache)
        if limit is not None:
            out = [i for i in out if i < limit]
        return out

    def _walk(self, c, subset, lo, hi, prefix, out, cc):
        strings, depth = self.strings, len(prefix)
        if depth:
            # tokens spelled exactly `prefix` sort first in [lo, hi)
            ok = None
            while lo < hi and len(strings[lo]) == depth:
                if ok is None:
                    ok = c.can_complete(subset, cc)
                if ok:
                    out.append(self.ids[lo])
                lo += 1
        if lo >= hi:
            return
        for ch in c.next_chars(subset):
            a = bisect_left(strings, prefix + ch, lo, hi)
            b = bisect_left(strings, prefix + chr(ord(ch) + 1), a, hi)
            if a >= b:
                continue
            nxt = c.step(subset, ch)
            if nxt:
                self._walk(c, nxt, a, b, prefix + ch, out, cc)
