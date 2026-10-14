"""Compiled Vadalog grammar for constrained ``<vadalog>`` decoding (Phase 1).

Wraps xgrammar: loads the EBNF in ``grammars/vadalog.ebnf``, assembles the
``root`` rule with the ``<vadalog> ... </vadalog>`` tags, compiles it once per
(vocab, tags) and hands out ``GrammarMatcher`` instances that constrain the
token distribution to syntactically valid Vadalog.

Grammar-constrained generation only enforces **structure** (Challenge 1).
Entity/relation/variable grounding (Challenge 2) is layered on top in later
phases — see ``VadalogPlan.md``.
"""
from __future__ import annotations

from pathlib import Path

import torch

_GRAMMAR_PATH = Path(__file__).resolve().parent / "grammars" / "vadalog.ebnf"
_compiled_cache = {}


def content_ebnf():
    return _GRAMMAR_PATH.read_text(encoding="utf-8")


def build_ebnf(open_text, close_text, content=None):
    """Assemble the ``root`` rule (with tags) around the content grammar."""
    content = content if content is not None else content_ebnf()
    return f'root ::= "{open_text}" ws program ws "{close_text}"\n' + content


def _raw_tokenizer(tokenizer):
    from transformers import ProcessorMixin
    return tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer


class VadalogGrammar:
    """A compiled Vadalog grammar bound to a tokenizer."""

    def __init__(self, tokenizer, open_text, close_text, ebnf=None):
        import xgrammar as xgr
        self._xgr = xgr
        self.open_text = open_text
        self.close_text = close_text
        raw = _raw_tokenizer(tokenizer)
        self.vocab_size = len(raw)
        self.compiled = _compile_cached(raw, open_text, close_text, ebnf)

    def new_matcher(self):
        """A ``GrammarMatcher`` positioned right after the opening tag."""
        matcher = self._xgr.GrammarMatcher(self.compiled)
        matcher.accept_string(self.open_text)
        return matcher

    def allocate_bitmask(self):
        return self._xgr.allocate_token_bitmask(1, self.vocab_size)

    def apply_mask(self, mask, mask_idx, matcher, bitmask):
        """Restrict ``mask[mask_idx]`` to the tokens the grammar allows next."""
        matcher.fill_next_token_bitmask(bitmask, 0)
        row = torch.zeros(1, self.vocab_size, device=mask.device, dtype=mask.dtype)
        self._xgr.apply_token_bitmask_inplace(row, bitmask.to(mask.device))
        mask[mask_idx, :] = row[0]


def _compile_cached(raw_tokenizer, open_text, close_text, ebnf):
    import xgrammar as xgr
    ebnf = ebnf if ebnf is not None else build_ebnf(open_text, close_text)
    key = (
        getattr(raw_tokenizer, "name_or_path", None) or type(raw_tokenizer).__name__,
        len(raw_tokenizer),
        open_text,
        close_text,
        hash(ebnf),
    )
    if key not in _compiled_cache:
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(raw_tokenizer)
        _compiled_cache[key] = xgr.GrammarCompiler(tokenizer_info).compile_grammar(ebnf)
    return _compiled_cache[key]
