"""Dynamic entity grounding for ``<vadalog>`` (Phase 2b / Challenge 2a).

Grammar-level alternation (Phase 2) does not scale to a 100k-entity KG.  This
module grounds the *string constants* of a Vadalog block at decode time instead,
so the entity set can be arbitrarily large:

* a **character trie** of the entity names tracks the content of the string
  currently being typed;
* a **token trie** of the vocabulary maps character-level constraints back to
  token ids — so it is token-boundary safe (a multi-char token is allowed iff
  its whole text stays a valid entity prefix);
* the entity mask is intersected with the grammar mask by *adding* the two
  (both use ``0`` = allowed, ``-inf`` = disallowed).

Only string constants are grounded; predicates/variables are handled by the
grammar (Phase 2).
"""
from __future__ import annotations

import math

import torch

_VOCAB_CACHE = {}


class CharTrie:
    __slots__ = ("children", "end")

    def __init__(self):
        self.children = {}
        self.end = False

    @classmethod
    def from_words(cls, words):
        root = cls()
        for word in words:
            node = root
            for ch in word:
                node = node.children.setdefault(ch, cls())
            node.end = True
        return root

    def node_for(self, prefix):
        node = self
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node


class TokenTrie:
    __slots__ = ("children", "token_id")

    def __init__(self):
        self.children = {}
        self.token_id = None

    @classmethod
    def from_texts(cls, texts):
        root = cls()
        for tid, text in enumerate(texts):
            if not text:
                continue
            node = root
            for ch in text:
                child = node.children.get(ch)
                if child is None:
                    child = cls()
                    node.children[ch] = child
                node = child
            node.token_id = tid
        return root


def _raw_tokenizer(tokenizer):
    from transformers import ProcessorMixin
    return tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer


def vocab_trie(tokenizer):
    """(TokenTrie, token_texts, [(id, text) for quote tokens]) cached per tokenizer."""
    raw = _raw_tokenizer(tokenizer)
    key = (getattr(raw, "name_or_path", None) or type(raw).__name__, len(raw))
    if key not in _VOCAB_CACHE:
        texts = [raw.decode([i]) for i in range(len(raw))]
        quote_items = [(i, t) for i, t in enumerate(texts) if '"' in t]
        _VOCAB_CACHE[key] = (TokenTrie.from_texts(texts), texts, quote_items)
    return _VOCAB_CACHE[key]


class EntityGrounder:
    """Restricts string constants to a (possibly very large) entity set."""

    def __init__(self, tokenizer, entities):
        self.trie = CharTrie.from_words(entities)
        self.token_trie, self.token_texts, self.quote_items = vocab_trie(tokenizer)
        self.vocab_size = len(self.token_texts)
        # Quote tokens that cannot start a valid string from outside any string
        # are static; precompute them once.
        self._out_invalid = [
            tid for tid, text in self.quote_items
            if self._simulate(False, "", text) is None
        ]

    # -- scanning ----------------------------------------------------------
    @staticmethod
    def in_string(text):
        """Return ``(inside, content)`` for the string the cursor is in."""
        inside, content, i = False, "", 0
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                if inside:
                    content += text[i:i + 2]
                i += 2
                continue
            if ch == '"':
                inside = not inside
                content = ""
                i += 1
                continue
            if inside:
                content += ch
            i += 1
        return inside, content

    def _simulate(self, inside, content, token_text):
        """Whether ``token_text`` can be appended; return the new state or None."""
        i = 0
        while i < len(token_text):
            ch = token_text[i]
            if ch == "\\" and i + 1 < len(token_text):
                if inside:
                    content += token_text[i:i + 2]
                    if self.trie.node_for(content) is None:
                        return None
                i += 2
                continue
            if ch == '"':
                if inside:
                    node = self.trie.node_for(content)
                    if node is None or not node.end:
                        return None
                    inside, content = False, ""
                else:
                    inside, content = True, ""
                i += 1
                continue
            if inside:
                content += ch
                if self.trie.node_for(content) is None:
                    return None
            i += 1
        return inside, content

    # -- token sets --------------------------------------------------------
    def _content_tokens(self, node):
        out = []

        def rec(lex_node, tok_node):
            if tok_node.token_id is not None:
                out.append(tok_node.token_id)
            for ch, child in tok_node.children.items():
                lex_child = lex_node.children.get(ch)
                if lex_child is not None:
                    rec(lex_child, child)

        rec(node, self.token_trie)
        return out

    def restrict(self, mask, mask_idx, text):
        """Intersect the mask row with the entity constraint."""
        inside, content = self.in_string(text)

        if not inside:
            if self._out_invalid:
                idx = torch.tensor(self._out_invalid, device=mask.device)
                mask[mask_idx, idx] = -math.inf
            return

        allowed = set()
        node = self.trie.node_for(content)
        if node is not None:
            allowed.update(self._content_tokens(node))
        for tid, token_text in self.quote_items:
            if self._simulate(True, content, token_text) is not None:
                allowed.add(tid)

        row = torch.full((mask.shape[-1],), -math.inf, device=mask.device,
                         dtype=mask.dtype)
        if allowed:
            row[torch.tensor(sorted(allowed), device=mask.device)] = 0
        mask[mask_idx, :] += row
