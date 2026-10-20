"""Explore a ReFactX index/trie (used by ``utils/logs_viewer.py``).

Supports the same sources as :func:`refactx.load_index`: a text file of
verbilized triples (one per line), a list of such lines, or a PostgreSQL index
URL.  Once loaded it answers three questions:

* **count** how many triples start with a prefix;
* **next tokens** the immediate continuations (with counts);
* **enumerate** the triples under a prefix (in-memory :class:`DictIndex` only).

A prefix can be given as text (tokenized with the index tokenizer) or as a raw
token-id list.
"""
from __future__ import annotations

import json

from refactx import load_index
from refactx.index import DictIndex, EmptyIndexException, TripleNotFoundException


# ---------------------------------------------------------------------------
# Tokenizer / token helpers
# ---------------------------------------------------------------------------

def get_tokenizer(model_name):
    from transformers import AutoProcessor, AutoTokenizer, ProcessorMixin
    try:
        tok = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        tok = AutoProcessor.from_pretrained(model_name).tokenizer
    return tok


def raw_tokenizer(tokenizer):
    """The underlying HF tokenizer when ``tokenizer`` is a processor."""
    from transformers import ProcessorMixin
    return tokenizer.tokenizer if isinstance(tokenizer, ProcessorMixin) else tokenizer


def parse_tokens(text, token_list_json, tokenizer, add_special_tokens=False):
    """Return the token ids for a query prefix.

    If ``token_list_json`` is a non-empty string it is parsed as a JSON list of
    ints (the "list of tokens" input); otherwise ``text`` is tokenized.
    """
    if token_list_json and token_list_json.strip():
        tokens = json.loads(token_list_json)
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise ValueError("token list must be a JSON array of integers")
        return tokens
    raw = raw_tokenizer(tokenizer)
    return list(raw(text, add_special_tokens=add_special_tokens)["input_ids"])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_lines_index(lines, tokenizer, add_special_tokens=False, batch_size=100):
    lines = [line for line in lines if line.strip()]
    return load_index(lines, tokenizer=tokenizer,
                      add_special_tokens=add_special_tokens, batch_size=batch_size)


def load_file_index(path, tokenizer, add_special_tokens=False, batch_size=100):
    return load_index(str(path), tokenizer=tokenizer,
                      add_special_tokens=add_special_tokens, batch_size=batch_size)


def load_postgres_index(url, configkey=-200, cache="simple"):
    cache = None if cache and str(cache).lower() == "none" else cache
    return load_index(url, configkey=configkey, cache=cache)


# ---------------------------------------------------------------------------
# DictIndex tree utilities
# ---------------------------------------------------------------------------

def _navigate(index, tokens):
    """Return the trie sub-node positioned right after ``tokens`` (or ``None``).

    The returned node keeps the ``[count, items]`` shape, with ``items``
    trimmed so it starts *after* the consumed prefix (so enumeration does not
    repeat the prefix).
    """
    level, cursor, level_cursor = index.tree, 0, 0
    while cursor < len(tokens) and level_cursor < len(level[1]):
        item = level[1][level_cursor]
        if isinstance(item, dict):
            if tokens[cursor] not in item:
                return None
            level = item[tokens[cursor]]
            level_cursor = 0
        else:
            if tokens[cursor] != item:
                return None
            level_cursor += 1
        cursor += 1
    if cursor < len(tokens):
        return None
    return [level[0], level[1][level_cursor:]]


def _iter_leaves(node, prefix=()):
    """Yield token sequences (as tuples) for every leaf under ``node``."""
    items = node[1]
    ints = []
    i = 0
    while i < len(items) and not isinstance(items[i], dict):
        ints.append(items[i])
        i += 1
    base = tuple(prefix) + tuple(ints)
    if i == len(items):
        yield base
        return
    for token, child in items[i].items():
        yield from _iter_leaves(child, base + (token,))


def prefix_count(index, tokens):
    node = _navigate(index, tokens)
    return 0 if node is None else node[0]


def next_token_counts(index, tokens):
    try:
        counts, _ = index.next_tokens(list(tokens))
        return counts
    except (EmptyIndexException, TripleNotFoundException):
        return {}


def enumerate_prefix(index, tokens, limit=200):
    """All ``tokens + continuation`` sequences under the prefix (bounded)."""
    node = _navigate(index, tokens)
    if node is None:
        return []
    out = []
    for seq in _iter_leaves(node):
        out.append(list(tokens) + list(seq))
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------
# High-level report
# ---------------------------------------------------------------------------

def _decode_next(counts, raw):
    rows = []
    for token, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        text = raw.decode([token]) if raw is not None else str(token)
        rows.append({"token": token, "decoded": text, "count": count})
    return rows


def prefix_report(index, tokens, tokenizer, limit=200):
    """Count / next tokens / enumerated triples for a prefix."""
    tokens = list(tokens)
    raw = raw_tokenizer(tokenizer)
    report = {
        "tokens": tokens,
        "decoded": raw.decode(tokens) if raw is not None else str(tokens),
        "count": None,
        "next": [],
        "triples": [],
        "enumerable": False,
        "note": None,
    }

    if isinstance(index, DictIndex):
        node = _navigate(index, tokens)
        if node is None:
            report["note"] = "prefix not found in the index"
            return report
        report["count"] = node[0]
        report["next"] = _decode_next(next_token_counts(index, tokens), raw)
        report["enumerable"] = True
        seqs = enumerate_prefix(index, tokens, limit=limit)
        report["triples"] = [(s, raw.decode(s) if raw is not None else str(s)) for s in seqs]
        report["truncated"] = report["count"] is not None and len(seqs) < report["count"]
        return report

    # PostgreSQL-backed index (and friends): only next-token lookup.
    from refactx.generate import PatternConstrainedState
    state = PatternConstrainedState(
        tokenizer=tokenizer, cache_index=DictIndex(), subtree_cache=DictIndex())
    try:
        counts, _ = index.next_tokens(tokens, state=state)
    except Exception as exc:  # pragma: no cover - depends on live backend
        report["note"] = f"query failed: {exc}"
        return report
    report["next"] = _decode_next(counts, raw)
    report["count"] = sum(counts.values())
    report["note"] = "enumeration is only supported for in-memory (text/list) indexes"
    return report
