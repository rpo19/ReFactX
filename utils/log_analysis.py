"""Parse and analyse ReFactX evaluation logs (used by ``utils/logs_viewer.py``).

A log is line-delimited JSON: a metadata header, one record per question, and a
summary line.  This module turns it into a :class:`pandas.DataFrame` and adds a
few basic error-analysis columns, notably:

* ``hit_max_tokens`` — whether generation ran out of the token budget;
* ``triple_mismatch`` / ``extra_triples`` — triples written in the text (found
  with a regex) that are not among the constrained ``triples`` recorded for the
  record, i.e. likely hallucinations.
"""
from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

SENTINEL = "no further records"
METRIC_KEYS = ("precision", "recall", "f1", "correct", "dont_know")

# ``<subject> <relation> <object> .`` — the object may contain spaces (e.g. the
# sentinel ``<no further records>``), so we only exclude the delimiters/newlines.
TRIPLE_RE = re.compile(r"<([^<>\n]+)>\s*<([^<>\n]+)>\s*<([^<>\n]+)>\s*\.")


def open_log(path):
    path = Path(path)
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") \
        if path.suffix == ".gz" else open(path, encoding="utf-8", errors="replace")


def load_log(path):
    """Return ``(metadata, records, summary)`` from a log file."""
    parsed = []
    with open_log(path) as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    if not parsed:
        return {}, [], {}
    metadata = parsed[0] if not isinstance(parsed[0], dict) or "evaluation" not in parsed[0] else {}
    start = 1 if metadata else 0
    summary = parsed[-1] if isinstance(parsed[-1], dict) and "macro" in parsed[-1] else {}
    end = len(parsed) - 1 if summary else len(parsed)
    records = [r for r in parsed[start:end] if isinstance(r, dict) and "evaluation" in r]
    return metadata, records, summary


def extract_triples(text):
    """All ``<s> <r> <o> .`` matches in ``text`` (order preserved)."""
    return [m.group(0) for m in TRIPLE_RE.finditer(text or "")]


def normalize_triple(text):
    """Collapse whitespace and drop a trailing period, for comparison."""
    return " ".join(text.split()).rstrip(".").strip()


def record_triples(record):
    """The constrained triples of a record, normalised for comparison.

    The recorded strings end with the ``</kg>`` tag (and often the newline), so
    they are re-parsed with the same regex used on the free text.
    """
    out = []
    for raw in record.get("triples", []) or []:
        matches = extract_triples(raw)
        if matches:
            out.append(normalize_triple(matches[0]))
        else:
            out.append(normalize_triple(raw))
    return out


def text_triples(record):
    """Triples appearing anywhere in ``full_prediction``, normalised."""
    return [normalize_triple(t) for t in extract_triples(record.get("full_prediction", ""))]


def triple_diff(record):
    """Return ``(extra, missing)`` counters.

    ``extra`` are triple *occurrences* in the text beyond what the constrained
    generator produced (this includes triples the model merely quotes again);
    ``missing`` are generated triples that did not make it into the text.
    """
    return (Counter(text_triples(record)) - Counter(record_triples(record)),
            Counter(record_triples(record)) - Counter(text_triples(record)))


def hallucinated_triples(record):
    """Distinct triples written in the text that the generator never produced.

    The regex looks everywhere, regardless of ``<kg>`` tags.  Triples equal to a
    generated one are discounted (that is how re-quotes are excluded).  The last
    ``incomplete_triples`` unconsumed occurrences are dropped too: those are
    lookups whose terminal was cut off (e.g. ``max_new_tokens``) before
    ``cache_add`` recorded them, and they always sit at the end of the text.
    """
    known = set(record_triples(record))
    found = text_triples(record)
    incomplete = int(record.get("incomplete_triples", 0) or 0)

    # Keep occurrence order so the truncated lookups (always last) can be
    # dropped precisely from the not-generated candidates.
    candidates = [t for t in found if t not in known]
    if incomplete > 0:
        candidates = candidates[:max(0, len(candidates) - incomplete)]
    return sorted(set(candidates))


def _estimate_tokens(text):
    """Rough token count (≈4 chars/token) when the log has no token field."""
    return len(text) / 4.0


def records_frame(records, metadata=None):
    """Build the analysis DataFrame for a list of records."""
    metadata = metadata or {}
    budget = metadata.get("max_new_tokens")
    rows = []
    for i, r in enumerate(records):
        text = r.get("full_prediction", "") or ""
        ev = r.get("evaluation", {}) or {}
        key_triples = record_triples(r)
        text_triples_ = text_triples(r)
        extra, missing = triple_diff(r)
        unknown = hallucinated_triples(r)

        hit_max = r.get("hit_max_tokens")
        n_new_tokens = r.get("n_new_tokens")
        est = hit_max is None
        if est and budget:
            # No recorded flag: guess from the length against the budget.
            hit_max = int(_estimate_tokens(text) >= budget * 0.95)

        rows.append({
            "idx": i,
            "kind": r.get("kind"),
            "correct": float(ev.get("correct", 0.0)),
            "f1": float(ev.get("f1", 0.0)),
            "answered": int(bool(r.get("prediction"))),
            "sentinel": int(SENTINEL in text),
            "n_sentinel": text.count(SENTINEL),
            "n_triples": len(key_triples),
            "n_triples_regex": len(text_triples_),
            "incomplete_triples": int(r.get("incomplete_triples", 0) or 0),
            "triple_mismatch": int(len(key_triples) != len(text_triples_)),
            "extra_triples": sum(extra.values()),
            "missing_triples": sum(missing.values()),
            "unknown_triples": len(unknown),
            "hit_max_tokens": int(hit_max) if hit_max is not None else None,
            "hit_max_is_estimate": est,
            "n_new_tokens": n_new_tokens,
            "question": r.get("question", ""),
            "prediction": r.get("prediction"),
        })
    return pd.DataFrame(rows)


def analyze(records, metadata=None):
    """Return ``(df, stats)`` with a few aggregate error-analysis numbers."""
    df = records_frame(records, metadata)
    stats = {"n": len(df)}
    if df.empty:
        return df, stats
    stats["correct"] = df["correct"].mean()
    stats["f1"] = df["f1"].mean()
    stats["answered"] = df["answered"].mean()
    stats["sentinel_rate"] = df["sentinel"].mean()
    stats["max_sentinels"] = int(df["n_sentinel"].max())
    stats["incomplete_triples"] = int(df["incomplete_triples"].sum())
    stats["hit_max_tokens"] = int(df["hit_max_tokens"].fillna(0).sum())
    stats["hit_max_tokens_estimated"] = bool(df["hit_max_is_estimate"].any())
    stats["triple_mismatch"] = int(df["triple_mismatch"].sum())
    stats["unknown_triples"] = int(df["unknown_triples"].sum())
    stats["records_with_unknown"] = int((df["unknown_triples"] > 0).sum())
    stats["extra_triples"] = int(df["extra_triples"].sum())
    stats["missing_triples"] = int(df["missing_triples"].sum())
    return df, stats
