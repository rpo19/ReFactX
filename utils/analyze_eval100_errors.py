"""Classify failures in 100-question eval logs (eval100_*.out.0).

Categories (first match wins, per failed question):
- no_answer:        no <answer> produced, or "I don't know"
- retrieved_wrong:  gold answer appears in the retrieved triples, but the
                    final answer differs (selection / aggregation error)
- count_no_tool:    gold is a number and no <count> call was made
- missing_gold:     gold answer not in any retrieved triple. For KQA Pro the
                    gold program names the entity and relation; we also record
                    whether the entity was ever retrieved as a subject and
                    whether the relation was ever used.

Usage: python -m utils.analyze_eval100_errors logs/eval100_*.out.0 --out SFT_ERROR_100.md
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

TRIPLE_RE = re.compile(r"<([^<>]*)>\s*<([^<>]*)>\s*<([^<>]*)>")


def norm(s) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def load(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{") and '"evaluation"' in line and '"full_prediction"' in line:
            rows.append(json.loads(line))
        elif line.startswith("{") and '"input_sample"' in line:
            rows.append(json.loads(line))
    return rows


def parse_triples(triples):
    out = []
    for t in triples or []:
        m = TRIPLE_RE.search(str(t))
        if m:
            out.append(tuple(x.strip() for x in m.groups()))
    return out


def answer_of(full: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", full or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(1))
        return v if isinstance(v, list) else [v]
    except Exception:
        return [m.group(1)]


def kqa_program_terms(sample):
    """Entity and relation names from the gold KoPL program, if present."""
    prog = (sample or {}).get("program") or []
    ents = [s["inputs"][0] for s in prog if s.get("function") == "Find" and s.get("inputs")]
    rels = []
    for s in prog:
        if s.get("function") in ("Relate", "QueryRelation", "QueryAttr", "QueryAttrQualifier", "QueryRelationQualifier") and s.get("inputs"):
            rels.append(s["inputs"][0])
    return ents, rels


def classify(row):
    full = row.get("full_prediction") or ""
    gold = row.get("gt_answer") or []
    gold = gold if isinstance(gold, list) else [gold]
    pred = answer_of(full)
    triples = parse_triples(row.get("triples"))
    info = {"triples": len(triples)}
    if pred is None or all(norm(p) in ("", "i dont know") for p in pred):
        return "no_answer", info
    ev = row.get("evaluation") or {}
    if ev.get("correct", 0) == 1 or ev.get("f1", 0) == 1:
        return "correct", info
    gold_norm = {norm(g) for g in gold if norm(g)}
    triple_text = norm(" ".join(" ".join(t) for t in triples))
    if gold_norm and all(g in triple_text for g in gold_norm):
        return "retrieved_wrong", info
    if gold and all(re.fullmatch(r"\d+(\.\d+)?", str(g).strip()) for g in gold) and "<count>" not in full:
        return "count_no_tool", info
    ents, rels = kqa_program_terms(row.get("input_sample"))
    if ents:
        subj_hit = any(norm(ents[0]) in norm(t[0]) for t in triples)
        rel_hit = any(norm(r) in norm(t[1]) for t in triples for r in rels)
        info["entity_retrieved"] = subj_hit
        info["relation_used"] = rel_hit
    return "missing_gold", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    lines = ["# Error classes on 100-question eval logs", ""]
    for path in args.logs:
        rows = load(path)
        if not rows:
            continue
        counts = collections.Counter()
        kqa_sub = collections.Counter()
        for r in rows:
            cls, info = classify(r)
            counts[cls] += 1
            if cls == "missing_gold" and "entity_retrieved" in info:
                kqa_sub[f"entity_retrieved={info['entity_retrieved']}, relation_used={info['relation_used']}"] += 1
        n = len(rows)
        lines.append(f"## {path.name} (n={n})")
        lines.append("")
        lines.append("| class | count | share |")
        lines.append("|---|---:|---:|")
        for cls, c in counts.most_common():
            lines.append(f"| {cls} | {c} | {c / n:.0%} |")
        if kqa_sub:
            lines.append("")
            lines.append("KQA Pro missing_gold breakdown (from gold program):")
            for k, c in kqa_sub.most_common():
                lines.append(f"- {k}: {c}")
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
