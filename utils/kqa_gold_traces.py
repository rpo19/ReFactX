"""Deterministic retrieval traces from gold KoPL programs (KQA Pro).

For each question whose program uses only a simple chain of operators
(Find, Relate, QueryRelation, QueryAttr, QueryName, Count), execute the
program over the local KB (`/workspace/data/kqa_pro/kb.json`), emit the
tool calls a retriever would make, and keep the trace only if the computed
answer equals the gold answer. No API or model calls are involved.

Trace syntax matches the teacher prompt (`prompts/prompt_kqa_pro_teacher.yaml`):
    <fact> <Subject> <relation> <Object> . </fact>
    <count> <Subject> <relation>\n= N\n</count>
    <answer>\n["..."]\n</answer>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

SUPPORTED = {"Find", "Relate", "QueryRelation", "QueryAttr", "QueryName", "Count"}


class Unsupported(Exception):
    pass


class KB:
    def __init__(self, path):
        raw = json.load(open(path, encoding="utf-8"))
        self.ents = raw["entities"]
        self.by_name = collections.defaultdict(list)
        for qid, ent in self.ents.items():
            self.by_name[ent["name"]].append(qid)

    def name(self, qid):
        return self.ents[qid]["name"]

    def label(self, qid):
        """Entity label as it appears in the index: name, or 'name (QID)' when ambiguous."""
        name = self.ents[qid]["name"]
        return name if len(self.by_name[name]) == 1 else f"{name} ({qid})"

    def find(self, name):
        ids = self.by_name.get(name, [])
        if len(ids) != 1:
            raise Unsupported("ambiguous_or_missing_entity")
        return ids[0]

    def edges(self, qid, predicate):
        """Yield (direction, other_qid) for edges of `qid` with predicate.

        Forward means qid --pred--> other; backward means other --pred--> qid.
        """
        for rel in self.ents[qid]["relations"]:
            if rel["predicate"] == predicate:
                yield rel["direction"], rel["object"]

    def attrs(self, qid, key):
        return [a["value"] for a in self.ents[qid]["attributes"] if a["key"] == key]


def fmt_value(v):
    if isinstance(v, dict):
        if v.get("type") == "quantity":
            return f"{v['value']} {v.get('unit', '1')}".strip() if v.get("unit", "1") != "1" else str(v["value"])
        return str(v.get("value"))
    return str(v)


def check_chain(program):
    """Accept only a single linear chain: one Find, then each step depends on the previous."""
    if not program or program[0]["function"] != "Find" or program[0]["dependencies"]:
        raise Unsupported("not_single_find_start")
    for i, step in enumerate(program[1:], start=1):
        if step["function"] == "Find":
            raise Unsupported("multiple_find")
        if step["dependencies"] != [i - 1]:
            raise Unsupported("non_linear")
        if step["function"] in ("Relate", "QueryRelation", "QueryAttr") and not step["inputs"]:
            raise Unsupported("implicit_relation")


def run_program(kb: KB, program):
    """Execute a simple chain. Returns (answer_list, calls) where calls are dicts."""
    check_chain(program)
    state = None  # current set of qids, or list of values
    kind = None   # "ent" or "val"
    ctx = None    # (subject, relation) of the last forward relation step
    calls = []
    for step in program:
        fn = step["function"]
        if fn not in SUPPORTED:
            raise Unsupported(fn)
        if fn == "Find":
            state = [kb.find(step["inputs"][0])]
            kind = "ent"
            subject = state[0]
        elif fn == "Relate":
            if kind != "ent" or len(state) != 1:
                raise Unsupported("relate_on_set")
            pred, direction = step["inputs"][0], step["inputs"][1]
            subj = state[0]
            objs = []
            for d, other in kb.edges(subj, pred):
                if d != direction:
                    continue
                objs.append(other)
                if d == "forward":
                    calls.append({"s": subj, "r": pred, "o": other})
                else:
                    calls.append({"s": other, "r": pred, "o": subj})
            ctx = (subj, pred) if direction == "forward" else None
            state = objs
            kind = "ent"
        elif fn == "QueryRelation":
            if kind != "ent" or len(state) != 1:
                raise Unsupported("queryrelation_on_set")
            subj = state[0]
            pred = step["inputs"][0]
            objs = [o for d, o in kb.edges(subj, pred) if d == "forward"]
            for o in objs:
                calls.append({"s": subj, "r": pred, "o": o})
            ctx = (subj, pred)
            state = objs
            kind = "ent"
        elif fn == "QueryAttr":
            if kind != "ent" or len(state) != 1:
                raise Unsupported("queryattr_on_set")
            subj = state[0]
            key = step["inputs"][0]
            vals = kb.attrs(subj, key)
            calls.append({"s": subj, "r": key, "attr_values": vals})
            state = [fmt_value(v) for v in vals]
            ctx = None
            kind = "val"
        elif fn == "QueryName":
            if kind != "ent":
                raise Unsupported("queryname_on_values")
            state = [kb.name(q) for q in state]
            kind = "val"
        elif fn == "Count":
            # Only counts directly after a forward relation of a single subject.
            if ctx is None:
                raise Unsupported("count_without_forward_relation")
            n = len(state)
            calls.append({"count": True, "s": ctx[0], "r": ctx[1], "n": n})
            state = [str(n)]
            kind = "val"
        else:
            raise Unsupported(fn)
    return state, calls


def norm(s):
    return str(s).strip().lower()


def build_trace(kb: KB, program, gold):
    """Return (trace_text, answer_list) or raise Unsupported."""
    answer, calls = run_program(kb, program)
    gold_list = gold if isinstance(gold, list) else [gold]
    if not calls:
        raise Unsupported("no_calls")
    if {norm(a) for a in answer} != {norm(g) for g in gold_list}:
        raise Unsupported("answer_mismatch")
    lines = []
    for c in calls:
        if c.get("count"):
            lines.append(f"<count> <{kb.label(c['s'])}> <{c['r']}>\n= {c['n']}\n</count>")
        elif "attr_values" in c:
            for v in c["attr_values"]:
                lines.append(f"<fact> <{kb.label(c['s'])}> <{c['r']}> <{fmt_value(v)}> . </fact>")
        else:
            lines.append(f"<fact> <{kb.label(c['s'])}> <{c['r']}> <{kb.label(c['o'])}> . </fact>")
    answer_block = "<answer>\n" + json.dumps(answer, ensure_ascii=False) + "\n</answer>"
    return "\n".join(lines) + "\n" + answer_block + "<|im_end|>", answer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train")
    ap.add_argument("--kb", default="/workspace/data/kqa_pro/kb.json")
    ap.add_argument("--data", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = args.data or f"/workspace/data/kqa_pro/{args.split}.jsonl"
    kb = KB(args.kb)
    reasons = collections.Counter()
    kept = 0
    with open(data, encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as out:
        for i, line in enumerate(src):
            if args.limit and i >= args.limit:
                break
            row = json.loads(line)
            try:
                trace, answer = build_trace(kb, row["program"], row["answer"])
            except Unsupported as exc:
                reasons[str(exc)] += 1
                continue
            except Exception as exc:
                reasons["error:" + type(exc).__name__] += 1
                continue
            out.write(json.dumps({"id": row["id"], "question": row["question"],
                                  "gt_answer": answer, "full_prediction": trace,
                                  "dataset": "kqa_pro", "trace_source": "gold_program"},
                                 ensure_ascii=False) + "\n")
            kept += 1
    total = i + 1
    print(f"kept {kept}/{total}")
    print(reasons.most_common())


if __name__ == "__main__":
    main()
