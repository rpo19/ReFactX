"""Build answer-conditioned, KB-verified SFT traces from RoG datasets.

For each question the RoG ``graph`` field is searched for a directed path from a
question entity to each gold answer.  Every path edge is then mapped to a fact
that actually exists in the ReFactX prefix tree (``qwen36fbwikipops``) and the
trace is kept only when all answers are reachable.  This filters out questions
whose dataset labels disagree with the KB and produces traces grounded in the
same facts the constrained decoder can generate at inference time.

Example::

    python -m utils.build_path_sft \
        --dataset rmanluo/RoG-webqsp --output logs/path_webqsp.jsonl \
        --max-samples 1000 --num-shards 8 --shard 0
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict, deque

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

import refactx
from refactx.generate import PatternConstrainedState
from refactx.index import DictIndex
from utils.load_freebase_properties import simplify_property


def norm(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value).strip().lower())
    value = re.sub(r"^the ", "", value)
    return value.replace("m.", "m/")


class KBResolver:
    def __init__(self, index, tokenizer):
        self.index = index
        self.tokenizer = tokenizer
        self._path_cache = {}
        self._subject_cache = {}
        self._fact_cache = {}

    def _state(self):
        return PatternConstrainedState(
            tokenizer=self.tokenizer,
            cache_index=DictIndex(),
            subtree_cache=DictIndex(),
        )

    def next_tokens(self, ids, state):
        return self.index.next_tokens(ids, state=state)[0]

    def text_path(self, target: str):
        """Token ids whose decoded text equals *target*, using one coherent state."""
        if target in self._path_cache:
            return self._path_cache[target]
        state = self._state()
        ids, text, steps = [], "", 0
        while text != target:
            options = self.next_tokens(ids, state)
            matches = [
                token for token in options
                if target.startswith(text + self.tokenizer.decode([token]))
            ]
            if not matches or steps > 200:
                self._path_cache[target] = None
                return None
            token = max(matches, key=lambda t: len(self.tokenizer.decode([t])))
            ids.append(token)
            text += self.tokenizer.decode([token])
            steps += 1
        self._path_cache[target] = ids
        return ids

    def first_leaf(self, ids):
        state = self._state()
        options = {}
        for i in range(len(ids) + 1):
            options = self.next_tokens(ids[:i], state)
        current = ids
        while options:
            token = next(iter(options))
            current = current + [token]
            options = self.next_tokens(current, state)
        return self.tokenizer.decode(current)

    def leaves(self, ids, limit=80):
        state = self._state()
        options = {}
        for i in range(len(ids) + 1):
            options = self.next_tokens(ids[:i], state)
        out = []
        stack = [(ids, options)]
        while stack and len(out) < limit:
            current, options = stack.pop()
            if not options:
                out.append(self.tokenizer.decode(current))
                continue
            for token in options:
                nxt = current + [token]
                stack.append((nxt, self.next_tokens(nxt, state)))
        return out

    def subjects_for(self, name: str):
        if name in self._subject_cache:
            return self._subject_cache[name]
        prefix = self.text_path(f" <{name}")
        if prefix is None:
            self._subject_cache[name] = []
            return []
        out = []
        state = self._state()
        options = {}
        for i in range(len(prefix) + 1):
            options = self.next_tokens(prefix[:i], state)
        if any(self.tokenizer.decode([t]) == ">" for t in options):
            out.append(name)
        stack = [(prefix, options)]
        seen = 0
        while stack and len(out) < 12 and seen < 5000:
            current, options = stack.pop()
            seen += 1
            for token in options:
                nxt = current + [token]
                nxt_options = self.next_tokens(nxt, state)
                text = self.tokenizer.decode(nxt)
                if text.endswith(">"):
                    candidate = text.strip()[1:-1]
                    if (norm(candidate) == norm(name) or
                            norm(candidate).startswith(norm(name) + " (")):
                        if candidate not in out:
                            out.append(candidate)
                else:
                    stack.append((nxt, nxt_options))
        self._subject_cache[name] = out
        return out

    @staticmethod
    def parse_leaf(leaf: str):
        match = re.match(r"^ <(.*?)> <(.*?)> <(.*)> \.$", leaf)
        return match.groups() if match else None

    @staticmethod
    def object_matches(leaf: str, target: str):
        parsed = KBResolver.parse_leaf(leaf)
        if not parsed:
            return None
        obj = parsed[2]
        on, tn = norm(obj), norm(target)
        base = re.sub(r"\s*\(m/[^)]*\)$", "", on)
        if (tn == on or on.startswith(tn + " (") or on.startswith(tn + "(")
                or (tn.startswith("m/") and ("(" + tn + ")" in on))
                or base == tn):
            return parsed
        return None

    def find_fact(self, subject: str, predicate: str, target: str):
        key = (subject, predicate, str(target))
        if key in self._fact_cache:
            return self._fact_cache[key]
        simple = simplify_property(predicate)
        for relation in (simple, f"{simple} ({predicate})"):
            rel_prefix = self.text_path(f" <{subject}> <{relation}> <")
            if rel_prefix is None:
                continue
            candidates = [str(target)]
            if str(target).lower().startswith("the "):
                candidates.append(str(target)[4:])
            else:
                candidates.append("the " + str(target))
            candidates.append(str(target).replace("m.", "m/"))
            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                obj_prefix = self.text_path(
                    f" <{subject}> <{relation}> <{candidate}")
                if obj_prefix is None:
                    continue
                parsed = self.object_matches(self.first_leaf(obj_prefix), target)
                if parsed:
                    self._fact_cache[key] = parsed
                    return parsed
                for leaf in self.leaves(obj_prefix, limit=80):
                    parsed = self.object_matches(leaf, target)
                    if parsed:
                        self._fact_cache[key] = parsed
                        return parsed
        self._fact_cache[key] = None
        return None


def graph_path(graph, sources, targets, max_depth=4):
    adjacency = defaultdict(list)
    for subject, predicate, obj in graph:
        adjacency[norm(subject)].append((norm(obj), subject, predicate, obj))
    targets = {norm(t) for t in targets}
    queue = deque((norm(s), [], 0) for s in sources if s)
    seen = set()
    while queue:
        node, path, depth = queue.popleft()
        if node in targets and path:
            return path
        if depth >= max_depth:
            continue
        for nxt, subject, predicate, obj in adjacency.get(node, []):
            if (nxt, depth) in seen:
                continue
            seen.add((nxt, depth))
            queue.append((nxt, path + [(subject, predicate, obj)], depth + 1))
    return None


def build_sample(sample, resolver: KBResolver, max_depth=4):
    answers = sample.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    facts, seen = [], set()
    for answer in answers:
        path = graph_path(
            sample.get("graph") or [], sample.get("q_entity") or [],
            [answer], max_depth=max_depth)
        if not path:
            return None, "no_path"
        subjects = resolver.subjects_for(path[0][0])
        if not subjects:
            return None, "no_subject"
        verified = None
        for subject in subjects:
            current, local = subject, []
            ok = True
            for _graph_subject, predicate, obj in path:
                fact = resolver.find_fact(current, predicate, obj)
                if not fact:
                    ok = False
                    break
                local.append(fact)
                current = fact[2]
            if ok:
                verified = local
                break
        if verified is None:
            return None, "edge_not_in_kb"
        for fact in verified:
            if fact not in seen:
                seen.add(fact)
                facts.append(fact)
    return facts, "ok"


def render_trace(question, facts, answers):
    lines = [f"To answer \"{question}\" I will retrieve the relevant KB facts.\n"]
    for subject, relation, obj in facts:
        lines.append(f"<fact> <{subject}> <{relation}> <{obj}> . </fact>\n")
    lines.append("The retrieved facts support the answer.\n\n")
    lines.append("<answer>\n")
    lines.append(json.dumps(answers, ensure_ascii=False))
    lines.append("\n</answer>")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="prompts/prompt_compact.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--index-model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--table", default="qwen36fbwikipops")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum samples to process (per shard)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--mask-mode", default="none", choices=["none", "objects", "triples"])
    args = parser.parse_args()

    load_dotenv()
    base = os.environ.get("BASE_INDEX_PATH")
    if not base:
        raise SystemExit("BASE_INDEX_PATH is not set")

    student_tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    index_tokenizer = AutoTokenizer.from_pretrained(
        args.index_model, trust_remote_code=True)
    template = refactx.load_prompt(args.prompt)
    index = refactx.load_index(f"{base}?tablename={args.table}")
    index.set_tokenizer(index_tokenizer)
    resolver = KBResolver(index, index_tokenizer)

    dataset = load_dataset(args.dataset, split=args.split)
    indices = range(args.shard, len(dataset), args.num_shards)
    if args.max_samples is not None:
        indices = list(indices)[:args.max_samples]

    counts = defaultdict(int)
    written = 0
    started = time.time()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as destination:
        for position, index_i in enumerate(indices):
            sample = dataset[index_i]
            facts, reason = build_sample(sample, resolver, args.max_depth)
            counts[reason] += 1
            if not facts:
                continue
            answers = sample.get("answer") or []
            if isinstance(answers, str):
                answers = [answers]
            trace = render_trace(sample["question"], facts, answers) + "<|im_end|>"
            prompt = refactx.apply_prompt_template(
                student_tokenizer, template, sample["question"],
                enable_thinking=False)
            destination.write(json.dumps({
                "id": sample.get("id"),
                "dataset": "webqsp" if "webqsp" in args.dataset.lower() else "cwq",
                "dataset_split": args.split,
                "question": sample["question"],
                "gt_answer": answers,
                "prompt": prompt,
                "full_prediction": trace,
                "provenance": "path_verified",
                "behavior": "supported",
                "mask_mode": args.mask_mode,
            }, ensure_ascii=False) + "\n")
            written += 1
            if (position + 1) % 50 == 0:
                elapsed = time.time() - started
                print(
                    f"[shard {args.shard}] {position + 1}/{len(indices)} "
                    f"kept={written} {dict(counts)} "
                    f"{elapsed / (position + 1):.2f}s/sample", flush=True)
    print(f"wrote {written} traces to {args.output} ({dict(counts)})")


if __name__ == "__main__":
    main()
