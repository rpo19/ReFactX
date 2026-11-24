"""Convert KQA Pro 27B teacher logs into ReFactX SFT traces.

The teacher log is a normal ``utils/eval.py`` JSONL log.  This builder keeps
strictly correct traces, rebuilds the prompt with the KQA Pro template, repairs
malformed tool calls, appends the chat end token, and attaches KQA Pro program
metadata used for the question-type study (count / verify / multi-hop /
qualifier).

Example::

    python -m utils.build_kqa_sft \
        --log logs/kqa_pro_train_eval_qwen38_27b.out \
        --output logs/kqa_pro_sft.jsonl \
        --prompt prompts/prompt_kqa_pro_teacher.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

import refactx
from utils.build_clean_sft import repair_trace

# Keep classification identical to utils/sample_kqa_pro.py.
from utils.sample_kqa_pro import program_flags, primary_type, answer_type


def load_log(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if line_no == 0 and "evaluation" not in obj:
                continue
            if "evaluation" not in obj:
                continue
            records.append(obj)
    return records


def strict_correct(record: dict) -> bool:
    return bool((record.get("evaluation") or {}).get("correct") == 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--prompt", default="prompts/prompt_kqa_pro_teacher.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                        help="Student tokenizer used to build the prompt")
    parser.add_argument("--max-tokens", type=int, default=1280,
                        help="Drop traces above this prompt+trace length")
    parser.add_argument("--mask-mode", default="none",
                        choices=["none", "objects", "triples"])
    parser.add_argument("--count-only", action="store_true",
                        help="Keep only questions with a Count program step")
    parser.add_argument("--primary-type", action="append", default=None,
                        help="Keep only these primary types (repeatable)")
    parser.add_argument("--max-per-type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    template = refactx.load_prompt(args.prompt)

    candidates = []
    skipped = collections.Counter()
    for log_path in args.log:
        for record in load_log(log_path):
            if not strict_correct(record):
                skipped["not_strict_correct"] += 1
                continue
            sample = record.get("input_sample") or {}
            program = sample.get("program") or []
            flags = program_flags(program)
            kind = primary_type(flags)
            if args.count_only and not flags["has_count"]:
                skipped["not_count"] += 1
                continue
            if args.primary_type and kind not in set(args.primary_type):
                skipped["wrong_primary_type"] += 1
                continue

            trace = repair_trace(record.get("full_prediction", "").rstrip())
            if "<answer>" not in trace:
                skipped["no_answer"] += 1
                continue
            if not trace.endswith("<|im_end|>"):
                trace += "<|im_end|>"

            question = record.get("question") or sample.get("question")
            prompt = refactx.apply_prompt_template(
                tokenizer, template, question, enable_thinking=False)
            length = (
                len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
                + len(tokenizer(trace, add_special_tokens=False)["input_ids"])
            )
            if length > args.max_tokens:
                skipped["too_long"] += 1
                continue

            answers = record.get("gt_answer", sample.get("answer"))
            candidates.append({
                "question": question,
                "gt_answer": answers,
                "prediction": record.get("prediction"),
                "full_prediction": trace,
                "prompt": prompt,
                "triples": record.get("triples") or [],
                "dataset": "kqa_pro",
                "dataset_split": record.get("dataset_split", "train"),
                "evaluation": record.get("evaluation") or {},
                "question_type": kind,
                "complexityType": kind,
                "answer_shape": answer_type(sample.get("answer")),
                "behavior": "supported",
                "provenance": "kqa_pro_teacher",
                "source_log": log_path.name,
                "synthetic": False,
                "mask_mode": args.mask_mode,
                "kqa_program_functions": [
                    step.get("function") for step in program
                ],
                "kqa_flags": flags,
                "has_count": flags["has_count"],
                "has_verify": flags["has_verify"],
                "has_qualifier": flags["has_qualifier"],
                "num_relate": flags["num_relate"],
            })

    if args.max_per_type:
        grouped = collections.defaultdict(list)
        for row in candidates:
            grouped[row["question_type"]].append(row)
        rng = random.Random(args.seed)
        capped = []
        for rows in grouped.values():
            rng.shuffle(rows)
            capped.extend(rows[:args.max_per_type])
        candidates = capped
    else:
        random.Random(args.seed).shuffle(candidates)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in candidates:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_type = collections.Counter(row["question_type"] for row in candidates)
    by_answer = collections.Counter(row["answer_shape"] for row in candidates)
    report = {
        "written": len(candidates),
        "skipped": dict(skipped),
        "by_type": dict(by_type),
        "by_answer_shape": dict(by_answer),
        "logs": [path.name for path in args.log],
        "mask_mode": args.mask_mode,
        "max_tokens": args.max_tokens,
    }
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
