"""Question-type analysis for KQA Pro teacher logs.

Groups results by KoPL program type (qualifier / count / verify / multi-hop /
base) and by answer type, and reports accuracy, truncation, tool use and token
usage.  This replaces Mintaka's ``complexityType`` study with the richer KQA Pro
program structure.

Example::

    python -m utils.analyze_kqa_types \
        --log logs/kqa_pro_train_eval_qwen38_27b.out \
        --output logs/kqa_pro_type_report.md
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from utils.sample_kqa_pro import answer_type, primary_type, program_flags


def load_log(paths):
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as source:
            for line_no, line in enumerate(source):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "evaluation" not in obj:
                    continue
                records.append(obj)
    return records


def metrics(records):
    n = len(records)
    correct = sum(int((r.get("evaluation") or {}).get("correct", 0)) for r in records)
    trunc = sum(int(r.get("reached_max_tokens", False)) for r in records)
    answered = sum(int(bool(r.get("prediction"))) for r in records)
    fact_calls = [str(r.get("full_prediction", "")).count("<fact>") + str(r.get("full_prediction", "")).count("<count>") for r in records]
    tokens = [r.get("new_tokens_generated", 0) for r in records]
    return {
        "n": n,
        "correct": correct,
        "acc": round(correct / n, 4) if n else 0.0,
        "trunc": round(trunc / n, 4) if n else 0.0,
        "answered": round(answered / n, 4) if n else 0.0,
        "avg_tool_calls": round(sum(fact_calls) / n, 2) if n else 0.0,
        "avg_tokens": round(sum(tokens) / n, 1) if n else 0.0,
    }


def markdown_table(title, rows):
    lines = [f"## {title}", "", "| Type | N | Acc | Trunc | Answered | Avg tools | Avg tokens |", "|---|---:|---:|---:|---:|---:|---:|"]
    for key, value in sorted(rows.items(), key=lambda item: (-item[1]["n"], item[0])):
        lines.append(
            f"| {key} | {value['n']} | {value['acc']:.1%} | {value['trunc']:.1%} | "
            f"{value['answered']:.1%} | {value['avg_tool_calls']:.2f} | {value['avg_tokens']:.1f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    records = load_log(args.log)
    grouped = collections.defaultdict(list)
    answer_grouped = collections.defaultdict(list)
    for record in records:
        sample = record.get("input_sample") or {}
        flags = program_flags(sample.get("program") or [])
        grouped[primary_type(flags)].append(record)
        answer_grouped[answer_type(sample.get("answer"))].append(record)

    report = {
        "overall": metrics(records),
        "by_program_type": {key: metrics(value) for key, value in grouped.items()},
        "by_answer_type": {key: metrics(value) for key, value in answer_grouped.items()},
    }
    text = "# KQA Pro question-type report\n\n" + markdown_table(
        "By KoPL program type", report["by_program_type"]) + "\n\n" + markdown_table(
        "By answer type", report["by_answer_type"]) + "\n\n## Overall\n\n" + json.dumps(
        report["overall"], indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        args.output.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
