"""Verify answers over an evaluation log, for any of the four datasets.

This generalises ``utils/verify_mintaka.py`` from ``feat/verification``: it
auto-detects the dataset from the log header and applies the same conservative
semantic comparison to Mintaka, 2Wiki, WebQSP and CWQ. It reports how many
strict-wrong answers the evaluator misevaluated (formatting, dates with
ambiguous two-digit years, quantities, aliases, entity folding), broken down by
the rule that matched and by question type.

Usage::

    python utils/verify_answers.py logs/sft_training/*.out*
    python utils/verify_answers.py <log> --show 15 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from utils.answer_normalization import verify_answer
    from utils.log_analysis import open_log
except ImportError:  # Direct execution: ``python utils/verify_answers.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.answer_normalization import verify_answer
    from utils.log_analysis import open_log


def dataset_name(meta: dict) -> str:
    raw = str(meta.get("dataset", "")).lower()
    for name in ("mintaka", "webqsp", "cwq"):
        if name in raw:
            return name
    if "2wiki" in raw or "twowiki" in raw:
        return "2wiki"
    return raw or "unknown"


def question_type(dataset: str, inp: dict) -> str:
    if dataset == "mintaka":
        return str(inp.get("complexityType", "unknown"))
    if dataset == "2wiki":
        return str(inp.get("type", "unknown"))
    if dataset == "webqsp":
        return "generic"
    q_entities = inp.get("q_entity") or []
    return "multi-entity" if len(q_entities) > 1 else "single-entity"


def _as_list(value):
    if value is None:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def analyze(path: Path, show: int) -> dict:
    meta = {}
    total = answered = strict_correct = normalized_correct = partial = 0
    rules = Counter()
    by_type = defaultdict(lambda: {"n": 0, "strict": 0, "normalized": 0})
    newly_accepted = []
    dataset = ""

    with open_log(path) as fd:
        for lineno, line in enumerate(fd):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if lineno == 0 or (not meta and "evaluation" not in row):
                meta = row
                dataset = dataset_name(meta)
                continue
            if "macro_precision" in row or "evaluation" not in row:
                continue

            total += 1
            prediction = _as_list(row.get("prediction"))
            reference = _as_list(row.get("gt_answer"))
            if not prediction or not reference:
                continue
            answered += 1

            result = verify_answer(
                prediction, reference, dataset=dataset,
                question=row.get("question"), input_sample=row.get("input_sample"),
            )
            evaluator_correct = (row.get("evaluation") or {}).get("correct") == 1
            strict_correct += int(evaluator_correct)
            normalized_correct += int(result.normalized_match)
            partial += int(result.partial_match and not result.normalized_match)

            qtype = question_type(dataset, row.get("input_sample") or {})
            by_type[qtype]["n"] += 1
            by_type[qtype]["strict"] += int(evaluator_correct)
            by_type[qtype]["normalized"] += int(result.normalized_match)

            if result.normalized_match and not evaluator_correct:
                rules[result.rule] += 1
                if len(newly_accepted) < show:
                    newly_accepted.append({
                        "question": row.get("question"),
                        "gt_answer": reference,
                        "prediction": prediction,
                        "rule": result.rule,
                        "reference_normalized": result.reference_normalized,
                        "prediction_normalized": result.prediction_normalized,
                        "type": qtype,
                    })

    return {
        "log": path.name,
        "dataset": dataset,
        "total": total,
        "answered": answered,
        "strict_correct": strict_correct,
        "normalized_correct": normalized_correct,
        "gain": normalized_correct - strict_correct,
        "partial": partial,
        "rules": dict(rules.most_common()),
        "by_type": {k: dict(v) for k, v in sorted(by_type.items())},
        "newly_accepted": newly_accepted,
    }


def print_report(result: dict) -> None:
    print(f"\n### {result['log']}  [{result['dataset']}]")
    print(f"  examples            : {result['total']}")
    print(f"  answered (scored)   : {result['answered']}")
    print(f"  strict correct      : {result['strict_correct']}")
    print(f"  normalized correct  : {result['normalized_correct']}  (+{result['gain']})")
    print(f"  partial only        : {result['partial']}")

    if result["rules"]:
        print("\n  newly accepted by rule:")
        for rule, count in result["rules"].items():
            print(f"    {rule:<30} {count}")

    print("\n  by question type (strict -> normalized):")
    for name, stats in result["by_type"].items():
        strict = stats["strict"] / stats["n"] if stats["n"] else 0
        normalized = stats["normalized"] / stats["n"] if stats["n"] else 0
        print(f"    {name:<16} n={stats['n']:<6} {strict:.3f} -> {normalized:.3f}")

    for item in result["newly_accepted"]:
        print(f"    [{item['rule']}] {item['question']}")
        print(f"        gt={item['gt_answer']!r} pred={item['prediction']!r}")
        print(f"        gt_norm={item['reference_normalized']} pred_norm={item['prediction_normalized']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", help="evaluation JSONL logs (optionally .gz)")
    parser.add_argument("--show", type=int, default=0, help="examples of newly accepted answers to print")
    parser.add_argument("--json", default=None, help="write machine-readable results")
    args = parser.parse_args()

    results = []
    for log in args.logs:
        result = analyze(Path(log), args.show)
        print_report(result)
        results.append(result)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
