"""Run Mintaka answer verification over an evaluation log.

Reports how many examples the strict evaluator marked wrong are actually
semantically equivalent, broken down by the rule that matched, so the size of
the "formatting vs genuine error" gap is measurable.

Usage::

    python utils/verify_mintaka.py logs/22_sept_2026/mintaka_test_eval_*.out.0
    python utils/verify_mintaka.py <log> --show 15 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from utils.answer_normalization import verify_mintaka
except ImportError:  # Direct execution: ``python utils/verify_mintaka.py``
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.answer_normalization import verify_mintaka


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle]
    return rows[1:-1]


def analyze(path: Path) -> dict:
    rows = load_rows(path)
    total = len(rows)

    answered = 0
    strict_correct = 0
    normalized_correct = 0
    partial = 0
    newly_accepted = []
    rules = Counter()
    by_type = defaultdict(lambda: {"n": 0, "strict": 0, "normalized": 0})

    for row in rows:
        prediction = row.get("prediction")
        reference = row.get("gt_answer")
        if prediction is None or reference is None:
            continue
        answered += 1

        result = verify_mintaka(
            prediction,
            reference,
            question=row.get("question"),
            input_sample=row.get("input_sample"),
        )

        evaluator_correct = (row.get("evaluation") or {}).get("correct") == 1
        strict_correct += int(evaluator_correct)
        normalized_correct += int(result.normalized_match)
        partial += int(result.partial_match and not result.normalized_match)

        qtype = (row.get("input_sample") or {}).get("complexityType", "unknown")
        by_type[qtype]["n"] += 1
        by_type[qtype]["strict"] += int(evaluator_correct)
        by_type[qtype]["normalized"] += int(result.normalized_match)

        if result.normalized_match and not evaluator_correct:
            rules[result.rule] += 1
            newly_accepted.append({
                "question": row.get("question"),
                "gt_answer": reference,
                "prediction": prediction,
                "rule": result.rule,
                "reference_normalized": result.reference_normalized,
                "prediction_normalized": result.prediction_normalized,
                "complexityType": qtype,
            })

    return {
        "log": path.name,
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


def print_report(result: dict, show: int) -> None:
    print(f"\n### {result['log']}")
    print(f"  examples            : {result['total']}")
    print(f"  answered (scored)   : {result['answered']}")
    print(f"  strict correct      : {result['strict_correct']}")
    print(f"  normalized correct  : {result['normalized_correct']}  (+{result['gain']})")
    print(f"  partial only        : {result['partial']}")

    if result["rules"]:
        print("\n  newly accepted by rule:")
        for rule, count in result["rules"].items():
            print(f"    {rule:<30} {count}")

    print("\n  by complexityType (strict -> normalized):")
    for name, stats in result["by_type"].items():
        strict = stats["strict"] / stats["n"] if stats["n"] else 0
        normalized = stats["normalized"] / stats["n"] if stats["n"] else 0
        print(f"    {name:<14} n={stats['n']:<5} {strict:.3f} -> {normalized:.3f}")

    if show and result["newly_accepted"]:
        print(f"\n  first {min(show, len(result['newly_accepted']))} newly accepted:")
        for item in result["newly_accepted"][:show]:
            print(f"    [{item['rule']}] {item['question']}")
            print(f"        gt={item['gt_answer']!r} pred={item['prediction']!r}")
            print(f"        gt_norm={item['reference_normalized']} "
                  f"pred_norm={item['prediction_normalized']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", help="Mintaka evaluation JSONL logs")
    parser.add_argument("--show", type=int, default=10,
                        help="How many newly accepted examples to print (0 to disable)")
    parser.add_argument("--json", default=None, help="Write machine-readable results")
    args = parser.parse_args()

    results = []
    for log in args.logs:
        result = analyze(Path(log))
        print_report(result, args.show)
        results.append(result)

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
