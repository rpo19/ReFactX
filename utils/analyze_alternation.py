"""Study how free reasoning relates to constrained generation and complexity.

This builds on :mod:`utils.measure_alternation` and adds three analyses:

1. **Volume balance** - free reasoning words versus constrained-generation
   words (fact and count calls), and the free-to-constrained ratio.
2. **Fact-count buckets** - accuracy as a function of how many fact calls the
   model made, with and without interleaved reasoning after each call.
3. **Complexity** - the same split grouped by question type, with each type
   labelled by its relative accuracy on the large-sample baselines.

Word counts are used as a token proxy because the logs store decoded text.
Each ``<fact>`` block in these runs contains exactly one triple, so
``fact_calls`` also equals the number of retrieved triples.

Usage::

    python utils/analyze_alternation.py logs/22_sept_2026/mintaka_test_eval_qwen35_4b_grpo_checkpoint720.out.0
    python utils/analyze_alternation.py logs/22_sept_2026/<run>.out.0 --min-free 4 --json analysis.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Optional

try:
    from utils.measure_alternation import ExampleMeasurement, measure_log
except ImportError:  # Direct execution: ``python utils/analyze_alternation.py``
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.measure_alternation import ExampleMeasurement, measure_log

FACT_BUCKETS = [(0, 0), (1, 2), (3, 5), (6, 10), (11, 20), (21, 10**9)]
FACT_BUCKET_LABELS = ["none", "1-2", "3-5", "6-10", "11-20", "21+"]

RUN_LENGTH_BUCKETS = [(1, 1), (2, 2), (3, 5), (6, 10), (11, 10**9)]
RUN_LENGTH_LABELS = ["1", "2", "3-5", "6-10", "11+"]

ABSTAIN_PATTERN = re.compile(
    r"i\s+don'?t\s+know|do not know|cannot (?:be )?(?:answer|determine)|insufficient",
    re.IGNORECASE,
)


def run_lengths(free_word_counts: list[int], gap_threshold: int = 1) -> int:
    """Longest run of consecutive fact calls with fewer than ``gap_threshold`` words between them.

    A singleton call (or no between-call span) has run length 1.
    """
    if not free_word_counts:
        return 1
    longest = 1
    current = 1
    for gap in free_word_counts:
        if gap < gap_threshold:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def by_run_length(measurements: list[ExampleMeasurement], gap_threshold: int = 1) -> list[dict]:
    """Accuracy by the longest unbroken sequence of adjacent fact calls (N in "reason every N facts")."""
    rows = []
    for label, (low, high) in zip(RUN_LENGTH_LABELS, RUN_LENGTH_BUCKETS):
        group = [
            m for m in measurements
            if low <= run_lengths(m.free_word_counts, gap_threshold) <= high
        ]
        if not group:
            continue
        rows.append(
            {
                "run": label,
                "n": len(group),
                "accuracy": _accuracy(group),
                "mean_fact_calls": statistics.mean([m.fact_calls for m in group]),
            }
        )
    return rows


def abstention_profile(rows: list[dict]) -> dict:
    """Measure abstention, guessing, and confident-but-wrong behavior.

    ``rows`` must be the raw evaluation records (not the measurement dataclass)
    because abstention needs ``prediction`` and ``triples``.
    """
    total = len(rows)
    if not total:
        return {}

    def is_abstain(row: dict) -> bool:
        prediction = row.get("prediction")
        return bool(prediction) and bool(ABSTAIN_PATTERN.search(str(prediction)))

    abstained = [row for row in rows if is_abstain(row)]
    answered = [row for row in rows if row.get("prediction") is not None]
    empty = [row for row in rows if row.get("prediction") is None]
    answered_not_abstained = [row for row in answered if not is_abstain(row)]
    confident_wrong = [
        row for row in answered_not_abstained
        if row.get("evaluation", {}).get("correct") != 1
    ]
    answered_zero_triples = [row for row in answered if not (row.get("triples") or [])]

    def acc(group: list[dict]) -> Optional[float]:
        if not group:
            return None
        return sum(1 for r in group if r.get("evaluation", {}).get("correct") == 1) / len(group)

    return {
        "examples": total,
        "abstain_rate": len(abstained) / total,
        "abstain_n": len(abstained),
        "answered_rate": len(answered) / total,
        "answered_accuracy": acc(answered),
        "empty_rate": len(empty) / total,
        "confident_wrong_rate": len(confident_wrong) / total,
        "confident_wrong_n": len(confident_wrong),
        "answered_zero_triples_n": len(answered_zero_triples),
        "answered_zero_triples_accuracy": acc(answered_zero_triples),
    }


def _accuracy(group: list[ExampleMeasurement]) -> Optional[float]:
    scored = [m for m in group if m.correct is not None]
    if not scored:
        return None
    return sum(1 for m in scored if m.correct == 1) / len(scored)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def volume_balance(measurements: list[ExampleMeasurement]) -> dict:
    """Compare free reasoning volume with constrained generation volume."""
    with_calls = [m for m in measurements if m.constrained_calls > 0]
    free = [m.free_words for m in with_calls]
    constrained = [m.constrained_words for m in with_calls]
    ratios = [
        m.free_words / m.constrained_words
        for m in with_calls
        if m.constrained_words > 0
    ]
    facts = [m.fact_calls for m in with_calls]
    free_per_fact = [m.free_per_fact for m in with_calls]
    return {
        "examples_with_calls": len(with_calls),
        "mean_free_words": statistics.mean(free) if free else 0.0,
        "median_free_words": statistics.median(free) if free else 0.0,
        "mean_constrained_words": statistics.mean(constrained) if constrained else 0.0,
        "median_constrained_words": statistics.median(constrained) if constrained else 0.0,
        "mean_fact_calls": statistics.mean(facts) if facts else 0.0,
        "median_fact_calls": statistics.median(facts) if facts else 0.0,
        "mean_free_per_fact": statistics.mean(free_per_fact) if free_per_fact else 0.0,
        "median_free_per_fact": statistics.median(free_per_fact) if free_per_fact else 0.0,
        "mean_free_to_constrained_ratio": statistics.mean(ratios) if ratios else 0.0,
        "median_free_to_constrained_ratio": statistics.median(ratios) if ratios else 0.0,
    }


def by_fact_bucket(measurements: list[ExampleMeasurement], min_free: int) -> list[dict]:
    """Accuracy per number-of-fact-calls bucket, split by interleaving."""
    rows = []
    for label, (low, high) in zip(FACT_BUCKET_LABELS, FACT_BUCKETS):
        group = [m for m in measurements if low <= m.fact_calls <= high]
        if not group:
            continue
        alternating = [m for m in group if m.min_free_words >= min_free]
        flat = [m for m in group if m.min_free_words < min_free]
        rows.append(
            {
                "bucket": label,
                "n": len(group),
                "accuracy": _accuracy(group),
                "alternating_n": len(alternating),
                "alternating_accuracy": _accuracy(alternating),
                "flat_n": len(flat),
                "flat_accuracy": _accuracy(flat),
                "mean_free_per_fact": statistics.mean([m.free_per_fact for m in group]),
            }
        )
    return rows


def by_free_range(measurements: list[ExampleMeasurement]) -> list[dict]:
    """Accuracy as free reasoning between calls increases per fact."""
    edges = [(0.0, 0.0), (0.01, 5.0), (5.0, 15.0), (15.0, 40.0), (40.0, 10**9)]
    labels = ["0 (no prose)", "0-5", "5-15", "15-40", "40+"]
    rows = []
    for label, (low, high) in zip(labels, edges):
        group = [m for m in measurements if low <= m.free_per_fact < high]
        if not group:
            continue
        rows.append(
            {
                "free_per_fact": label,
                "n": len(group),
                "accuracy": _accuracy(group),
                "mean_free_words": statistics.mean([m.free_words for m in group]),
                "mean_fact_calls": statistics.mean([m.fact_calls for m in group]),
            }
        )
    return rows


def complexity_by_type(
    measurements: list[ExampleMeasurement], baseline_accuracy: Optional[dict] = None
) -> list[dict]:
    """Per-type accuracy, optionally annotated with baseline relative accuracy."""
    groups: dict[str, list[ExampleMeasurement]] = {}
    for measurement in measurements:
        groups.setdefault(str(measurement.question_type or "unknown"), []).append(measurement)

    rows = []
    for name, group in sorted(groups.items()):
        row = {
            "type": name,
            "n": len(group),
            "accuracy": _accuracy(group),
            "mean_fact_calls": statistics.mean([m.fact_calls for m in group]),
            "mean_free_per_fact": statistics.mean([m.free_per_fact for m in group]),
            "mean_free_words": statistics.mean([m.free_words for m in group]),
        }
        if baseline_accuracy and name in baseline_accuracy:
            row["baseline_accuracy"] = baseline_accuracy[name]
        rows.append(row)
    return rows


def difficulty_tiers(baseline_accuracy: dict) -> dict:
    """Bucket question types into easy/medium/hard by baseline accuracy."""
    ordered = sorted(baseline_accuracy.items(), key=lambda item: item[1])
    if not ordered:
        return {}
    third = max(1, len(ordered) // 3)
    tiers = {
        "hard": [name for name, _ in ordered[:third]],
        "medium": [name for name, _ in ordered[third:-third]] if len(ordered) > 3 else [],
        "easy": [name for name, _ in ordered[-third:]],
    }
    return tiers


def positional_profile(measurements: list[ExampleMeasurement]) -> dict:
    """Where free reasoning happens: before, between, or after calls."""
    return {
        "mean_prefix_words": statistics.mean([m.prefix_words for m in measurements]),
        "mean_between_words": statistics.mean([m.free_words for m in measurements]),
        "mean_suffix_words": statistics.mean([m.suffix_words for m in measurements]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="Evaluation JSONL log file")
    parser.add_argument("--min-free", type=int, default=4)
    parser.add_argument(
        "--run-gap",
        type=int,
        default=1,
        help="Free words below which two consecutive fact calls count as one run",
    )
    parser.add_argument(
        "--type-field",
        default=None,
        help="Per-dataset type field (default: complexityType for Mintaka, type for 2Wiki)",
    )
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    path = Path(args.log)
    type_field = args.type_field
    if type_field is None:
        type_field = "type" if "twowiki" in path.name.lower() else "complexityType"

    measurements = measure_log(path, type_field=type_field)
    with path.open() as handle:
        raw_rows = [json.loads(line) for line in handle][1:-1]

    volume = volume_balance(measurements)
    fact_rows = by_fact_bucket(measurements, args.min_free)
    run_rows = by_run_length(measurements, gap_threshold=args.run_gap)
    free_rows = by_free_range(measurements)
    type_rows = complexity_by_type(measurements)
    position = positional_profile(measurements)
    abstention = abstention_profile(raw_rows)

    print(f"\n### {path.name}  (n={len(measurements)})")

    print("\n## Volume balance (free words vs constrained words)")
    for key, value in volume.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n## Accuracy by number of fact calls")
    print(f"  {'facts':<8}{'n':>5}{'acc':>9}{'alt n':>7}{'alt acc':>10}{'flat n':>8}{'flat acc':>10}{'free/fact':>11}")
    for row in fact_rows:
        print(
            f"  {row['bucket']:<8}{row['n']:>5}{_pct(row['accuracy']):>9}"
            f"{row['alternating_n']:>7}{_pct(row['alternating_accuracy']):>10}"
            f"{row['flat_n']:>8}{_pct(row['flat_accuracy']):>10}"
            f"{row['mean_free_per_fact']:>11.2f}"
        )

    print(f"\n## Accuracy by longest adjacent fact run (gap < {args.run_gap} words)")
    print(f"  {'N facts':<10}{'n':>5}{'acc':>9}{'mean facts':>12}")
    for row in run_rows:
        print(
            f"  {row['run']:<10}{row['n']:>5}{_pct(row['accuracy']):>9}"
            f"{row['mean_fact_calls']:>12.1f}"
        )

    print("\n## Abstention profile")
    print(f"  abstain rate            : {_pct(abstention['abstain_rate'])} (n={abstention['abstain_n']})")
    print(f"  answered rate           : {_pct(abstention['answered_rate'])}")
    print(f"  accuracy when answered  : {_pct(abstention['answered_accuracy'])}")
    print(f"  empty/no-answer rate    : {_pct(abstention['empty_rate'])}")
    print(f"  confident-wrong rate    : {_pct(abstention['confident_wrong_rate'])} (n={abstention['confident_wrong_n']})")
    print(f"  answered w/ 0 triples   : {abstention['answered_zero_triples_n']} acc={_pct(abstention['answered_zero_triples_accuracy'])}")

    print("\n## Accuracy by free words per fact call")
    print(f"  {'free/fact':<14}{'n':>5}{'acc':>9}{'mean free':>11}{'mean facts':>12}")
    for row in free_rows:
        print(
            f"  {row['free_per_fact']:<14}{row['n']:>5}{_pct(row['accuracy']):>9}"
            f"{row['mean_free_words']:>11.1f}{row['mean_fact_calls']:>12.1f}"
        )

    print("\n## Tool-free examples (no fact or count call)")
    no_calls = [m for m in measurements if m.constrained_calls == 0]
    print(f"  n={len(no_calls)} accuracy={_pct(_accuracy(no_calls))}")

    print("\n## By question type")
    print(f"  {'type':<18}{'n':>5}{'acc':>9}{'facts':>8}{'free/fact':>11}{'free words':>12}")
    for row in type_rows:
        print(
            f"  {row['type']:<18}{row['n']:>5}{_pct(row['accuracy']):>9}"
            f"{row['mean_fact_calls']:>8.1f}{row['mean_free_per_fact']:>11.2f}"
            f"{row['mean_free_words']:>12.1f}"
        )

    print("\n## Positional profile (mean words)")
    for key, value in position.items():
        print(f"  {key}: {value:.1f}")

    if args.json:
        payload = {
            "log": path.name,
            "volume_balance": volume,
            "by_fact_bucket": fact_rows,
            "by_run_length": run_rows,
            "by_free_range": free_rows,
            "by_type": type_rows,
            "positional_profile": position,
            "abstention": abstention,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
