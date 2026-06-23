import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict


def load_eval_log(path):
    samples = []
    meta = {}
    with open(path) as f:
        header = json.loads(f.readline())
        meta["dataset"] = header.get("dataset", "unknown")
        meta["dataset_split"] = header.get("dataset_split", "unknown")
        for line in f:
            line = line.strip()
            if not line or 'macro_precision' in line:
                continue
            s = json.loads(line)
            s['_dataset'] = meta["dataset"]
            s['_dataset_split'] = meta["dataset_split"]
            samples.append(s)
    return samples


def filter_positive_with_triples(samples):
    result = []
    for s in samples:
        if s['evaluation']['correct'] == 1 and len(s.get('triples', [])) > 0:
            result.append(s)
    return result


def type_distribution(samples, type_field='complexityType'):
    types = Counter()
    for s in samples:
        key = (s['input_sample'].get(type_field, 'unknown'), s['_dataset'], s['_dataset_split'])
        types[key] += 1
    return types


def filter_by_types(samples, types, type_field='complexityType'):
    if isinstance(types, str):
        types = [types]
    return [s for s in samples if s['input_sample'].get(type_field, 'unknown') in types]


def stratify_by_type(samples, target_per_type=None, total=None, seed=42, type_field='complexityType'):
    by_type = defaultdict(list)
    for s in samples:
        by_type[s['input_sample'].get(type_field, 'unknown')].append(s)

    if total is None:
        total = len(samples)
    n_types = len(by_type)
    per_type = target_per_type if target_per_type else total // n_types

    rng = random.Random(seed)
    selected = []
    for t, group in sorted(by_type.items()):
        rng.shuffle(group)
        n = min(per_type, len(group))
        selected.extend(group[:n])

    rng.shuffle(selected)
    return selected


def type_distribution_report(samples, type_field='complexityType'):
    dist = type_distribution(samples, type_field)
    grouped = defaultdict(Counter)
    for (t, ds, split), c in dist.items():
        grouped[t][f"{ds} ({split})"] = c
    print(f"\nQuestion type distribution ({type_field}):")
    for t in sorted(grouped.keys()):
        sources = grouped[t]
        total = sum(sources.values())
        source_details = ", ".join(f"{s}: {c}" for s, c in sorted(sources.items()))
        print(f"  {t}: {total} ({source_details})")
    return dist


def to_output_format(samples):
    result = []
    for s in samples:
        result.append({
            "question": s["question"],
            "gt_answer": s["gt_answer"],
            "prediction": s["prediction"],
            "full_prediction": s["full_prediction"],
            "prompt": s["prompt"],
            "triples": s["triples"],
            "dataset": s["_dataset"],
            "dataset_split": s["_dataset_split"],
            "complexityType": s['input_sample'].get('complexityType'),
            "category": s['input_sample'].get('category'),
            "evaluation": s["evaluation"],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description="Filter positive training samples from eval logs")
    parser.add_argument("--eval-log", required=True, help="Path to enriched eval log file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--type-field", default="complexityType", help="Field name for question type in input_sample")
    parser.add_argument("--type-distribution", action="store_true", help="Show question type distribution and exit")
    parser.add_argument("--filter-types", nargs="+", default=None, help="Only keep these question types")
    parser.add_argument("--stratify", action="store_true", help="Stratify by question type")
    parser.add_argument("--target-per-type", type=int, default=None, help="Samples per type (for stratification)")
    parser.add_argument("--n", type=int, default=None, help="Total samples to select (for stratification)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratification")

    args = parser.parse_args()

    print(f"Loading eval log: {args.eval_log}")
    samples = load_eval_log(args.eval_log)
    print(f"  Total samples: {len(samples)}")

    samples = filter_positive_with_triples(samples)
    print(f"  Positive with triples: {len(samples)}")

    if args.type_distribution:
        type_distribution_report(samples, args.type_field)
        return

    if args.filter_types:
        samples = filter_by_types(samples, args.filter_types, args.type_field)
        print(f"  After type filter: {len(samples)}")

    if args.stratify:
        samples = stratify_by_type(samples, target_per_type=args.target_per_type, total=args.n, seed=args.seed, type_field=args.type_field)
        print(f"  After stratification: {len(samples)}")

    output = to_output_format(samples)

    raw_dist = type_distribution(samples, args.type_field)
    grouped = defaultdict(dict)
    for (t, ds, split), c in raw_dist.items():
        grouped[t][f"{ds} ({split})"] = c
    type_distribution_report(samples, args.type_field)

    report = {
        "_report": True,
        "total_input": len(samples),
        "total_output": len(output),
        "type_field": args.type_field,
        "type_distribution": {t: dict(sources) for t, sources in sorted(grouped.items())},
    }

    with open(args.output, "w") as f:
        f.write(json.dumps(report) + "\n")
        for s in output:
            f.write(json.dumps(s) + "\n")

    print(f"\nSaved {len(output)} samples to {args.output}")


if __name__ == "__main__":
    main()
