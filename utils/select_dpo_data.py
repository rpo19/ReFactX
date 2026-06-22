import json
import random
import argparse
from collections import Counter, defaultdict
from datasets import load_dataset


def load_eval_log(path):
    with open(path) as f:
        lines = f.readlines()
    samples = []
    for l in lines:
        if 'macro_precision' in l:
            continue
        d = json.loads(l)
        if 'input_sample' in d:
            samples.append(d)
    return samples


def get_question_type_mapping(dataset_name="AmazonScience/mintaka", split="train"):
    ds = load_dataset(dataset_name, "default", split=split, revision="refs/convert/parquet")
    mapping = {}
    for s in ds:
        mapping[s["question"]] = s.get("category", "unknown")
    return mapping


def filter_samples(samples, min_f1=0.0, correct_only=False, incorrect_only=False, answered_only=False):
    filtered = []
    for s in samples:
        ev = s["evaluation"]
        if ev["f1"] < min_f1:
            continue
        if correct_only and ev["correct"] != 1:
            continue
        if incorrect_only and ev["correct"] != 0:
            continue
        if answered_only and ev["dont_know"] == 1:
            continue
        filtered.append(s)
    return filtered


def stratify(samples, type_mapping, target_per_type=None, total=None):
    by_type = defaultdict(list)
    for s in samples:
        qtype = type_mapping.get(s["question"], "unknown")
        by_type[qtype].append(s)

    if total is None:
        total = len(samples)
    n_types = len(by_type)
    per_type = target_per_type if target_per_type else total // n_types

    selected = []
    for qtype, group in sorted(by_type.items()):
        random.shuffle(group)
        selected.extend(group[:per_type])
        print(f"  {qtype}: {len(group)} available, selected {min(per_type, len(group))}")

    random.shuffle(selected)
    return selected


def format_dpo(sample):
    return {
        "question": sample["question"],
        "prompt": sample.get("prompt", ""),
        "chosen": sample.get("full_prediction", ""),
        "answer": sample["gt_answer"],
        "prediction": sample["prediction"],
        "evaluation": sample["evaluation"],
    }


def main():
    parser = argparse.ArgumentParser(description="Select DPO data from eval logs")
    parser.add_argument("--eval-log", required=True, help="Path to eval log file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--n", type=int, default=None, help="Total samples to select")
    parser.add_argument("--target-per-type", type=int, default=None, help="Samples per question type")
    parser.add_argument("--correct-only", action="store_true", help="Select only correct samples")
    parser.add_argument("--incorrect-only", action="store_true", help="Select only incorrect samples")
    parser.add_argument("--answered-only", action="store_true", help="Exclude dont_know samples")
    parser.add_argument("--min-f1", type=float, default=0.0, help="Minimum F1 threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset", default="AmazonScience/mintaka", help="Dataset for question types")
    parser.add_argument("--dataset-split", default="train", help="Dataset split")
    args = parser.parse_args()

    random.seed(args.seed)

    print("Loading eval log...")
    samples = load_eval_log(args.eval_log)
    print(f"  {len(samples)} samples loaded")

    print("Loading question type mapping...")
    type_mapping = get_question_type_mapping(args.dataset, args.dataset_split)
    print(f"  {len(type_mapping)} questions mapped")
    matched = sum(1 for s in samples if s["question"] in type_mapping)
    print(f"  {matched}/{len(samples)} eval samples matched to question types")

    print("Filtering samples...")
    samples = filter_samples(samples, min_f1=args.min_f1, correct_only=args.correct_only,
                             incorrect_only=args.incorrect_only, answered_only=args.answered_only)
    print(f"  {len(samples)} after filtering")

    print("Stratifying by question type...")
    selected = stratify(samples, type_mapping, target_per_type=args.target_per_type, total=args.n)
    if args.n and len(selected) > args.n:
        selected = selected[:args.n]
    print(f"  {len(selected)} samples selected")

    with open(args.output, "w") as f:
        for s in selected:
            f.write(json.dumps(format_dpo(s)) + "\n")

    print(f"Output written to {args.output}")
    type_counts = Counter(type_mapping.get(s["question"], "unknown") for s in selected)
    print("Distribution by type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t}: {c}")


if __name__ == "__main__":
    main()
