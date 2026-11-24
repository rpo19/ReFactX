"""Create a balanced KQA Pro train sample for 27B teacher generation.

KQA Pro has 94k train questions. Running a 27B teacher over all of them is
expensive, so this script selects a balanced subset by program type:

* ``qualifier``: any ``*Qualifier`` KoPL function;
* ``count``: a ``Count`` function (without qualifier);
* ``verify``: a ``Verify*`` function (yes/no);
* ``multihop``: at least two ``Relate`` steps;
* ``base``: everything else.

The output is a JSON list of question ids, consumable by ``utils/eval.py`` via
``sample_index_file`` with ``id_key: "id"``.  A manifest with the selected and
full-corpus type distribution is written next to it.

Example::

    python -m utils.sample_kqa_pro \
        --data-dir /workspace/data/kqa_pro \
        --output /workspace/data/kqa_pro/train_sample_ids.json \
        --total 12000
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def program_flags(program: list[dict]) -> dict:
    functions = [str(step.get("function", "")) for step in program or []]
    return {
        "has_count": any(fn == "Count" for fn in functions),
        "has_verify": any(fn.startswith("Verify") for fn in functions),
        "has_qualifier": any("Qualifier" in fn for fn in functions),
        "num_relate": sum(fn == "Relate" for fn in functions),
        "num_steps": len(functions),
    }


def answer_type(answer) -> str:
    if isinstance(answer, bool):
        return "yesno"
    if isinstance(answer, int):
        return "integer"
    if isinstance(answer, float):
        return "number"
    text = str(answer).strip().lower()
    if text in ("yes", "no", "unknown"):
        return "yesno"
    if re.fullmatch(r"-?\d+", text):
        return "integer"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return "date"
    return "string"


def primary_type(flags: dict) -> str:
    # Overlaps are resolved toward the more specific curriculum bucket.
    if flags["has_qualifier"]:
        return "qualifier"
    if flags["has_count"]:
        return "count"
    if flags["has_verify"]:
        return "verify"
    if flags["num_relate"] >= 2:
        return "multihop"
    return "base"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def allocate(total: int, sizes: dict[str, int]) -> dict[str, int]:
    available = sum(sizes.values())
    if total >= available:
        return dict(sizes)
    quotas = {}
    remaining = total
    keys = [k for k, v in sizes.items() if v > 0]
    for key in keys:
        quota = int(round(total * sizes[key] / available))
        quotas[key] = min(quota, sizes[key])
        remaining -= quotas[key]
    index = 0
    while remaining > 0 and keys:
        key = keys[index % len(keys)]
        if quotas[key] < sizes[key]:
            quotas[key] += 1
            remaining -= 1
        index += 1
        if all(quotas[k] >= sizes[k] for k in keys):
            break
    return quotas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/data/kqa_pro"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: <data-dir>/<split>_sample_ids.json")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--total", type=int, default=12000,
                        help="Total questions to select (default: 12000)")
    parser.add_argument("--count", type=int, default=None,
                        help="Override quota for count questions")
    parser.add_argument("--verify", type=int, default=None,
                        help="Override quota for verify questions")
    parser.add_argument("--qualifier", type=int, default=None,
                        help="Override quota for qualifier questions")
    parser.add_argument("--multihop", type=int, default=None,
                        help="Override quota for multi-hop questions")
    parser.add_argument("--base", type=int, default=None,
                        help="Override quota for base questions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = args.data_dir / f"{args.split}.jsonl"
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; run utils/prep_kqa_pro.py first")
    rows = load_jsonl(data_path)
    rng = random.Random(args.seed)

    buckets: dict[str, list[str]] = defaultdict(list)
    corpus_types = Counter()
    corpus_answers = Counter()
    for row in rows:
        flags = program_flags(row.get("program") or [])
        kind = primary_type(flags)
        buckets[kind].append(str(row["id"]))
        corpus_types[kind] += 1
        corpus_answers[answer_type(row.get("answer"))] += 1

    overrides = {
        "count": args.count,
        "verify": args.verify,
        "qualifier": args.qualifier,
        "multihop": args.multihop,
        "base": args.base,
    }
    sizes = {key: len(values) for key, values in buckets.items()}
    if any(value is not None for value in overrides.values()):
        quotas = {
            key: min(overrides[key], sizes.get(key, 0))
            if overrides[key] is not None else 0
            for key in sizes
        }
    else:
        quotas = allocate(args.total, sizes)

    selected = []
    selected_by_type = {}
    for key, ids in buckets.items():
        quota = quotas.get(key, 0)
        chosen = rng.sample(ids, min(quota, len(ids)))
        selected.extend(chosen)
        selected_by_type[key] = len(chosen)
    rng.shuffle(selected)

    output = args.output or args.data_dir / f"{args.split}_sample_ids.json"
    manifest = args.manifest or output.with_suffix(".manifest.json")
    output.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps({
        "split": args.split,
        "total": len(selected),
        "requested_total": args.total,
        "quotas": quotas,
        "selected_by_type": selected_by_type,
        "corpus_by_type": dict(corpus_types),
        "corpus_answer_type": dict(corpus_answers),
        "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(selected)} ids to {output}")
    print(json.dumps({
        "selected_by_type": selected_by_type,
        "corpus_by_type": dict(corpus_types),
    }, indent=2))


if __name__ == "__main__":
    main()
