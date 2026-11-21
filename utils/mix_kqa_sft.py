"""Mix KQA Pro SFT traces into the clean teacher mixture.

By default this replaces Mintaka count training examples with KQA Pro traces
while keeping the rest of the clean teacher mixture.  KQA Pro can also be added
in full (all program types) for the question-type study.

Example::

    python -m utils.mix_kqa_sft \
        --base logs/sft_training_mix_v3.jsonl \
        --kqa logs/kqa_pro_sft.jsonl \
        --output logs/sft_training_mix_kqa.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path


def norm_question(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def is_mintaka_count(record: dict) -> bool:
    if record.get("dataset") != "mintaka":
        return False
    return (
        str(record.get("complexityType", "")).lower() == "count"
        or str(record.get("question_type", "")).lower() == "count"
        or bool(record.get("has_count"))
    )


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("_report"):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True,
                        help="Clean teacher mixture (e.g. logs/sft_training_mix_v3.jsonl)")
    parser.add_argument("--kqa", nargs="+", type=Path, required=True,
                        help="KQA Pro SFT files from utils/build_kqa_sft.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--keep-mintaka-count", action="store_true",
                        help="Do not drop Mintaka count traces (default: drop)")
    parser.add_argument("--kqa-count-only", action="store_true",
                        help="Only add KQA Pro count traces")
    parser.add_argument("--kqa-repeat", type=int, default=1,
                        help="Repeat KQA Pro traces N times (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = load(args.base)
    kqa = []
    for path in args.kqa:
        kqa.extend(load(path))
    if args.kqa_count_only:
        kqa = [row for row in kqa if row.get("has_count")]

    dropped_base = []
    kept_base = []
    for row in base:
        if not args.keep_mintaka_count and is_mintaka_count(row):
            dropped_base.append(row)
        else:
            kept_base.append(row)

    combined = kept_base + kqa * max(1, args.kqa_repeat)
    seen = set()
    deduped = []
    duplicates = collections.Counter()
    for row in combined:
        key = norm_question(row.get("question"))
        if not key:
            continue
        if key in seen:
            duplicates[row.get("dataset", "?")] += 1
            continue
        seen.add(key)
        deduped.append(row)
    random.Random(args.seed).shuffle(deduped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        for row in deduped:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "written": len(deduped),
        "base_total": len(base),
        "base_kept": len(kept_base),
        "mintaka_count_dropped": len(dropped_base),
        "kqa_total": len(kqa),
        "kqa_repeat": args.kqa_repeat,
        "duplicates": dict(duplicates),
        "by_dataset": dict(collections.Counter(
            row.get("dataset", "?") for row in deduped)),
        "by_type": dict(collections.Counter(
            row.get("question_type", "?") for row in deduped)),
    }
    manifest = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
