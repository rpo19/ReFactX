"""Build a stratified evaluation set for the mixed RL run.

The training mixture's validation split has only 200 rows (50 per dataset) and
just a handful of count questions, which makes the eval too noisy to tell
whether the policy is learning. This builder samples a larger evaluation set
from the held-out HuggingFace validation splits, balanced by dataset and
question type, with a dedicated quota for count questions.

Questions already present in the RL mixture (train or validation) are excluded
(normalized match) so the eval stays held out. 2Wiki is sampled from its *train*
split because the project reports 2Wiki results on the canonical dev split;
`--split-overrides 2wiki=train` is the default.

Example::

    python -m utils.build_eval_set \
        --output "${WS_PATH}/datasets/rl_eval_stratified.jsonl" \
        --exclude-from "${WS_PATH}/datasets/rl_train_mix.jsonl" \
        --per-dataset 50 --count-per-dataset 40
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from utils.build_rl_dataset import (
    DATASETS,
    _load_split,
    infer_question_type,
    parse_answer_values,
)
from utils.build_sft_dataset import normalize_question


def _load_excluded(paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                nested = record.get("input_sample")
                question = record.get("question")
                if not question and isinstance(nested, dict):
                    question = nested.get("question")
                if question:
                    excluded.add(normalize_question(str(question)))
    return excluded


def _collect_rows(dataset: str, split: str, *, excluded: set[str],
                  wikidata_table: str, freebase_table: str,
                  enumeration_datasets: set[str]) -> list[dict]:
    config = DATASETS[dataset]
    try:
        data = _load_split(dataset, split)
    except Exception as exc:  # noqa: BLE001 - report and skip missing splits
        print(f"warning: cannot load {dataset}/{split}: {exc}")
        return []
    table = wikidata_table if config["index_table"] == "wikidata" else freebase_table
    rows: list[dict] = []
    for index in range(len(data)):
        example = data[index]
        question = str(example.get(config["question"]) or "").strip()
        answer = parse_answer_values(example.get(config["answer"]))
        if not question or not answer:
            continue
        if normalize_question(question) in excluded:
            continue
        question_type = infer_question_type(dataset, example)
        requires_enumeration = (
            dataset in enumeration_datasets and len(answer) >= 2
        )
        rows.append({
            "id": f"{dataset}:{example.get(config['id'], index)}",
            "dataset": dataset,
            "index_table": table,
            "split": "validation",
            "source_split": split,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "requires_full_enumeration": bool(requires_enumeration),
            "source": config["hf"],
        })
    return rows


def _quota_for_types(rows: list[dict], per_dataset: int, count_per_dataset: int,
                     count_type: str) -> dict[str, int]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["question_type"])].append(index)
    count_label = count_type.lower()
    quotas: dict[str, int] = {}
    non_count = [k for k in grouped if k.lower() != count_label]
    non_count_total = sum(len(grouped[k]) for k in non_count) or 1
    for key, indices in grouped.items():
        if key.lower() == count_label:
            quotas[key] = min(len(indices), count_per_dataset)
        else:
            share = per_dataset * len(indices) / non_count_total
            quotas[key] = min(len(indices), max(1, round(share)))
    return quotas


def build(output: str, datasets: Iterable[str], splits: Iterable[str], *,
          per_dataset: int, count_per_dataset: int, exclude_from: Iterable[str],
          wikidata_table: str, freebase_table: str,
          enumeration_datasets: Iterable[str], count_type: str,
          split_overrides: dict[str, str], seed: int) -> dict:
    excluded = _load_excluded(exclude_from)
    enumeration = {name.lower() for name in enumeration_datasets}
    rng = random.Random(seed)
    selected: list[dict] = []
    for dataset in datasets:
        dataset_splits = (
            [split_overrides[dataset]] if dataset in split_overrides else list(splits)
        )
        for split in dataset_splits:
            rows = _collect_rows(
                dataset, split, excluded=excluded,
                wikidata_table=wikidata_table,
                freebase_table=freebase_table,
                enumeration_datasets=enumeration,
            )
            if not rows:
                continue
            quotas = _quota_for_types(
                rows, per_dataset, count_per_dataset, count_type
            )
            grouped: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(rows):
                grouped[str(row["question_type"])].append(index)
            for qtype, indices in grouped.items():
                take = min(quotas.get(qtype, 0), len(indices))
                if take <= 0:
                    continue
                chosen = indices if take >= len(indices) else rng.sample(sorted(indices), take)
                selected.extend(rows[i] for i in chosen)
            print(f"{dataset}/{split}: {len(rows)} candidates -> "
                  f"{sum(min(quotas.get(k, 0), len(v)) for k, v in grouped.items())} selected")

    # Deduplicate identical questions across datasets/splits.
    seen: set[str] = set()
    unique: list[dict] = []
    for row in selected:
        key = normalize_question(str(row["question"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in unique:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(out_path),
        "records": len(unique),
        "datasets": sorted(set(row["dataset"] for row in unique)),
        "splits": sorted(set(splits)),
        "split_overrides": dict(split_overrides),
        "source_splits": dict(Counter(
            f"{row['dataset']}:{row.get('source_split', '')}" for row in unique
        )),
        "seeded_from": seed,
        "excluded_questions": len(excluded),
        "per_dataset": per_dataset,
        "count_per_dataset": count_per_dataset,
        "counts_by_dataset": dict(Counter(row["dataset"] for row in unique)),
        "counts_by_type": dict(Counter(str(row["question_type"]) for row in unique)),
        "count_questions": sum(
            1 for row in unique
            if str(row["question_type"]).lower() == count_type.lower()
        ),
        "index_counts": dict(Counter(row["index_table"] for row in unique)),
    }
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=sorted(DATASETS))
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument(
        "--split-overrides", nargs="+", default=["2wiki=train"],
        help="Per-dataset split override, e.g. 2wiki=train (the project evaluates "
             "2Wiki on the canonical dev split, so it must not be used here)",
    )
    parser.add_argument("--per-dataset", type=int, default=50,
                        help="Budget per dataset for non-count question types")
    parser.add_argument("--count-per-dataset", type=int, default=40,
                        help="Dedicated budget for count questions per dataset")
    parser.add_argument("--exclude-from", nargs="+", default=[],
                        help="RL training JSONL(s) whose train questions are excluded")
    parser.add_argument("--wikidata-table", default="qwen36")
    parser.add_argument("--freebase-table", default="qwen36fbwikiprops")
    parser.add_argument("--enumeration-datasets", nargs="+", default=["webqsp"])
    parser.add_argument("--count-type", default="count")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    split_overrides: dict[str, str] = {}
    for item in args.split_overrides or []:
        if "=" not in item:
            parser.error(f"--split-overrides expects dataset=split, got {item!r}")
        dataset, split = item.split("=", 1)
        split_overrides[dataset] = split

    manifest = build(
        args.output,
        args.datasets,
        args.splits,
        per_dataset=args.per_dataset,
        count_per_dataset=args.count_per_dataset,
        exclude_from=args.exclude_from,
        wikidata_table=args.wikidata_table,
        freebase_table=args.freebase_table,
        enumeration_datasets=args.enumeration_datasets,
        count_type=args.count_type,
        split_overrides=split_overrides,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
