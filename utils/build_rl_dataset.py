"""Build a mixed RL training set across Mintaka, 2Wiki, WebQSP and CWQ.

Each output record carries the KB table it must be decoded against:

* Mintaka and 2Wiki -> ``qwen36`` (Wikidata)
* WebQSP and CWQ    -> ``qwen36fbwikiprops`` (Freebase; the table name is
  cluster-specific, so it stays configurable via ``--freebase-table``)

The builder only uses training/validation splits, deduplicates questions
globally, and can subtract the questions already used by SFT so RL sees a
different (or at least non-identical) sample.

Example::

    python -m utils.build_rl_dataset \
        --output logs/rl_train_mix.jsonl \
        --max-per-dataset 2000 --val-per-dataset 50 \
        --exclude-from logs/sft_training_mix_supported.jsonl
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from utils.build_sft_dataset import normalize_question

DATASETS = {
    "mintaka": {
        "hf": "rpozzi/mintaka",
        "question": "question",
        "answer": "answerText",
        "id": "id",
        "type_key": "complexityType",
        "index_table": "wikidata",
        "columns": ["id", "question", "answerText", "complexityType"],
        "splits": ("train", "validation"),
    },
    "2wiki": {
        "hf": "xanhho/2WikiMultihopQA",
        "question": "question",
        "answer": "answer",
        "id": "_id",
        "type_key": "type",
        "index_table": "wikidata",
        "columns": ["_id", "question", "answer", "type"],
        "splits": ("train", "validation"),
    },
    "webqsp": {
        "hf": "rmanluo/RoG-webqsp",
        "question": "question",
        "answer": "answer",
        "id": "id",
        "type_key": None,
        "index_table": "freebase",
        "columns": ["id", "question", "answer", "q_entity"],
        "splits": ("train", "validation"),
    },
    "cwq": {
        "hf": "rmanluo/RoG-cwq",
        "question": "question",
        "answer": "answer",
        "id": "id",
        "type_key": None,
        "index_table": "freebase",
        "columns": ["id", "question", "answer", "q_entity"],
        "splits": ("train", "validation"),
    },
}

DEFAULT_SFT_FILES = (
    "sft_training_mix_supported.jsonl.gz",
    "logs/sft_training_mix_supported.jsonl.gz",
    "logs/sft_training_mix_supported.jsonl",
    "logs/sft_training_mix.jsonl",
)


def parse_answer_values(value: Any) -> list[str]:
    """Return an answer as a list of strings.

    Mintaka/2Wiki answers are plain strings; RoG answers are usually a string
    representation of a Python list.  Parse the latter without changing plain
    answers such as ``"yes"`` or ``"1965"``.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if item is not None and str(item).strip()]
    return [str(parsed).strip()]


def infer_question_type(dataset: str, example: dict) -> str:
    if dataset == "mintaka":
        return str(example.get("complexityType") or "unknown")
    if dataset == "2wiki":
        return str(example.get("type") or "unknown")
    if dataset == "webqsp":
        return "generic"
    entities = parse_answer_values(example.get("q_entity"))
    return "multi-entity" if len(entities) > 1 else "single-entity"


def _question_from_record(record: dict) -> str:
    if isinstance(record, dict):
        for key in ("question", "source_question"):
            value = record.get(key)
            if value:
                return str(value)
        nested = record.get("input_sample")
        if isinstance(nested, dict) and nested.get("question"):
            return str(nested["question"])
    return ""


def load_excluded_questions(paths: Iterable[str]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path or not os.path.isfile(path):
            print(f"warning: exclusion file not found: {path}")
            continue
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = _question_from_record(record)
                if question:
                    excluded.add(normalize_question(question))
        print(f"loaded {len(excluded)} excluded questions so far from {path}")
    return excluded


def _load_split(dataset: str, split: str):
    from datasets import load_dataset

    config = DATASETS[dataset]
    ds = load_dataset(config["hf"], split=split)
    keep = [column for column in config["columns"] if column in ds.column_names]
    if keep:
        if hasattr(ds, "select_columns"):
            ds = ds.select_columns(keep)
        else:
            ds = ds.remove_columns(
                [column for column in ds.column_names if column not in keep]
            )
    return ds


def _rows_from_split(
    dataset: str,
    split: str,
    *,
    limit: int,
    rng: random.Random,
    excluded: set[str],
    seen: set[str],
    wikidata_table: str,
    freebase_table: str,
    enumeration_datasets: set[str],
    count_target: int = 0,
    allow_overlap_types: set[str] | None = None,
) -> list[dict]:
    if limit <= 0:
        return []
    config = DATASETS[dataset]
    ds = _load_split(dataset, split)
    n = len(ds)
    if n == 0:
        return []
    table = wikidata_table if config["index_table"] == "wikidata" else freebase_table
    allow_overlap_types = {str(name).lower() for name in (allow_overlap_types or [])}

    def make_row(index: int) -> dict | None:
        example = ds[index]
        question = str(example.get(config["question"]) or "").strip()
        answer = parse_answer_values(example.get(config["answer"]))
        if not question or not answer:
            return None
        key = normalize_question(question)
        if key in seen:
            return None
        question_type = infer_question_type(dataset, example)
        if key in excluded and question_type.lower() not in allow_overlap_types:
            return None
        if dataset in enumeration_datasets:
            requires_enumeration = len(answer) >= 2
        else:
            requires_enumeration = False
        row = {
            "id": f"{dataset}:{example.get(config['id'], index)}",
            "dataset": dataset,
            "index_table": table,
            "split": "validation" if split == "validation" else "train",
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "requires_full_enumeration": bool(requires_enumeration),
            "source": config["hf"],
        }
        seen.add(key)
        return row

    # A count quota keeps the RL mixture useful for the count tool: strict SFT
    # exclusion otherwise removes almost all Mintaka count questions.  Select
    # the count pool first, then fill the rest of the per-dataset budget with
    # other types.
    if split == "train" and count_target > 0 and dataset == "mintaka":
        target = min(limit, count_target)
        count_indices = []
        other_indices = []
        for index in range(n):
            if infer_question_type(dataset, ds[index]).lower() == "count":
                count_indices.append(index)
            else:
                other_indices.append(index)
        rng.shuffle(count_indices)
        rng.shuffle(other_indices)
        rows = []
        for index in count_indices:
            if len(rows) >= target:
                break
            row = make_row(index)
            if row is not None:
                rows.append(row)
        for index in other_indices:
            if len(rows) >= limit:
                break
            row = make_row(index)
            if row is not None:
                rows.append(row)
        return rows

    # Over-select so SFT-overlap filtering does not starve the sample.  If no
    # exclusions are configured, sampling the exact limit is enough.
    pool_size = min(n, limit if not excluded else max(limit * 4, limit + 500))
    indices = rng.sample(range(n), pool_size) if pool_size < n else list(range(n))
    rows = []
    for index in indices:
        if len(rows) >= limit:
            break
        row = make_row(index)
        if row is not None:
            rows.append(row)
    return rows


def build(
    output: str,
    *,
    datasets: Iterable[str],
    max_per_dataset: int,
    val_per_dataset: int,
    seed: int,
    exclude_from: Iterable[str],
    wikidata_table: str,
    freebase_table: str,
    enumeration_datasets: Iterable[str],
    count_target: int = 0,
    allow_overlap_types: Iterable[str] = (),
) -> dict:
    rng = random.Random(seed)
    excluded = load_excluded_questions(exclude_from)
    seen: set[str] = set()
    enumeration = {str(name).lower() for name in enumeration_datasets}
    overlap_types = (
        {str(name).lower() for name in allow_overlap_types}
        if count_target > 0 else set()
    )
    validation_rows: list[dict] = []
    train_rows: list[dict] = []

    # Validation first so an eval question is not also used for training.
    for dataset in datasets:
        validation_rows.extend(
            _rows_from_split(
                dataset, "validation",
                limit=val_per_dataset, rng=rng, excluded=excluded, seen=seen,
                wikidata_table=wikidata_table, freebase_table=freebase_table,
                enumeration_datasets=enumeration,
            )
        )
    for dataset in datasets:
        train_rows.extend(
            _rows_from_split(
                dataset, "train",
                limit=max_per_dataset, rng=rng, excluded=excluded, seen=seen,
                wikidata_table=wikidata_table, freebase_table=freebase_table,
                enumeration_datasets=enumeration,
                count_target=count_target,
                allow_overlap_types=overlap_types,
            )
        )

    # Interleave datasets so a small max_eval_samples still covers all four and
    # so mixed-index training batches are the norm rather than the exception.
    rng.shuffle(validation_rows)
    rng.shuffle(train_rows)
    rows = validation_rows + train_rows

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fd:
        for row in rows:
            fd.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = Counter((row["split"], row["dataset"]) for row in rows)
    manifest = {
        "output": str(output_path),
        "records": len(rows),
        "seed": seed,
        "max_per_dataset": max_per_dataset,
        "val_per_dataset": val_per_dataset,
        "count_target": count_target,
        "allow_overlap_types": sorted(overlap_types),
        "wikidata_table": wikidata_table,
        "freebase_table": freebase_table,
        "enumeration_datasets": sorted(enumeration),
        "excluded_questions": len(excluded),
        "counts": {f"{split}/{dataset}": count for (split, dataset), count in sorted(counts.items())},
        "by_type": dict(Counter(row["question_type"] for row in rows)),
        "index_counts": dict(Counter(row["index_table"] for row in rows)),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="logs/rl_train_mix.jsonl")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--max-per-dataset", type=int, default=2000)
    parser.add_argument("--val-per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-from", nargs="*", default=None,
                        help="SFT JSONL/JSONL.GZ files whose questions must be skipped")
    parser.add_argument("--no-default-exclude", action="store_true",
                        help="Do not auto-use logs/sft_training_mix*.jsonl")
    parser.add_argument("--wikidata-table", default="qwen36")
    parser.add_argument("--freebase-table", default="qwen36fbwikiprops",
                        help="Freebase table; use qwen36fbwikipops on databases that name it without the r")
    parser.add_argument("--count-target", type=int, default=0,
                        help="Number of count-type training questions to keep (Mintaka)")
    parser.add_argument("--allow-overlap-types", nargs="*", default=[],
                        help="Question types allowed to overlap the SFT mixture (e.g. count)")
    parser.add_argument("--enumeration-datasets", nargs="+", default=["webqsp"])
    args = parser.parse_args(argv)

    exclude_from = list(args.exclude_from or [])
    if not args.no_default_exclude and not exclude_from:
        exclude_from = [path for path in DEFAULT_SFT_FILES if os.path.isfile(path)]
        if not exclude_from:
            print(
                "warning: no default SFT mixture found; RL may reuse SFT questions. "
                "Pass --exclude-from <file> to subtract them."
            )
    manifest = build(
        args.output,
        datasets=args.datasets,
        max_per_dataset=args.max_per_dataset,
        val_per_dataset=args.val_per_dataset,
        seed=args.seed,
        exclude_from=exclude_from,
        wikidata_table=args.wikidata_table,
        freebase_table=args.freebase_table,
        enumeration_datasets=args.enumeration_datasets,
        count_target=args.count_target,
        allow_overlap_types=args.allow_overlap_types,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
