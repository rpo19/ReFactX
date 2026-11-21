"""Prepare the KQA Pro KB for ReFactX prefix-tree indexing.

KQA Pro ships its own dense Wikidata subset (``kb.json``); it is *not* built by
processing a full Wikidata dump at index time.  This script verbalizes that KB
into the ``<subject> <predicate> <object> .`` format consumed by
``utils/populate_postgres.py`` and converts the QA splits to JSONL for
``utils/eval.py``.

Example::

    python utils/prep_kqa_pro.py \
        --data-dir /workspace/data/kqa_pro \
        --out-dir /workspace/data/kqa_pro

The KB files can be downloaded from the HuggingFace mirror::

    huggingface-cli download drt/kqa_pro --repo-type dataset \
        --local-dir /workspace/data/kqa_pro
"""
from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import click


def clean_text(value) -> str:
    return " ".join(str(value).split())


def value_to_string(value: dict) -> str | None:
    kind = value.get("type")
    raw = value.get("value")
    if kind == "string":
        text = clean_text(raw)
        return text or None
    if kind == "quantity":
        number = float(raw)
        if abs(number - int(number)) < 1e-9:
            number = int(number)
        unit = clean_text(value.get("unit") or "")
        return f"{number} {unit}" if unit and unit != "1" else str(number)
    if kind == "year":
        return str(int(raw))
    if kind == "date":
        try:
            year, month, day = (int(part) for part in str(raw).split("/"))
        except ValueError:
            return clean_text(raw) or None
        return f"{year:04d}-{month:02d}-{day:02d}"
    return clean_text(raw) or None


def display_names(kb: dict) -> tuple[dict[str, str], dict]:
    names = {}
    for section in ("concepts", "entities"):
        for qid, info in kb[section].items():
            names[qid] = clean_text(info.get("name", "")) or qid
    counts = Counter(names.values())
    display = {
        qid: (f"{name} ({qid})" if counts[name] > 1 else name)
        for qid, name in names.items()
    }
    collisions = {
        "duplicate_names": sum(1 for count in counts.values() if count > 1),
        "max_duplicate_count": max(counts.values()) if counts else 0,
        "examples": [
            {"name": name, "count": count}
            for name, count in counts.most_common(10) if count > 1
        ],
    }
    return display, collisions


def iter_triples(kb: dict, names: dict[str, str], include_qualifiers: bool,
                 kinds: Counter | None = None):
    kinds = kinds if kinds is not None else Counter()

    def emit(subject, predicate, obj, kind):
        subject = clean_text(subject)
        predicate = clean_text(predicate)
        obj = clean_text(obj)
        if not subject or not predicate or not obj:
            return None
        if kinds is not None:
            kinds[kind] += 1
        return f"<{subject}> <{predicate}> <{obj}> ."

    for qid, info in kb["concepts"].items():
        for parent in info.get("instanceOf", []):
            if parent in names:
                line = emit(names[qid], "instance of", names[parent], "instance_of")
                if line:
                    yield line
    for qid, info in kb["entities"].items():
        for parent in info.get("instanceOf", []):
            if parent in names:
                line = emit(names[qid], "instance of", names[parent], "instance_of")
                if line:
                    yield line
        for attr in info.get("attributes", []):
            obj = value_to_string(attr.get("value", {}))
            if obj is not None:
                line = emit(names[qid], attr.get("key", ""), obj, "attribute")
                if line:
                    yield line
            if include_qualifiers:
                for qkey, qvalues in attr.get("qualifiers", {}).items():
                    for qvalue in qvalues:
                        value = value_to_string(qvalue)
                        if value is not None:
                            line = emit(
                                names[qid],
                                f"{attr.get('key', '')} ({qkey})",
                                value,
                                "attribute_qualifier",
                            )
                            if line:
                                yield line
        for rel in info.get("relations", []):
            obj_id = rel.get("object")
            if obj_id not in names:
                continue
            predicate = rel.get("predicate", "")
            if rel.get("direction") == "backward":
                subject, obj = names[obj_id], names[qid]
            else:
                subject, obj = names[qid], names[obj_id]
            line = emit(subject, predicate, obj, "relation")
            if line:
                yield line
            if include_qualifiers:
                for qkey, qvalues in rel.get("qualifiers", {}).items():
                    for qvalue in qvalues:
                        value = value_to_string(qvalue)
                        if value is not None:
                            line = emit(
                                subject,
                                f"{predicate} ({qkey})",
                                value,
                                "relation_qualifier",
                            )
                            if line:
                                yield line


def write_split(source: Path, destination: Path, split: str) -> int:
    data = json.loads(source.read_text(encoding="utf-8"))
    with destination.open("w", encoding="utf-8") as out:
        for index, sample in enumerate(data):
            row = {
                "id": f"{split}-{index}",
                "question": sample["question"],
                "choices": sample.get("choices", []),
                "program": sample.get("program", []),
                "sparql": sample.get("sparql", ""),
            }
            if "answer" in sample:
                row["answer"] = sample["answer"]
                choices = sample.get("choices", [])
                row["answer_index"] = (
                    choices.index(sample["answer"]) if sample["answer"] in choices else -1
                )
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(data)


@click.command()
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("/workspace/data/kqa_pro"),
    show_default=True,
    help="Directory containing kb.json, train.json, val.json and test.json.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (defaults to --data-dir).",
)
@click.option(
    "--triples-name",
    default="kqa_pro_triples.txt.gz",
    show_default=True,
    help="Verbalized triple file name inside --out-dir.",
)
@click.option(
    "--include-qualifiers/--no-include-qualifiers",
    default=False,
    show_default=True,
    help=(
        "Also emit qualifier values as '<subject> <predicate (qualifier key)> "
        "<value> .' triples. Disabled by default because ReFactX facts are "
        "plain subject-relation-object triples."
    ),
)
@click.option(
    "--write-splits/--no-write-splits",
    default=True,
    show_default=True,
    help="Convert train/val/test to JSONL for utils/eval.py.",
)
def main(data_dir: Path, out_dir: Path | None, triples_name: str,
         include_qualifiers: bool, write_splits: bool):
    out_dir = out_dir or data_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    kb = json.loads((data_dir / "kb.json").read_text(encoding="utf-8"))
    names, collisions = display_names(kb)

    kinds = Counter()
    triples = set(iter_triples(kb, names, include_qualifiers, kinds))
    triples_path = out_dir / triples_name
    with gzip.open(triples_path, "wt", encoding="utf-8") as out:
        for line in sorted(triples):
            out.write(line + "\n")

    split_counts = {}
    if write_splits:
        for split in ("train", "val", "test"):
            source = data_dir / f"{split}.json"
            if source.exists():
                split_counts[split] = write_split(source, out_dir / f"{split}.jsonl", split)

    stats = {
        "concepts": len(kb["concepts"]),
        "entities": len(kb["entities"]),
        "names": len(names),
        "unique_triples": len(triples),
        "triples_by_kind": dict(kinds),
        "include_qualifiers": include_qualifiers,
        "splits": split_counts,
        **collisions,
    }
    (out_dir / "kqa_pro_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    click.echo(f"Wrote {len(triples):,} unique triples to {triples_path}")
    for split, count in split_counts.items():
        click.echo(f"Wrote {count:,} {split} samples to {out_dir / (split + '.jsonl')}")
    click.echo(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
