"""Verbalize a Freebase TSV dump for prefix-tree ingestion."""

from __future__ import annotations

import bz2
import pickle
from pathlib import Path

import click
from tqdm import tqdm


TEMPLATE = "<{subject}> <{predicate}> <{object_}> .\n"

# Types that every entity carries and that therefore say nothing about it.
GENERIC_TYPE_PREFIXES = ("common.", "type.", "base.", "freebase.")


def is_generic_type(type_id: str) -> bool:
    normalized = type_id.replace("/", ".").lstrip(".")
    return normalized.startswith(GENERIC_TYPE_PREFIXES)


def type_display_name(type_id: str, labels: dict[str, list]) -> str:
    """Resolve a ``type.object.type`` value to a readable type name."""
    record = labels.get(type_id)
    if record and record[0]:
        return record[0]
    if type_id.startswith("m."):
        return type_id.replace(".", "/", 1)
    return type_id.replace("/", ".").lstrip(".")


def entity_type(entity_id: str, labels: dict[str, list]) -> str | None:
    """Return the most specific (first non-generic) type of an entity."""
    record = labels.get(entity_id)
    if not record or not record[1]:
        return None
    for type_id in record[1]:
        if not is_generic_type(type_id):
            return type_display_name(type_id, labels)
    return None


def verbalize_entity(
    entity_id: str, labels: dict[str, list], include_types: bool = False
) -> str:
    """Create a unique display name from label and Freebase MID.

    By default the name is ``label (id)``, or just the ``id`` when no label is
    available. When ``include_types`` is set, the most specific type is inserted
    before the id as ``label (type id)``.
    """
    record = labels.get(entity_id)
    label = record[0] if record else None
    display_id = entity_id.replace(".", "/", 1)
    if not include_types:
        if label:
            return f"{label} ({display_id})"
        return display_id
    type_name = entity_type(entity_id, labels)
    if label and type_name:
        return f"{label} ({type_name} {display_id})"
    if label:
        return f"{label} ({display_id})"
    return display_id


def verbalize(
    dump: Path,
    labels: dict[str, list],
    output: Path,
    total: int | None = None,
    include_types: bool = False,
) -> int:
    written = 0
    with dump.open("r", encoding="utf-8") as source, bz2.open(
        output, "wt", encoding="utf-8"
    ) as destination:
        for line in tqdm(source, total=total, desc="Verbalizing Freebase"):
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3:
                continue

            subject, predicate, object_ = fields
            if not subject.startswith("m."):
                continue

            subject = verbalize_entity(subject, labels, include_types)
            if object_.startswith("m."):
                object_ = verbalize_entity(object_, labels, include_types)
            destination.write(
                TEMPLATE.format(subject=subject, predicate=predicate, object_=object_)
            )
            written += 1
    return written


@click.command()
@click.option(
    "--freebase-labels",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Pickle produced by load_freebase_labels.py.",
)
@click.argument("dump", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--total-number-of-triples", type=int, default=None)
@click.option(
    "--include-types/--no-include-types",
    default=False,
    help="Include the entity's most specific type in its display name.",
)
def main(
    freebase_labels: Path,
    dump: Path,
    output: Path,
    total_number_of_triples: int | None,
    include_types: bool,
) -> None:
    """Write verbalized Freebase triples to a compressed output file."""
    with freebase_labels.open("rb") as source:
        labels = pickle.load(source)
    count = verbalize(dump, labels, output, total_number_of_triples, include_types)
    click.echo(f"Wrote {count} triples to {output}")


if __name__ == "__main__":
    main()
