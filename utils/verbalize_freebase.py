"""Verbalize a Freebase TSV dump for prefix-tree ingestion."""

from __future__ import annotations

import bz2
import pickle
from pathlib import Path

import click
from tqdm import tqdm


TEMPLATE = "<{subject}> <{predicate}> <{object_}> .\n"


def verbalize_entity(entity_id: str, labels: dict[str, list]) -> str:
    """Create a unique display name while retaining the Freebase MID."""
    label = labels.get(entity_id, [None])[0]
    display_id = entity_id.replace(".", "/", 1)
    return f"{label} ({display_id})" if label else display_id


def verbalize(dump: Path, labels: dict[str, list], output: Path, total: int | None = None) -> int:
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

            subject = verbalize_entity(subject, labels)
            if object_.startswith("m."):
                object_ = verbalize_entity(object_, labels)
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
def main(
    freebase_labels: Path, dump: Path, output: Path, total_number_of_triples: int | None
) -> None:
    """Write verbalized Freebase triples to a compressed output file."""
    with freebase_labels.open("rb") as source:
        labels = pickle.load(source)
    count = verbalize(dump, labels, output, total_number_of_triples)
    click.echo(f"Wrote {count} triples to {output}")


if __name__ == "__main__":
    main()
