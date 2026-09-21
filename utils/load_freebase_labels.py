"""Load Freebase English names and descriptions into a pickle file."""

from __future__ import annotations

import pickle
from pathlib import Path

import click
from tqdm import tqdm


def load_labels(labels_path: Path) -> dict[str, list]:
    """Return ``{mid: [label, alternative_labels, description]}`` records."""
    labels: dict[str, list] = {}
    errors = 0

    with labels_path.open("r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Loading Freebase labels"):
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3:
                errors += 1
                continue

            subject, predicate, value = fields
            record = labels.setdefault(subject, ["", set(), ""])
            if predicate == "type.object.name":
                if record[0]:
                    record[1].add(value)
                else:
                    record[0] = value
            elif predicate == "common.topic.description":
                # Freebase may have multiple descriptions; use the shortest.
                if not record[2] or len(value) < len(record[2]):
                    record[2] = value

    click.echo(f"Skipped {errors} malformed lines")
    return labels


@click.command()
@click.argument("labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def main(labels_path: Path, output: Path) -> None:
    """Convert a filtered Freebase label dump to a pickle file."""
    with output.open("wb") as destination:
        pickle.dump(load_labels(labels_path), destination)
    click.echo(f"Wrote Freebase labels to {output}")


if __name__ == "__main__":
    main()
