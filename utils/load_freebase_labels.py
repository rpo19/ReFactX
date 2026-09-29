"""Load Freebase English names and types into a pickle file."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path

import click
from tqdm import tqdm


def load_labels(labels_path: Path) -> dict[str, list]:
    """Return ``{mid: [label, types]}`` records.

    ``label`` is the entity's ``type.object.name`` (empty string when unknown)
    and ``types`` is a list of its ``type.object.type`` values in encounter
    order, or ``None`` while it has none.  The type of a type is itself an
    entity in ``labels``, so a type value can be resolved to its human-readable
    name through the same dictionary.
    """
    labels: dict[str, list] = {}
    errors = 0

    with labels_path.open("r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Loading Freebase labels"):
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3:
                errors += 1
                continue

            subject, predicate, value = fields
            record = labels.get(subject)
            if predicate == "type.object.name":
                if record is None:
                    labels[subject] = [value, None]
                elif not record[0]:
                    record[0] = value
            elif predicate == "type.object.type":
                if record is None:
                    labels[subject] = ["", [value]]
                elif record[1] is None:
                    record[1] = [value]
                elif value not in record[1]:
                    record[1].append(value)

    click.echo(f"Skipped {errors} malformed lines")
    return labels


def write_pickle_atomically(labels: dict[str, list], output: Path) -> None:
    """Write the pickle through a temp file so an interrupted run can't corrupt it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=output.parent, prefix=f"{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            pickle.dump(labels, destination, protocol=pickle.HIGHEST_PROTOCOL)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(tmp_name, output)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@click.command()
@click.argument("labels_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
def main(labels_path: Path, output: Path) -> None:
    """Convert a filtered Freebase label dump to a pickle file."""
    write_pickle_atomically(load_labels(labels_path), output)
    click.echo(f"Wrote Freebase labels to {output}")


if __name__ == "__main__":
    main()
