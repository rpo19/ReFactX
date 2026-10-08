"""Extract Freebase property names from an English TSV dump into a pickle file.

Every triple in the dump has the shape ``subject<TAB>property<TAB>object`` and
the property is a hierarchical id such as ``music.recording.artist``.  This
script maps each property id to a simplified display name made of its final
segment (``artist``, ``place of birth``, ``alpha-2``), turning underscores into
spaces so the name reads as a phrase rather than a machine identifier.

Display names are unique by construction.  When several property ids share the
same final segment (``film.film.genre`` and ``music.artist.genre``) the bare
segment is ambiguous, so the most frequent id keeps the short name and the
others append their full id (``genre`` vs ``genre (music.artist.genre)``).

The pickle mirrors ``load_freebase_labels.py`` and maps ``{property_id:
[display_name, keeps_full_id]}``; ``display_name`` is always unique, so
``verbalize_freebase.py`` can use it directly as the predicate.
"""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path

import click
from tqdm import tqdm

# A property id is a chain of segments joined by ``.`` (``people.person.place_of_birth``).
# A few RDF-ish predicates use ``/`` or ``#`` instead (``rdf-schema#domain``).
_SEGMENT_SEPARATOR_RE = re.compile(r"[./#]")
_WHITESPACE_RE = re.compile(r"\s+")


def property_segments(property_id: str) -> list[str]:
    """Split a property id into readable segments.

    ``music.recording.artist`` -> ``["music", "recording", "artist"]``
    """
    return [
        _WHITESPACE_RE.sub(" ", segment.replace("_", " ")).strip()
        for segment in _SEGMENT_SEPARATOR_RE.split(property_id)
        if segment.strip()
    ]


def simplify_property(property_id: str) -> str:
    """Reduce a property id to its final, most specific segment as a phrase.

    The domain and type prefix is implied by the surrounding triple, so only the
    last segment is kept and underscores become spaces::

        music.recording.artist        -> "artist"
        people.person.place_of_birth  -> "place of birth"
        authority.iso.3166-1.alpha-2  -> "alpha-2"
        rdf-schema#domain             -> "domain"
    """
    return property_segments(property_id)[-1]


def load_property_counts(dump: Path, total: int | None = None) -> dict[str, int]:
    """Return ``{property_id: number of triples}`` for every property in ``dump``.

    Lines that do not have exactly three tab-separated fields are counted and
    skipped.
    """
    counts: dict[str, int] = {}
    errors = 0

    with dump.open("r", encoding="utf-8") as source:
        for line in tqdm(source, total=total, desc="Loading Freebase properties"):
            fields = line.rstrip("\n").split("\t", 2)
            if len(fields) != 3:
                errors += 1
                continue

            property_id = fields[1]
            if property_id:
                counts[property_id] = counts.get(property_id, 0) + 1

    click.echo(f"Skipped {errors} malformed lines")
    return counts


def unique_display_names(
    property_ids: list[str],
    counts: dict[str, int] | None = None,
) -> tuple[dict[str, list], dict[str, list[str]]]:
    """Build ``{property_id: [display_name, keeps_full_id]}`` with unique names.

    A property keeps its final segment when no other property shares it:

        people.person.place_of_birth -> "place of birth"

    When several properties share a final segment the bare name is ambiguous, so
    the most frequent id keeps it and the rest append their full id and are
    flagged (ties are broken alphabetically):

        film.film.genre    -> "genre"          (most frequent of the group)
        music.artist.genre -> "genre (music.artist.genre)"

    Returns the records and any remaining collisions, which are impossible by
    construction and therefore normally empty.
    """
    counts = counts or {}
    segments = {pid: property_segments(pid)[-1] for pid in property_ids}

    grouped: dict[str, list[str]] = {}
    for pid in property_ids:
        grouped.setdefault(segments[pid], []).append(pid)

    records: dict[str, list] = {}
    for segment, ids in grouped.items():
        if len(ids) == 1:
            records[ids[0]] = [segment, False]
            continue
        winner = min(ids, key=lambda pid: (-counts.get(pid, 0), pid))
        for pid in ids:
            if pid == winner:
                records[pid] = [segment, False]
            else:
                records[pid] = [f"{segment} ({pid})", True]

    grouped_names: dict[str, list[str]] = {}
    for pid, record in records.items():
        grouped_names.setdefault(record[0], []).append(pid)
    collisions = {
        name: sorted(ids) for name, ids in grouped_names.items() if len(ids) > 1
    }
    return records, collisions


def report_collisions(
    collisions: dict[str, list[str]], preview: int = 20, max_ids: int = 8
) -> None:
    """Print the collisions that could not be resolved into unique names."""
    if not collisions:
        click.echo("All display names are unique")
        return

    click.echo(f"{len(collisions)} display name(s) could not be made unique")
    ranked = sorted(collisions.items(), key=lambda item: (-len(item[1]), item[0]))
    for name, ids in ranked[:preview]:
        shown = ", ".join(ids[:max_ids])
        if len(ids) > max_ids:
            shown += f", ... (+{len(ids) - max_ids} more)"
        click.echo(f"  {name} ({len(ids)}): {shown}")


def write_collisions_report(collisions: dict[str, list[str]], output: Path) -> None:
    """Write one ``display_name<TAB>comma separated ids`` line per collision."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as destination:
        for name, ids in sorted(collisions.items()):
            destination.write(f"{name}\t{', '.join(ids)}\n")


def write_pickle_atomically(records: dict[str, list], output: Path) -> None:
    """Write the pickle through a temp file so an interrupted run can't corrupt it."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=output.parent, prefix=f"{output.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "wb") as destination:
            pickle.dump(records, destination, protocol=pickle.HIGHEST_PROTOCOL)
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
@click.argument("dump", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output", type=click.Path(dir_okay=False, path_type=Path))
@click.option("--total-number-of-triples", type=int, default=None)
@click.option(
    "--collisions-report",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write the names that could not be made unique here.",
)
@click.option(
    "--fail-on-collisions",
    is_flag=True,
    default=False,
    help="Abort instead of writing the pickle when a name stays ambiguous.",
)
def main(
    dump: Path,
    output: Path,
    total_number_of_triples: int | None,
    collisions_report: Path | None,
    fail_on_collisions: bool,
) -> None:
    """Build a ``{property_id: [display_name, keeps_full_id]}`` pickle from a TSV dump."""
    counts = load_property_counts(dump, total_number_of_triples)
    records, collisions = unique_display_names(list(counts), counts)

    unique_names = len({record[0] for record in records.values()})
    click.echo(
        f"Found {len(records)} unique properties "
        f"({unique_names} unique display names)"
    )
    report_collisions(collisions)

    if collisions_report is not None:
        write_collisions_report(collisions, collisions_report)
        click.echo(f"Wrote collisions report to {collisions_report}")

    if collisions and fail_on_collisions:
        raise click.ClickException(
            f"{len(collisions)} display name(s) stayed ambiguous; names are "
            f"unique by construction, so this indicates a bug."
        )

    write_pickle_atomically(records, output)
    click.echo(f"Wrote Freebase properties to {output}")


if __name__ == "__main__":
    main()
