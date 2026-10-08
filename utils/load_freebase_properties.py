"""Extract Freebase property names from an English TSV dump into a pickle file.

Every triple in the dump has the shape ``subject<TAB>property<TAB>object`` and
the property is a hierarchical id such as ``music.recording.artist``.  This
script collects the property ids with how often each occurs and maps them to
simplified display names made of the final segment only (``artist``,
``place of birth``, ``alpha-2``), turning underscores into spaces so the name
reads as a phrase rather than a machine identifier.

Display names are unique by construction.  When several property ids share the
same final segment (``film.film.genre`` and ``music.artist.genre``), the most
frequent id keeps the short name and the others are lengthened one segment at a
time until unique (``genre`` vs ``artist genre``), so the common, canonical
property stays short.  Ids that cannot be told apart from the id alone add their
full id to the name and are the only ones flagged.

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


def simplify_property(property_id: str, segments: int = 1) -> str:
    """Reduce a property id to a human-readable name from its last segments.

    The domain and type prefix is usually implied by the surrounding triple, so
    one segment is the default, and underscores become spaces::

        music.recording.artist        -> "artist"
        people.person.place_of_birth  -> "place of birth"
        authority.iso.3166-1.alpha-2  -> "alpha-2"
        rdf-schema#domain             -> "domain"

    ``segments=2`` reads ``film.actor.film`` as ``actor film``, which is how
    colliding names are disambiguated.  Fewer segments than requested are used
    when the id is too short.
    """
    parts = property_segments(property_id)
    return " ".join(parts[-segments:])


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
    max_segments: int | None = None,
) -> tuple[dict[str, list], dict[str, list[str]]]:
    """Build ``{property_id: [display_name, keeps_full_id]}`` with unique names.

    Within each group of ids sharing a final segment, the most frequent id (ties
    prefer the id with fewer segments, then alphabetical order) keeps the bare
    name and the rest are lengthened one segment at a time until unique:

        film.film.genre      -> "genre"         (most common of the group)
        music.artist.genre   -> "artist genre"

    Ids that still collide at ``max_segments`` (or once all their segments are
    used) add their full id to the name and are flagged.  Returns the records and
    the remaining unresolvable collisions, which are normally empty.
    """
    counts = counts or {}
    segments = {pid: property_segments(pid) for pid in property_ids}

    grouped: dict[str, list[str]] = {}
    for pid in property_ids:
        grouped.setdefault(segments[pid][-1], []).append(pid)

    records: dict[str, list] = {}
    taken: set[str] = set()
    needed: dict[str, int] = {}
    pending: list[str] = []

    for bare, ids in grouped.items():
        if len(ids) == 1:
            records[ids[0]] = [bare, False]
            taken.add(bare)
            continue
        winner = min(ids, key=lambda pid: (-counts.get(pid, 0), len(segments[pid]), pid))
        records[winner] = [bare, False]
        taken.add(bare)
        needed[winner] = 1
        pending.extend(pid for pid in ids if pid != winner)

    with_id_suffix: set[str] = set()
    while pending:
        buckets: dict[str, list[str]] = {}
        for pid in pending:
            want = needed.get(pid, 1) + 1
            if want > len(segments[pid]) or (
                max_segments is not None and want > max_segments
            ):
                with_id_suffix.add(pid)
                continue
            buckets.setdefault(simplify_property(pid, want), []).append(pid)

        still_pending = []
        for name, ids in buckets.items():
            if len(ids) == 1 and name not in taken:
                pid = ids[0]
                needed[pid] = needed.get(pid, 1) + 1
                records[pid] = [name, False]
                taken.add(name)
            else:
                still_pending.extend(ids)
        pending = still_pending

    for pid in with_id_suffix:
        name = f"{simplify_property(pid)} ({pid})"
        records[pid] = [name, True]

    final = {pid: records[pid] for pid in property_ids}
    grouped_final: dict[str, list[str]] = {}
    for pid, record in final.items():
        grouped_final.setdefault(record[0], []).append(pid)
    collisions = {
        name: sorted(ids) for name, ids in grouped_final.items() if len(ids) > 1
    }
    return final, collisions


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
    "--max-segments",
    type=int,
    default=3,
    help=(
        "Cap on how many segments a disambiguated name may keep. Ids that still "
        "collide at the cap fall back to 'name (property_id)'."
    ),
)
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
    max_segments: int,
    collisions_report: Path | None,
    fail_on_collisions: bool,
) -> None:
    """Build a ``{property_id: [display_name, keeps_full_id]}`` pickle from a TSV dump."""
    counts = load_property_counts(dump, total_number_of_triples)
    records, collisions = unique_display_names(list(counts), counts, max_segments)

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
            f"{len(collisions)} display name(s) stayed ambiguous; rerun without "
            f"--fail-on-collisions to keep 'name (property_id)'."
        )

    write_pickle_atomically(records, output)
    click.echo(f"Wrote Freebase properties to {output}")


if __name__ == "__main__":
    main()
