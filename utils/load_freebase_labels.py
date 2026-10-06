"""Load Freebase English names and Wikipedia titles into a pickle file."""

from __future__ import annotations

import os
import pickle
import re
import tempfile
from pathlib import Path

import click
from tqdm import tqdm


NT_SUBJECT_PREFIX = "<http://rdf.freebase.com/ns/"
NT_EN_TITLE_PREDICATE = "<http://rdf.freebase.com/key/wikipedia.en_title>"

# FastRDFStore escapes remarkable characters in keys/titles as ``$XXXX`` (four
# hex digits, e.g. ``$00E9`` for ``é`` and ``$002C`` for ``,``).
_FASTRDF_ESCAPE_RE = re.compile(r"\$([0-9A-Fa-f]{4})")
# Standard N-Triples escapes that may still show up in a value.
_NT_CHAR_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "'": "'"}
_NT_ESCAPE_RE = re.compile(r"\\(u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8}|.)")


def unescape_nt_value(value: str) -> str:
    """Decode the ``$XXXX`` and ``\\uXXXX`` escapes used in Freebase literals."""
    value = _FASTRDF_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), value)

    def _replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token[0] in "uU":
            return chr(int(token[1:], 16))
        return _NT_CHAR_ESCAPES.get(token, token)

    return _NT_ESCAPE_RE.sub(_replace, value)


def parse_en_title_line(line: str) -> tuple[str, str] | None:
    """Parse a ``key/wikipedia.en_title`` N-Triples line into ``(mid, title)``.

    Lines look like::

        <http://rdf.freebase.com/ns/m.06jz1m>\t<http://rdf.freebase.com/key/wikipedia.en_title>\t"Qiu_Ying"\t.
    """
    fields = line.rstrip("\n").split("\t")
    if len(fields) < 3:
        return None
    subject, predicate, value = fields[0], fields[1], fields[2]
    if predicate != NT_EN_TITLE_PREDICATE:
        return None
    if not subject.startswith(NT_SUBJECT_PREFIX) or not subject.endswith(">"):
        return None

    mid = subject[len(NT_SUBJECT_PREFIX) : -1]
    if value.startswith('"'):
        value = value[1 : value.rfind('"')]
    return mid, unescape_nt_value(value)


def load_en_titles(en_titles_path: Path) -> dict[str, str]:
    """Return ``{mid: wikipedia en_title}`` from a filtered N-Triples dump."""
    titles: dict[str, str] = {}
    with en_titles_path.open("r", encoding="utf-8") as source:
        for line in tqdm(source, desc="Loading Freebase en_titles"):
            parsed = parse_en_title_line(line)
            if parsed is None:
                continue
            mid, title = parsed
            if title:
                titles.setdefault(mid, title)
    return titles


def load_labels(
    labels_path: Path,
    en_titles_path: Path | None = None,
    fail_on_duplicates: bool = True,
) -> dict[str, list]:
    """Return ``{mid: [label, en_title, en_title_unique]}`` records.

    ``label`` is the entity's ``type.object.name`` (empty string when unknown),
    ``en_title`` is its ``wikipedia.en_title`` (empty string when unknown) and
    ``en_title_unique`` is ``False`` when another entity shares the same title.

    Raises ``ValueError`` when two entities share a Wikipedia title, unless
    ``fail_on_duplicates`` is ``False`` (in which case such titles are flagged as
    ambiguous so the id can be appended during verbalization).
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
            if predicate != "type.object.name":
                continue
            record = labels.get(subject)
            if record is None:
                labels[subject] = [value, "", True]
            elif not record[0]:
                record[0] = value

    click.echo(f"Skipped {errors} malformed lines")
    if en_titles_path is not None:
        duplicates = attach_en_titles(labels, load_en_titles(en_titles_path))
        if duplicates and fail_on_duplicates:
            preview = ", ".join(
                f"{title!r} ({count})" for title, count in sorted(duplicates.items())[:10]
            )
            raise ValueError(
                f"{len(duplicates)} duplicate Wikipedia title(s): {preview}. "
                f"Pass --no-fail-on-duplicates-id to disambiguate them by "
                f"appending the id."
            )
    return labels


def attach_en_titles(labels: dict[str, list], en_titles: dict[str, str]) -> dict[str, int]:
    """Add Wikipedia titles to ``labels`` and flag the ambiguous ones.

    Returns the titles shared by more than one entity mapped to their counts.
    """
    title_counts: dict[str, int] = {}
    for mid, title in en_titles.items():
        record = labels.get(mid)
        if record is None:
            record = ["", "", True]
            labels[mid] = record
        if not record[1]:
            record[1] = title
            title_counts[title] = title_counts.get(title, 0) + 1

    duplicates = {title: count for title, count in title_counts.items() if count > 1}
    for record in labels.values():
        title = record[1]
        if title and title in duplicates:
            record[2] = False
    return duplicates


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
@click.option(
    "--en-titles",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Filtered N-Triples dump of key/wikipedia.en_title facts.",
)
@click.option(
    "--fail-on-duplicates-id/--no-fail-on-duplicates-id",
    default=True,
    help=(
        "Fail when a Wikipedia title is shared by multiple entities; disable to "
        "disambiguate duplicates by appending the id instead."
    ),
)
def main(
    labels_path: Path,
    output: Path,
    en_titles: Path | None,
    fail_on_duplicates_id: bool,
) -> None:
    """Convert a filtered Freebase label dump to a pickle file."""
    try:
        labels = load_labels(labels_path, en_titles, fail_on_duplicates_id)
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    write_pickle_atomically(labels, output)
    click.echo(f"Wrote Freebase labels to {output}")


if __name__ == "__main__":
    main()
