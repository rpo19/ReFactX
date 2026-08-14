"""Verbalize a ConceptNet assertions dump as simple triples."""

import argparse
import csv
import gzip
import json
from pathlib import Path
from urllib.parse import unquote


def concept_label(uri: str) -> str:
    parts = uri.strip().split("/")
    if len(parts) < 4 or parts[1] != "c":
        return unquote(uri)
    language, term = parts[2], unquote(parts[3])
    label = term.replace("_", " ")
    return label if language == "en" else f"{language}:{label}"


def relation_label(uri: str) -> str:
    parts = uri.strip().split("/")
    return unquote(parts[-1]) if parts else unquote(uri)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def format_part(value: str, brackets: bool) -> str:
    return f"<{value}>" if brackets else value


def verbalize(input_path: Path, output_path: Path, brackets: bool) -> int:
    written = 0
    with open_text(input_path, "r") as source, open_text(output_path, "w") as target:
        reader = csv.reader(source, delimiter="\t")
        for line_number, row in enumerate(reader, start=1):
            if len(row) != 5:
                raise ValueError(f"Expected 5 columns at line {line_number}, got {len(row)}")
            _, relation, subject, object_, _metadata = row
            # Validate the metadata while streaming so malformed assertions fail loudly.
            json.loads(_metadata)
            target.write(
                f"{format_part(concept_label(subject), brackets)} "
                f"{format_part(relation_label(relation), brackets)} "
                f"{format_part(concept_label(object_), brackets)} .\n"
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--brackets", dest="brackets", action="store_true")
    group.add_argument("--no-brackets", dest="brackets", action="store_false")
    parser.set_defaults(brackets=True)
    args = parser.parse_args()
    count = verbalize(args.input, args.output, args.brackets)
    print(f"Wrote {count} triples to {args.output}")


if __name__ == "__main__":
    main()
