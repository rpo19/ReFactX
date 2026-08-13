"""Convert the MedQA-USMLE NetworkX graph to angle-bracket triples.

The graph stores entity IDs in ``ptrs.txt`` and their labels, in matching
order, in ``vocab.txt``.  Relation labels are integer edge attributes because
the dataset does not ship human-readable relation names.
"""

import argparse
import bz2
import pickle
from pathlib import Path

RELATION_LABELS = [
    "belongs_to_the_category_of",
    "is_a_category",
    "may_cause",
    "is_a_subtype_of",
    "is_a_risk_factor_of",
    "is_associated_with",
    "may_contraindicate",
    "interacts_with",
    "belongs_to_the_drug_family_of",
    "belongs_to_drug_super-family",
    "is_a_vector_for",
    "may_be_allelic_with",
    "see_also",
    "is_an_ingredient_of",
    "may_treat",
]


def load_entity_labels(ptrs_path: Path, vocab_path: Path) -> dict[int, str]:
    # Graph nodes are vocabulary row indices; ptrs contains the original IDs
    # (which may be UMLS strings such as C0002965), so it is only a row check.
    ptrs = [line.strip() for line in ptrs_path.read_text().splitlines() if line.strip()]
    labels = vocab_path.read_text().splitlines()
    if len(ptrs) != len(labels):
        raise ValueError(f"{ptrs_path} has {len(ptrs)} IDs but {vocab_path} has {len(labels)} labels")
    return dict(enumerate(labels))


def open_output(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")


def verbalize(graph_path: Path, ptrs_path: Path, vocab_path: Path, output_path: Path) -> int:
    with graph_path.open("rb") as fd:
        graph = pickle.load(fd)

    entity_labels = load_entity_labels(ptrs_path, vocab_path)
    written = 0
    with open_output(output_path) as out:
        for subject, object_, data in graph.edges(data=True):
            subject_label = entity_labels.get(subject, str(subject))
            object_label = entity_labels.get(object_, str(object_))
            relation = data.get("rel")
            if not isinstance(relation, int):
                raise ValueError(
                    f"Unknown relation {relation!r} on edge ({subject}, {object_})"
                )
            if 0 <= relation < len(RELATION_LABELS):
                relation_label = RELATION_LABELS[relation]
            elif len(RELATION_LABELS) <= relation < 2 * len(RELATION_LABELS):
                relation_label = f"inverse_of_{RELATION_LABELS[relation - len(RELATION_LABELS)]}"
            else:
                raise ValueError(
                    f"Unknown relation ID {relation} on edge ({subject}, {object_})"
                )
            out.write(f"<{subject_label}> <{relation_label}> <{object_label}> .\n")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--ptrs", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = verbalize(args.graph, args.ptrs, args.vocab, args.output)
    print(f"Wrote {count} triples to {args.output}")


if __name__ == "__main__":
    main()
