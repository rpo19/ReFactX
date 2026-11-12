"""Create a deterministic random sample of dataset ids for ``eval.py``.

The output is a JSON list suitable for a config's ``sample_index_file`` key.
Use the same ``--dataset``/``--dataset-config``/``--dataset-revision``/``--split``
as the evaluation config so the ids resolve to the expected rows.

Example::

    python utils/make_sample_ids.py \
        --dataset rpozzi/mintaka --dataset-config default \
        --dataset-revision refs/convert/parquet --split test \
        --id-key id --n 100 --seed 42 \
        --output configs/samples/mintaka_test_100_seed42.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--id-key", default="id", help="Column holding a unique row id")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        revision=args.dataset_revision,
        split=args.split,
    )
    if args.n > len(dataset):
        raise ValueError(f"Requested {args.n} rows but split has {len(dataset)}")
    if args.id_key not in dataset.column_names:
        raise KeyError(
            f"id key {args.id_key!r} not in columns {dataset.column_names}"
        )

    ids = [row[args.id_key] for row in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError(f"id key {args.id_key!r} is not unique")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), args.n))
    sample = [ids[i] for i in indices]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        f"Wrote {len(sample)} ids from {args.dataset}"
        f" [{args.split}] to {output} (seed={args.seed})"
    )


if __name__ == "__main__":
    main()
