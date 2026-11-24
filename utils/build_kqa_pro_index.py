"""Populate a PostgreSQL ReFactX prefix-tree index from the KQA Pro KB.

The input is the verbalized file produced by ``utils/prep_kqa_pro.py``.  This
wrapper fixes the conventions used by the existing Wikidata index (``qwen36``):
leading-space prefix, `` .`` triple terminator, rootkey -100 and
switch parameter 7.

Example::

    python utils/build_kqa_pro_index.py \
        /workspace/data/kqa_pro/kqa_pro_triples.txt.gz \
        --table-name kqapro_qwen36 \
        --model-name Qwen/Qwen3.6-27B
"""
from __future__ import annotations

import bz2
import gzip
import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import click
from dotenv import load_dotenv
from transformers import AutoTokenizer

from refactx import populate_postgres_index
from refactx.index import DictIndex, save_local_index


def opener_for(path: Path):
    if path.suffix == ".gz":
        return gzip.open
    if path.suffix == ".bz2":
        return bz2.open
    return open


def postgres_url_from_env() -> str:
    url = os.environ.get("POSTGRES_CONNECTION") or os.environ.get("POSTGRES_BASE_URL")
    if not url:
        raise click.ClickException(
            "Set POSTGRES_CONNECTION or POSTGRES_BASE_URL, or pass --postgres-connection."
        )
    parsed = urlparse(url)
    if parsed.path in ("", "/"):
        parsed = parsed._replace(path="/postgres")
    return urlunparse(parsed)


def count_lines(path: Path) -> int:
    opener = opener_for(path)
    with opener(path, "rb") as fd:
        return sum(1 for _ in fd)


@click.command()
@click.argument("triples", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--backend", type=click.Choice(["postgres", "local"]), default="postgres", show_default=True)
@click.option("--table-name", default="kqapro_qwen36", show_default=True)
@click.option("--output", type=click.Path(dir_okay=False, path_type=Path), default=None, help="Output .pkl path for --backend local.")
@click.option("--model-name", default="Qwen/Qwen3.6-27B", show_default=True)
@click.option(
    "--postgres-connection",
    default=None,
    help="PostgreSQL URL. Defaults to POSTGRES_CONNECTION or POSTGRES_BASE_URL.",
)
@click.option("--prefix", default=" ", show_default=True)
@click.option("--rootkey", default=-100, show_default=True)
@click.option("--switch-parameter", default=7, show_default=True)
@click.option("--batch-size", default=1_000_000, show_default=True)
@click.option("--tokenizer-batch-size", default=10_000, show_default=True)
@click.option("--count-leaves", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def main(triples: Path, backend: str, table_name: str, output: Path | None,
         model_name: str, postgres_connection: str | None, prefix: str,
         rootkey: int, switch_parameter: int, batch_size: int,
         tokenizer_batch_size: int, count_leaves: bool, debug: bool):
    load_dotenv()
    if batch_size % tokenizer_batch_size:
        raise click.ClickException(
            "--batch-size must be a multiple of --tokenizer-batch-size"
        )

    total = count_lines(triples)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        click.echo("WARNING: tokenizer is not fast.")

    opener = opener_for(triples)
    if backend == "local":
        if output is None:
            output = triples.with_suffix(".pkl")
        index = DictIndex()
        index.set_tokenizer(tokenizer)
        with opener(triples, "rt", encoding="utf-8") as fd:
            index.verbalized_triples = [
                prefix + line.rstrip("\n") for line in fd
            ]
        click.echo(f"Tokenizing {total:,} triples for local index")
        index.tokenize_triples(batch_size=tokenizer_batch_size)
        save_local_index(
            index,
            output,
            rootkey=rootkey,
            switch_parameter=switch_parameter,
            tokenizer_name=model_name,
        )
        click.echo(f"Done: local index {output}")
        return

    postgres_connection = postgres_connection or postgres_url_from_env()
    click.echo(f"Indexing {total:,} triples into table {table_name!r}")
    with opener(triples, "rb") as fd:
        populate_postgres_index(
            file_reader=fd,
            postgres_url=postgres_connection,
            tokenizer=tokenizer,
            table_name=table_name,
            batch_size=batch_size,
            rootkey=rootkey,
            switch_parameter=switch_parameter,
            total_number_of_triples=total,
            prefix=prefix,
            tokenizer_batch_size=tokenizer_batch_size,
            count_leaves=count_leaves,
            debug=debug,
        )
    click.echo(f"Done: table {table_name!r}")


if __name__ == "__main__":
    main()
