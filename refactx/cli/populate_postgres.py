import bz2
from tqdm import tqdm
from refactx import populate_postgres_index
from transformers import AutoTokenizer
import click

@click.command()
@click.argument('fname', type=click.Path(exists=True))
@click.option("--model-name", required=True, help="Name of the model to use.")
@click.option('--postgres-connection', type=str, required=True, help='PostgreSQL connection string')
@click.option('--table-name', type=str, required=True, help='Database table name')
@click.option("--prefix", required=True, help="Prefix used in processing.")
@click.option("--end-of-triple", required=True, help="End of triple marker.")
@click.option('--rootkey', type=int, required=True, help='Root key value')
@click.option("--tokenizer-batch-size", type=int, required=True, help="Batch size for tokenizer.")
@click.option("--batch-size", type=int, required=True, help='Batch size for ingestion (must be multiple of --tokenizer-batch-size).')
@click.option('--switch-parameter', type=int, required=True, help='Switch parameter value')
@click.option("--total-number-of-triples", type=int, default=None, help="Total number of items (optional).")
@click.option("--count-leaves", is_flag=True, default=False, help="Count leaves to veify the count is correct (slower).")
@click.option("--add-special-tokens", is_flag=True, default=False, help="Add special tokens when tokenizing.")
@click.option("--debug", is_flag=True, default=False, help="Break after first batch for debugging.")
def main(fname, model_name, postgres_connection, table_name, prefix, end_of_triple, rootkey,
    tokenizer_batch_size, batch_size, switch_parameter, total_number_of_triples, count_leaves, add_special_tokens, debug):
    """Command-line wrapper that delegates to refactx.populate_postgres_index."""

    assert batch_size % tokenizer_batch_size == 0, f'ERROR: --batch-size ({batch_size}) must be multiple of --tokenizer-batch-size ({tokenizer_batch_size})'

    if not end_of_triple.startswith(' '):
        print(f'WARNING: --end-of-triple ("{end_of_triple}") does not start with " "')

    # create tokenizer to pass into populate_postgres_index
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        print('WARNING: tokenizer is not fast.')

    # open the compressed file and pass the file reader to the helper
    with bz2.BZ2File(fname) as file_reader:
        populate_postgres_index(
            file_reader=file_reader,
            postgres_url=postgres_connection,
            tokenizer=tokenizer,
            table_name=table_name,
            batch_size=batch_size,
            rootkey=rootkey,
            switch_parameter=switch_parameter,
            total_number_of_triples=total_number_of_triples,
            prefix=prefix,
            tokenizer_batch_size=tokenizer_batch_size,
            add_special_tokens=add_special_tokens,
            count_leaves=count_leaves,
            debug=debug,
        )


if __name__ == "__main__":
    main()
