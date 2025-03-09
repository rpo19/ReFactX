import bz2
import psycopg
from tqdm import tqdm
from ctrie import PostgresIngestIndex
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
@click.option('--batch-size', type=int, required=True, help='Batch size for ingestion (must be multiple of --tokenizer-batch-size).')
@click.option('--switch-parameter', type=int, required=True, help='Switch parameter value')
@click.option("--total-number-of-triples", type=int, default=None, help="Total number of items (optional).")
@click.option("--count-leaves", is_flag=True, default=False, help="Count leaves to veify the count is correct (slower).")
@click.option("--add-special-tokens", is_flag=True, default=False, help="Add special tokens when tokenizing.")
@click.option("--debug", is_flag=True, default=False, help="Break after first batch for debugging.")
def main(fname, model_name, postgres_connection, table_name, prefix, end_of_triple, rootkey,
    tokenizer_batch_size, batch_size, switch_parameter, total_number_of_triples, count_leaves, add_special_tokens, debug):
    """Command-line tool for processing data and storing it in a PostgreSQL database."""

    assert batch_size % tokenizer_batch_size == 0, f'ERROR: --batch-size ({batch_size}) must be multiple of --tokenizer-batch-size ({tokenizer_batch_size})'

    if not end_of_triple.startswith(' '):
        print(f'WARNING: --end-of-triple ("{end_of_triple}") does not start with " "')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        print('WARNING: tokenizer is not fast.')

    index = PostgresIngestIndex(
                rootkey=rootkey,
                switch_parameter=switch_parameter,
                table_name=table_name)

    def batch_append(nested_token_ids, index):
        for sequence in nested_token_ids:
            index.add(sequence)

    tbar_update = batch_size
    count = 0

    with psycopg.connect(postgres_connection, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Create table and ensure index
            # and pkey are not present for fast ingestion
            cur.execute(index.create_table_query)
            cur.execute(index.drop_pkey_query)
            cur.execute(index.drop_index_query)
            conn.commit()

            cur.execute(index.check_indexes_query)
            count_indexes = cur.fetchone()[0]
            assert count_indexes == 0, f"Expected 0 indexes, but found {count_indexes}"

            cur.execute(index.truncate_query)
            with cur.copy(index.copy_query) as copy:
                with bz2.BZ2File(fname) as bz2file:
                    with tqdm(total=total_number_of_triples) as pbar:
                        tokenizer_batch = []
                        for count, bline in enumerate(bz2file):
                            try:
                                line = bline.decode()
                                if line[-1] == '\n':
                                    line = line[:-1]

                                if not line.endswith(end_of_triple):
                                    print(f'WARNING: line ({line}) w/o --end-of-triple ({end_of_triple})')
                                    line = line + end_of_triple

                                line = prefix + line

                                tokenizer_batch.append(line)

                                if len(tokenizer_batch) == tokenizer_batch_size:
                                    ids = tokenizer(tokenizer_batch, add_special_tokens=add_special_tokens)['input_ids']
                                    tokenizer_batch = []

                                    batch_append(ids, index)

                                if count % batch_size == 0 and count > 0:
                                    # batch on number or rows processed
                                    for row in index.get_rows():
                                        copy.write_row(row)

                                    if count_leaves:
                                        try:
                                            lfc = index.count_leaves(fail_on_wrong_num=True)
                                        except WrongNumleavesException as e:
                                            print(e, 'at', count)

                                    # reset batch
                                    index.reset()

                                    if debug:
                                        print('DEBUG! Breaking after first batch.')
                                        break

                                if count % tbar_update == 0:
                                    pbar.n = count
                                    pbar.refresh()

                            except EOFError:
                                print('Reached end of file.')
                                break  # End of file reached
                            except Exception as e:
                                print(f'Encountered exception at {count}')
                                raise e

            conn.commit()

            print('Ingestion finished.')
            print('Creating index.')
            cur.execute(index.create_index_query)
            print('Creating primary key.')
            cur.execute(index.create_pkey_query)
            conn.commit()

if __name__ == "__main__":
    main()