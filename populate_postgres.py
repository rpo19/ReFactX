import pickle
import bz2
import psycopg
from tqdm import tqdm
import pickle
from ctrie import PostgresIngestIndex
import click

@click.command()
@click.argument('fname', type=click.Path(exists=True))
@click.option('--postgres-connection', type=str, required=True, help='PostgreSQL connection string')
@click.option('--table-name', type=str, required=True, help='Database table name')
@click.option('--rootkey', type=int, required=True, help='Root key value')
@click.option('--batch-size', type=int, required=True, help='Batch size for processing')
@click.option('--switch-parameter', type=int, required=True, help='Switch parameter value')
@click.option('--total-rows', type=int, required=False, help='Total number of rows (optional)')
def main(fname, postgres_connection, table_name, rootkey, batch_size, switch_parameter, total_rows):
    """Command-line tool for processing data and storing it in a PostgreSQL database."""
    click.echo(f"Processing file: {fname}")
    click.echo(f"Database connection: {postgres_connection}")
    click.echo(f"Table name: {table_name}")
    click.echo(f"Root key: {rootkey}")
    click.echo(f"Batch size: {batch_size}")
    click.echo(f"Switch parameter: {switch_parameter}")
    if total_rows is not None:
        click.echo(f"Total rows: {total_rows}")
    else:
        click.echo("Total rows: Not provided")

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
                with bz2.BZ2File(fname, "rb") as bz2file:
                    with tqdm(total=total_rows) as pbar:
                        while True:
                            try:
                                # Load each pickled object from the bz2 file
                                array = pickle.load(bz2file)
                                batch_append(array, index)

                                count += 1

                                if count % batch_size == 0:
                                    # batch on number or rows processed
                                    for row in index.get_rows():
                                        copy.write_row(row)
                                    # reset batch
                                    index.reset()

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