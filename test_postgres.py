import psycopg
import pickle
from transformers import AutoTokenizer
import sys
import random
import time
import json
import pickle
import click
from ctrie import PostgresTrieIndex

class TimeMeasure:
    def __init__(self, tag='default', verbose=False, outfile=sys.stdout):
        self.tag = tag
        self.verbose = verbose
        self.start_time = None
        self.duration = None
        self.outfile = outfile

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.verbose:
            print(self.tag, 'duration:', self.duration_ms(self.duration), file=self.outfile)

    def duration_ms(self, duration):
        return "{:.2f} ms".format(duration * 1000)

def merge(dst, src):
    for key in src:
        if key in dst:
            # sum numleaves
            dst[key][0] += src[key][0]
            merge(dst[key][1], src[key][1])
        else:
            # If the key exists only in `src`, the value from the `src` object will be used.
            dst[key] = src[key]

def choose(children, initial_tokens):
    if len(initial_tokens) > 0:
        choice = initial_tokens.pop(0)
        # if not choice in children:
        #     import pdb
        #     pdb.set_trace()
        assert choice in children
    else:
        choice = random.choice(children)
    return choice

@click.command()
@click.option("--postgres-connection", required=True, help="Postgres connection string")
@click.option("--table-name", required=True, help="Table name")
@click.option("--rootkey", type=int, required=True, help="Root key")
@click.option("--end-of-triple", type=int, required=True, help="End of triple")
@click.option("--model-name", required=True, help="Model name")
@click.option("--switch-parameter", type=int, required=True, help="Switch parameter")
@click.option("--random-seed", type=int, required=False, help="Random seed")
@click.option("--initial-tokens", default='', help="Initial tokens")
@click.option("--json-tokens", required=False, help="JSON tokens")
def main(postgres_connection, table_name, rootkey, end_of_triple, model_name, switch_parameter, random_seed, initial_tokens, json_tokens):
    if json_tokens:
        assert not initial_tokens, 'ERROR: specify either intitial tokens or json tokens. Not both.'
        initial_tokens = json.loads(json_tokens)
        assert isinstance(initial_tokens, list)

    click.echo(f"Postgres Connection: {postgres_connection}")
    click.echo(f"Table Name: {table_name}")
    click.echo(f"Root Key: {rootkey}")
    click.echo(f"End of Triple: {end_of_triple}")
    click.echo(f"Model Name: {model_name}")
    click.echo(f"Switch Parameter: {switch_parameter}")
    click.echo(f"Random Seed: {random_seed}")
    click.echo(f"Initial Tokens: {initial_tokens}")

    if random_seed is not None:
        random.seed(random_seed)

    print('Start.')

    with TimeMeasure(tag='Loading tokenizer', verbose=True) as tm:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if initial_tokens:
        print('Initial tokens:', initial_tokens)
        print(tokenizer.decode(initial_tokens))

    if isinstance(initial_tokens, str):
        initial_tokens = tokenizer(initial_tokens)['input_ids']

    conn = psycopg.connect(postgres_connection, autocommit=False)

    index = PostgresTrieIndex(rootkey=rootkey,
                                postgresql_connection=conn,
                                switch_parameter=switch_parameter,
                                table_name=table_name,
                                end_of_triple=end_of_triple)

    sentence = []

    while True:
        with TimeMeasure(tag=f'Query {len(sentence)}', verbose=True) as tm:
            print(sentence)
            print(tokenizer.decode(sentence))
            # if sentence == [366, 60704, 29, 366]:
            #     import pdb
            #     pdb.set_trace()
            possible_tokens = index.next_tokens(sentence)



        possible_tokens = list(possible_tokens.keys()) if possible_tokens else []

        if len(possible_tokens) > 0:
            next_token = choose(possible_tokens, initial_tokens)
            sentence.append(next_token)
        else:
            print('.')
            break

    print(tokenizer.decode(sentence))


if __name__ == "__main__":
    main()




