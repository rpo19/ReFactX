import bz2
import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
import click

@click.command()
@click.argument("verbalized_path")
@click.argument("outfile")
@click.option("--model-name", required=True, help="Name of the model to use.")
@click.option("--prefix", required=True, help="Prefix used in processing.")
@click.option("--end-of-triple", required=True, help="End of triple marker.")
@click.option("--batchsize", type=int, required=True, help="Batch size for processing.")
@click.option("--total-number-of-triples number", type=int, default=None, help="Total number of items (optional).")
def main(verbalized_path, outfile, model_name, prefix, endoftriple, batchsize, total_number_of_triples):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        print('WARNING: tokenizer is not fast.')

    with bz2.BZ2File(outfile, 'wb') as fout:
        with bz2.BZ2File(verbalized_path) as fd:
            with tqdm(total=total_number_of_triples) as pbar:
                batch = []
                for count, bline in enumerate(fd):
                    line = bline.decode()
                    if line[-1] == '\n':
                        line = line[:-1]

                    if not line.endswith(endoftriple):
                        print(f'WARNING: {line} w/o end-of-triple {endoftriple}')
                        line = line + endoftriple

                    line = prefix + line

                    batch.append(line)

                    if len(batch) > batchsize:
                        ids = tokenizer(batch)['input_ids']
                        batch = []
                        pickle.dump(ids, fout)

                    if count % batchsize == 0:
                        pbar.n = count
                        pbar.refresh()

if __name__ == "__main__":
    main()