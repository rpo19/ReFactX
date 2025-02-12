# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (conda base)
#     language: python
#     name: condabase
# ---

# +
# pip install bitsandbytes accelerate flash-attn

# +
import torch
import transformers
import bz2
import pickle
from transformers import AutoTokenizer #, CodeGenTokenizer

from tqdm import tqdm


import time
from IPython.display import JSON
import sys
sys.settrace(None)
import pdb
# -

model_name = sys.argv[1]
verbalized_path = sys.argv[2]
outfile = sys.argv[3]
prefix = sys.argv[4]
endoftriple = sys.argv[5]
batchsize = int(sys.argv[6])
total = int(sys.argv[7]) if len(sys.argv) > 7 else None

tokenizer = AutoTokenizer.from_pretrained(model_name)
assert tokenizer.is_fast

# +
from typing import List
from transformers import PreTrainedModel
# from transformers.generation.beam_constraints import DisjunctiveTrie
import torch
# -

# load the triples

with bz2.BZ2File(outfile, 'wb') as fout:
    with bz2.BZ2File(verbalized_path) as fd:
        with tqdm(total=total) as pbar:
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
