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

#model_name = "microsoft/Phi-3-mini-128k-instruct"
model_name = sys.argv[1]
verbalized_path = sys.argv[2]
outfile = sys.argv[3]
total = int(sys.argv[4]) if len(sys.argv) > 4 else None

tokenizer = AutoTokenizer.from_pretrained(model_name)

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
            for count, bline in enumerate(fd):
                line = bline.decode()
                if line[-1] == '\n':
                    line = line[:-1]
                ids = tokenizer(line)['input_ids']

                pickle.dump(ids, fout)
    
                if count % 10000 == 0:
                    pbar.n = count
                    pbar.refresh()
