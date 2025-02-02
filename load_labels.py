# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import gzip
import time
import re
from tqdm import tqdm
import pickle
import sys

triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/.*\/([QP][0-9]+)>\s+'
r'<(\S+)>\s+'
r'"([^"]+)"@en\s+\.')

labels_path = sys.argv[1] #'/workspace/data/latest-truthy-labels.nt.gz'
outfile_ents = sys.argv[2]
outfile_props = sys.argv[3]

ents = {}
props = {}

# +
start = time.time()

try:
    with gzip.open(labels_path, 'r') as fd:
        with tqdm() as pbar:
            for count, bline in enumerate(fd):
                line = bline.decode('unicode_escape') # correctly load unicode characters
                match = triple_regex.match(line)
                if match:
                    sub, pred, obj = match.groups()
                    res_id = int(sub[1:])
                    if sub[0] == 'Q':
                        mydict = ents
                    else:
                        mydict = props
                    if res_id not in mydict:
                        mydict[res_id] = ['',set(),''] # label, altLabels, description
                    if pred == 'http://www.w3.org/2000/01/rdf-schema#label':
                        assert mydict[res_id][0] == ''
                        mydict[res_id][0] = obj
                    elif pred == 'http://www.w3.org/2004/02/skos/core#altLabel':
                        mydict[res_id][1].add(obj)
                    elif pred == 'http://schema.org/description':
                        mydict[res_id][2] = obj

                if count % 10000 == 0:
                    pbar.n = count
                    pbar.refresh()
except EOFError:
    print('EOF error')

elapsed = time.time() - start
print('elapsed', elapsed)
# -

with open(outfile_ents, 'wb') as fd:
    pickle.dump(ents, fd)

with open(outfile_props, 'wb') as fd:
    pickle.dump(props, fd)

