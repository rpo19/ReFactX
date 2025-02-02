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
outfile = sys.argv[2]

ents = {}

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
                    if sub not in ents:
                        ents[sub] = {'altlabels':set()}
                    if pred == 'http://www.w3.org/2000/01/rdf-schema#label':
                        assert 'label' not in ents[sub]
                        ents[sub]['label'] = obj
                    elif pred == 'http://www.w3.org/2004/02/skos/core#altLabel':
                        ents[sub]['altlabels'].add(obj)
                    elif pred == 'http://schema.org/description':
                        ents[sub]['description'] = obj

                if count % 10000 == 0:
                    pbar.n = count
                    pbar.refresh()
except EOFError:
    print('EOF error')

elapsed = time.time() - start
print('elapsed', elapsed)
# -

with open(outfile, 'wb') as fd:
    pickle.dump(ents, fd)

