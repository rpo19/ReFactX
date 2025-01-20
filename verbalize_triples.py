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

import pickle
from tqdm import tqdm
import gzip
import json
import bz2
import re

with gzip.open('/workspace/data/filtered_props_dict.gz', 'r') as fd:
    prop_labels = json.load(fd)

with open('/workspace/data/ents_truthy_fix.pickle', 'rb') as fd: # TODO update
    ent_labels = pickle.load(fd)

wiki_dump = '/workspace/data/latest-truthy.nt.bz2'

outfile = '/workspace/data/verbalized.latest-truthy.nt.bz2'

triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
r'<(?:http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)|http:\/\/schema\.org\/description)>\s+' # added schema description # todo test
r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"([^"]+)"@en)\s+\.')

tqdm_params = {'total': 7923563616}
#tqdm_params = {}

with tqdm(**tqdm_params) as pbar:
    with bz2.BZ2File(outfile, 'w') as outfd:
        with bz2.BZ2File(wiki_dump, 'r') as fd:
            for count, bline in enumerate(fd):
                line = bline.decode()
                match = triple_regex.match(line)
                if match:
                    sub, prop, obj_ent, obj_lit = match.groups()

                    v_sub = ent_labels.get('Q'+sub, {}).get('label')
                    v_prop = prop_labels.get('P'+prop, {}).get('label')

                    if v_sub and v_prop:
                        if obj_ent:
                            # obj is entity
                            v_obj = ent_labels.get('Q'+obj_ent, {}).get('label')
                        if obj_lit:
                            # obj is literal
                            v_obj = obj_lit

                        if v_obj:
                            # verify v_obj is not None or ''
                            outfd.write(f'<{v_sub}> <{v_prop}> <{v_obj}> .\n'.encode())

                    # else:
                    #     # cannot find v_sub or v_prop --> skip
                    #     pass
                if count % 1000000 == 0:
                    pbar.n = count
                    pbar.refresh()


