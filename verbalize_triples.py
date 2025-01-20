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
import sys

props_mapping = sys.argv[1] # '/workspace/data/filtered_props_dict.gz'
entity_mapping = sys.argv[2] # '/workspace/data/wikidata_titles_mapping.pickle'
wiki_dump = sys.argv[3] # '/workspace/data/latest-truthy.nt.bz2'
outfile = sys.argv[4] # '/workspace/data/verbalized.latest-truthy.nt.bz2'
total_number_of_triples = sys.argv[5] if len(sys.argv) > 5 else None # 7923563616

with gzip.open(props_mapping, 'r') as fd:
    prop_labels = json.load(fd)

with open(entity_mapping, 'rb') as fd:
    ent_labels = pickle.load(fd)

triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
r'<(?:http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)|http:\/\/schema\.org\/description)>\s+' # added schema description # todo test
r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"([^"]+)"@en)\s+\.')

tqdm_params = {'total': int(total_number_of_triples)}

added_shortdesc = set()

with tqdm(**tqdm_params) as pbar:
    with bz2.BZ2File(outfile, 'w') as outfd:
        with bz2.BZ2File(wiki_dump, 'r') as fd:
            for count, bline in enumerate(fd):
                line = bline.decode('unicode_escape') # correctly load unicode characters
                match = triple_regex.match(line)
                if match:
                    sub, prop, obj_ent, obj_lit = match.groups()

                    if sub.isnumeric():
                        sub_id = int(sub)
                        v_sub = ent_labels.get(sub_id, {}).get('title')

                        # short description for subject entity
                        sub_short_desc = ent_labels.get(sub_id, {}).get('short_desc')
                        if sub_id not in added_shortdesc and sub_short_desc:
                            added_shortdesc.add(sub_id)
                            outfd.write(f'<{v_sub}> <short description> <{sub_short_desc}> .\n'.encode('utf-8'))

                        if v_sub:
                            if prop is None:
                                # description
                                assert obj_lit is not None
                                v_prop = 'description'
                                v_obj = obj_lit

                                outfd.write(f'<{v_sub}> <{v_prop}> <{v_obj}> .\n'.encode('utf-8'))
                            else:
                                v_prop = prop_labels.get('P'+prop, {}).get('label')
                                if v_sub and v_prop:
                                    if obj_ent and obj_ent.isnumeric():
                                        # obj is entity
                                        obj_id = int(obj_ent)
                                        v_obj = ent_labels.get(obj_id, {}).get('title')

                                        # short description for object entity
                                        obj_short_desc = ent_labels.get(obj_id, {}).get('short_desc')
                                        if obj_id not in added_shortdesc and obj_short_desc:
                                            added_shortdesc.add(obj_id)
                                            outfd.write(f'<{v_obj}> <short description> <{obj_short_desc}> .\n'.encode('utf-8'))
                                    if obj_lit:
                                        # obj is literal
                                        v_obj = obj_lit

                                    if v_obj:
                                        # verify v_obj is not None or ''
                                        outfd.write(f'<{v_sub}> <{v_prop}> <{v_obj}> .\n'.encode('utf-8'))

                    # else:
                    #     # cannot find v_sub or v_prop --> skip
                    #     pass
                if count % 1000000 == 0:
                    pbar.n = count
                    pbar.refresh()


