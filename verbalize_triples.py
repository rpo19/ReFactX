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
wikidata_labels = sys.argv[2] # wikidata-labels.pickle
wikipedia_entity_mapping = sys.argv[3] # '/workspace/data/wikidata_titles_mapping.pickle'
wiki_dump = sys.argv[4] # '/workspace/data/latest-truthy.nt.bz2'
outfile = sys.argv[5] # '/workspace/data/verbalized.latest-truthy.nt.bz2'
total_number_of_triples = sys.argv[6] if len(sys.argv) > 6 else None # 7923563616

with gzip.open(props_mapping, 'r') as fd:
    prop_labels = json.load(fd)

with open(wikidata_labels, 'rb') as fd:
    ent_labels_wikidata = pickle.load(fd)

with open(wikipedia_entity_mapping, 'rb') as fd:
    ent_labels_wikipedia = pickle.load(fd)

triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
r'<http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)>\s+'
r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"([^"]+)"@en|"([^"]+)"\^\^<.+>)\s+\.')

tqdm_params = {'total': int(total_number_of_triples)}

added_shortdesc = set()
added_desc = set()

with tqdm(**tqdm_params) as pbar:
    with bz2.BZ2File(outfile, 'w') as outfd:
        with bz2.BZ2File(wiki_dump, 'r') as fd:
            for count, bline in enumerate(fd):
                line = bline.decode('unicode_escape') # correctly load unicode characters
                match = triple_regex.match(line)
                if match:
                    sub, prop, obj_ent, obj_lit_en, obj_lit_datatype = match.groups()

                    obj_lit = None
                    if obj_lit_en:
                        obj_lit = obj_lit_en
                    elif obj_lit_datatype:
                        obj_lit = obj_lit_datatype

                    if sub.isnumeric():
                        sub_id = int(sub)
                        v_sub = ent_labels_wikipedia.get(sub_id, {}).get('title')

                        # short description for subject entity
                        sub_short_desc = ent_labels_wikipedia.get(sub_id, {}).get('short_desc')
                        if sub_short_desc and sub_id not in added_shortdesc:
                            added_shortdesc.add(sub_id)
                            outfd.write(f'<{v_sub}> <short description> <{sub_short_desc}> .\n'.encode('utf-8'))

                        sub_wikidata = ent_labels_wikidata.get(sub_id, ['',set(),'']) #label, altlabels, description
                        sub_description = sub_wikidata[2]

                        if not v_sub:
                            # not in wikipedia --> use wikidata label (description)
                            v_sub = sub_wikidata[0]
                            if v_sub and sub_description:
                                vsub = '{} ({})'.format(v_sub, sub_description)

                        if v_sub:
                            # add description
                            if sub_description and sub_id not in added_desc:
                                added_desc.add(sub_id)
                                outfd.write(f'<{v_sub}> <description> <{sub_description}> .\n'.encode('utf-8'))

                            v_prop = prop_labels.get('P'+prop, {}).get('label')
                            if v_sub and v_prop:
                                obj_description = None
                                if obj_ent and obj_ent.isnumeric():
                                    # obj is entity
                                    obj_id = int(obj_ent)
                                    v_obj = ent_labels_wikipedia.get(obj_id, {}).get('title')

                                    # short description for object entity
                                    obj_short_desc = ent_labels_wikipedia.get(obj_id, {}).get('short_desc')
                                    if obj_short_desc and obj_id not in added_shortdesc:
                                        added_shortdesc.add(obj_id)
                                        outfd.write(f'<{v_obj}> <short description> <{obj_short_desc}> .\n'.encode('utf-8'))

                                    obj_wikidata = ent_labels_wikidata.get(obj_id, ['',set(),''])
                                    obj_description = obj_wikidata[2]

                                    if not v_obj:
                                        # not in wikipedia --> use wikidata label (description)
                                        v_obj = obj_wikidata[0]
                                        if v_obj and obj_description:
                                            v_obj = '{} ({})'.format(v_obj, obj_description)

                                if obj_lit:
                                    # obj is literal
                                    v_obj = obj_lit

                                if v_obj:
                                    # verify v_obj is not None or ''
                                    if obj_description and obj_id not in added_desc:
                                        # add description
                                        added_desc.add(obj_id)
                                        outfd.write(f'<{v_obj}> <description> <{obj_description}> .\n'.encode('utf-8'))

                                    outfd.write(f'<{v_sub}> <{v_prop}> <{v_obj}> .\n'.encode('utf-8'))

                    # else:
                    #     # cannot find v_sub or v_prop --> skip
                    #     pass
                if count % 1000000 == 0:
                    pbar.n = count
                    pbar.refresh()


