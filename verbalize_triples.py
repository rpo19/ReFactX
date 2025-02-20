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
import bz2
import re
import click

def verbalize_property(prop_id: int, prop_labels):
    return prop_labels.get('P'+prop_id, {}).get('label')

def verbalize_entity(entity_id: int, ent_labels_wikidata, ent_labels_wikipedia, output=None, added_shortdesc=None, added_desc=None):
    # -> unique_title, short descr, description
    unique_title = None # None when not in wikipedia and no description in wikidata

    wikipedia_title = ent_labels_wikipedia.get(entity_id, {}).get('title')
    wikipedia_short_desc = ent_labels_wikipedia.get(entity_id, {}).get('short_desc')
    wikidata_label, wikidata_alt_labels, wikidata_description = ent_labels_wikidata.get(entity_id, ['',set(),'']) #label, altlabels, description

    if wikipedia_title:
        unique_title = wikipedia_title
    elif wikidata_label and wikidata_description:
        unique_title = f'{wikidata_label} ({wikidata_description})'

    if output:
        # avoid duplicate descriptions (same entity is part of several triples)
        if wikipedia_short_desc and entity_id not in added_shortdesc:
            added_shortdesc.add(entity_id)
            output.write(f'<{unique_title}> <short description> <{wikipedia_short_desc}> .\n'.encode('utf-8'))
        if wikidata_description and entity_id not in added_desc:
            added_desc.add(entity_id)
            output.write(f'<{unique_title}> <description> <{wikidata_description}> .\n'.encode('utf-8'))

    return unique_title

@click.command()
@click.option("--props-mapping", required=True, help="Path to the filtered properties pickle file.")
@click.option("--wikidata-labels", required=True, help="Path to the Wikidata labels pickle file.")
@click.option("--wikipedia-entity-mapping", required=True, help="Path to the Wikidata titles mapping pickle file.")
@click.argument("wiki_dump")
@click.argument("outfile")
@click.option("--total-number-of-triples", type=int, default=None, help="Total number of triples (optional).")
def main(props_mapping, wikidata_labels, wikipedia_entity_mapping, wiki_dump, outfile, total_number_of_triples):

    with open(props_mapping, 'rb') as fd:
        prop_labels = pickle.load(fd)

    with open(wikidata_labels, 'rb') as fd:
        ent_labels_wikidata = pickle.load(fd)

    with open(wikipedia_entity_mapping, 'rb') as fd:
        ent_labels_wikipedia = pickle.load(fd)

    triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
    r'<http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)>\s+'
    r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"([^"]+)"@en|"([^"]+)"\^\^<.+>)\s+\.')

    tqdm_params = {'total': total_number_of_triples}

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
                            v_sub = verbalize_entity(
                                        entity_id = sub_id,
                                        ent_labels_wikidata = ent_labels_wikidata,
                                        ent_labels_wikipedia = ent_labels_wikipedia,
                                        output = outfd,
                                        added_shortdesc = added_shortdesc,
                                        added_desc = added_desc)

                            if v_sub:
                                v_prop = verbalize_property(prop, prop_labels)

                                if v_sub and v_prop:
                                    if obj_ent and obj_ent.isnumeric():
                                        # obj is entity
                                        obj_id = int(obj_ent)
                                        v_obj = ent_labels_wikipedia.get(obj_id, {}).get('title')
                                        v_obj = verbalize_entity(
                                                    entity_id = obj_id,
                                                    ent_labels_wikidata = ent_labels_wikidata,
                                                    ent_labels_wikipedia = ent_labels_wikipedia,
                                                    output = outfd,
                                                    added_shortdesc = added_shortdesc,
                                                    added_desc = added_desc)

                                    if obj_lit:
                                        # obj is literal
                                        v_obj = obj_lit

                                    if v_obj:
                                        outfd.write(f'<{v_sub}> <{v_prop}> <{v_obj}> .\n'.encode('utf-8'))

                        # else:
                        #     # cannot find v_sub or v_prop --> skip
                        #     pass
                    if count % 1000000 == 0:
                        pbar.n = count
                        pbar.refresh()

if __name__ == "__main__":
    main()