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
import nltk

wikidata_template = '<{v_sub}> <{v_prop}> <{v_obj}> .\n'
freebase_template = '{v_sub} {v_prop} {v_obj} .\n'

freebase_desc_max_len = 100 # max length of description in Freebase to be included in verbalization

wikidata_triple_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>\s+'
    r'<http:\/\/www\.wikidata\.org\/prop\/direct\/P([0-9]+)>\s+'
    r'(?:<http:\/\/www\.wikidata\.org\/entity\/Q([0-9]+)>|"([^"]+)"@en|"([^"]+)"\^\^<.+>)\s+\.')

def verbalize_property(prop_id: int, prop_labels):
    return prop_labels.get('P'+prop_id, {}).get('label')

def wikidata_verbalize_entity(entity_id: int, ent_labels_wikidata, ent_labels_wikipedia, output=None, added_shortdesc=None, added_desc=None):
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

def freebase_verbalize_entity(entity_id: str, ent_labels_freebase):
    label = ent_labels_freebase.get(entity_id, [None])[0]
    description = ent_labels_freebase.get(entity_id, [None, None, ''])[2]
    if description and len(description) > freebase_desc_max_len:
        description = nltk.sent_tokenize(description)[0] # take first sentence if too long
        if len(description) > freebase_desc_max_len:
            description = description[:freebase_desc_max_len]
    if label and description:
        return f'{label} ({description})'
    else:
        return label

@click.command()
@click.option("--wikidata-props-mapping", required=False, help="Path to the filtered properties pickle file.")
@click.option("--wikidata-labels", required=False, help="Path to the Wikidata labels pickle file.")
@click.option("--freebase-labels", required=False, help="Path to the Freebase labels pickle file.")
@click.option("--wikipedia-entity-mapping", required=False, help="Path to the Wikidata titles mapping pickle file.")
@click.argument("dump")
@click.argument("outfile")
@click.option("--total-number-of-triples", type=int, default=None, help="Total number of triples (optional).")
def main(wikidata_props_mapping, wikidata_labels, freebase_labels, wikipedia_entity_mapping, dump, outfile, total_number_of_triples):

    if freebase_labels:
        if wikidata_labels or wikidata_props_mapping or wikipedia_entity_mapping:
            raise ValueError('When using Freebase labels, do not provide Wikidata or Wikipedia files')
        mode = 'freebase'
        template = freebase_template

        # nltk sent_tokenizer test
        try:
            nltk.sent_tokenize("This is a sentence. This is another one.")
        except LookupError:
            print("Downloading punkt_tab for nltk")
            nltk.download('punkt_tab')

        with open(freebase_labels, 'rb') as fd:
            ent_labels_freebase = pickle.load(fd)
    else:
        # wikidata
        if not (wikidata_labels and wikidata_props_mapping and wikipedia_entity_mapping):
            raise ValueError('Please provide all three files for Wikidata: wikidata-props-mapping, wikidata-labels, wikipedia-entity-mapping')
        if freebase_labels:
            raise ValueError('When using Wikidata labels, do not provide Freebase labels file')
        
        mode = 'wikidata'
        template = wikidata_template

        with open(wikidata_props_mapping, 'rb') as fd:
            wikidata_prop_labels = pickle.load(fd)

        with open(wikidata_labels, 'rb') as fd:
            ent_labels_wikidata = pickle.load(fd)

        with open(wikipedia_entity_mapping, 'rb') as fd:
            ent_labels_wikipedia = pickle.load(fd)



    tqdm_params = {'total': total_number_of_triples}

    added_shortdesc = set()
    added_desc = set()

    # stats
    noent = 0
    nosublabel = 0
    nooblabel = 0
    #

    with tqdm(**tqdm_params) as pbar:
        with bz2.BZ2File(outfile, 'w') as outfd:
            if mode == 'wikidata':
                fd = bz2.BZ2File(dump, 'r')
            elif mode == 'freebase':
                fd = open(dump, 'r', encoding='utf-8')



            for count, bline in enumerate(fd):
                v_obj = None
                if mode == 'wikidata':
                    line = bline.decode('unicode_escape') # correctly load unicode characters
                    match = wikidata_triple_regex.match(line)
                    if match:
                        sub, prop, obj_ent, obj_lit_en, obj_lit_datatype = match.groups()
                        obj_lit = None
                        if obj_lit_en:
                            obj_lit = obj_lit_en
                        elif obj_lit_datatype:
                            obj_lit = obj_lit_datatype

                        if sub.isnumeric():
                            sub_id = int(sub)
                            v_sub = wikidata_verbalize_entity(
                                        entity_id = sub_id,
                                        ent_labels_wikidata = ent_labels_wikidata,
                                        ent_labels_wikipedia = ent_labels_wikipedia,
                                        output = outfd,
                                        added_shortdesc = added_shortdesc,
                                        added_desc = added_desc)

                            if v_sub:
                                v_prop = verbalize_property(prop, wikidata_prop_labels)

                                if v_sub and v_prop:
                                    if obj_ent and obj_ent.isnumeric():
                                        # obj is entity
                                        obj_id = int(obj_ent)
                                        v_obj = ent_labels_wikipedia.get(obj_id, {}).get('title')
                                        v_obj = wikidata_verbalize_entity(
                                                    entity_id = obj_id,
                                                    ent_labels_wikidata = ent_labels_wikidata,
                                                    ent_labels_wikipedia = ent_labels_wikipedia,
                                                    output = outfd,
                                                    added_shortdesc = added_shortdesc,
                                                    added_desc = added_desc)

                                    if obj_lit:
                                        # obj is literal
                                        v_obj = obj_lit
                elif mode == 'freebase':
                    line = bline
                    sub, v_prop, obj = line.split('\t')
                    if obj[-1] == '\n':
                        obj = obj[:-1]
                    if sub.startswith('m.'):
                        v_sub = freebase_verbalize_entity(sub, ent_labels_freebase)

                        if v_sub:
                            if obj.startswith('m.'):
                                v_obj = freebase_verbalize_entity(obj, ent_labels_freebase)
                            else:
                                v_obj = obj
                        else:
                            nosublabel += 1
                            # debug
                            print(sub)
                    else:
                        noent += 1

                if v_obj:
                    outfd.write(template.format(v_sub=v_sub, v_prop=v_prop, v_obj=v_obj).encode('utf-8'))
                else:
                    nooblabel += 1
                if count % 1000000 == 0:
                    pbar.n = count
                    pbar.refresh()

                    print(f'\rProcessed {count} lines. No entity: {noent}, no sub label: {nosublabel}, no obj label: {nooblabel}', end='')

if __name__ == "__main__":
    main()