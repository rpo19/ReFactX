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
import click
import pdb

wikidata_regex = re.compile(r'<http:\/\/www\.wikidata\.org\/.*\/([QP][0-9]+)>\s+'
r'<(\S+)>\s+'
r'"([^"]+)"@en\s+\.')
@click.command()
@click.argument('labels_path', type=click.Path(exists=True))
@click.argument('outfile_ents', type=click.Path())
@click.argument('outfile_props', type=click.Path(), required=False)
@click.option('--freebase', is_flag=True, help='Parse using Freebase regex')
def main(labels_path, outfile_ents, outfile_props, freebase):
    if not freebase and outfile_props is None:
        raise ValueError('Please provide outfile_props for Wikidata')
    ents = {}
    props = {}
    start = time.time()
    try:
        if freebase:
            with open(labels_path, 'r', encoding='utf-8') as fd:
                with tqdm() as pbar:
                    for count, line in enumerate(fd):
                        sub, pred, obj = line.strip().split('\t')
                        res_id = sub
                        if res_id not in ents:
                            ents[res_id] = ['',set(),''] # label, altLabels, description
                        if pred == 'type.object.name':
                            # first is main label
                            # next are alternative labels
                            if ents[res_id][0] == '':
                                ents[res_id][0] = obj
                            else:
                                ents[res_id][1].add(obj)
                        elif pred == 'common.topic.description':
                            # for descriptions take the shortest
                            if ents[res_id][2] == '':
                                ents[res_id][2] = obj
                            else:
                                if len(obj) < len(ents[res_id][2]):
                                    ents[res_id][2] = obj
        else:
            with gzip.open(labels_path, 'r') as fd:
                with tqdm() as pbar:
                    for count, bline in enumerate(fd):
                        line = bline.decode('unicode_escape') # correctly load unicode characters
                        match = wikidata_regex.match(line)
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
                                assert mydict[res_id][2] == ''
                                mydict[res_id][2] = obj

                    if count % 10000 == 0:
                        pbar.n = count
                        pbar.refresh()
    except EOFError:
        print('EOF error')

    elapsed = time.time() - start
    print('elapsed', elapsed)

    with open(outfile_ents, 'wb') as fd:
        pickle.dump(ents, fd)

    if not freebase:
        # freebase props are already verbalized
        with open(outfile_props, 'wb') as fd:
            pickle.dump(props, fd)

if __name__ == '__main__':
    main()

