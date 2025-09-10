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

import json
import gzip
import pickle
import sys

infile = sys.argv[1]
outfile = sys.argv[2]

# downloaded from https://hay.toolforge.org/propbrowse/
# https://www.wikidata.org/wiki/Wikidata:List_of_properties
with open(infile, 'r') as fd:
    props = json.load(fd)

non_id_props = [p for p in props if 'ID' not in p['label']]

datatypes = set([p['datatype'] for p in props])

datatypes_to_keep = set([
    #'commonsMedia',
    #'entity-schema', # could be useful for future works in entity linking or slot filling
    #'external-id',
    #'geo-shape',
    'globe-coordinate',
    #'math',
    #'monolingualtext',
    #'musical-notation',
    'quantity',
    'string',
    #'tabular-data', # maybe in the future but requires formatting the tabular data for the LLM
    'time',
    #'url',
    #'wikibase-form', # lexemes
    'wikibase-item',
    #'wikibase-lexeme',
    #'wikibase-property',
    #'wikibase-sense' # lexemes. could be useful in future work but need to verbalize lexemes too
])

filtered_props = [p for p in props if p['datatype'] in datatypes_to_keep]

types = set([t for p in props for t in p['types']])

types_to_keep = set([
 'Wikidata name property',
 'Wikidata qualifier',
 #'Wikidata sandbox property',
 #'about Wikimedia categories',
 'asymmetric property',
 #'for Commons',
 'for a taxon',
 'for astronomical objects',
 'for items about languages',
 'for items about organizations',
 'for items about people',
 'for items about works',
 'for language',
 'for physical quantities',
 'for places',
 #'for property documentation',
 #'for romanization system',
 'for software',
 #'metaproperty',
 #'multi-source external identifier',
 #'obsolete Wikidata property',
 'orderable Wikidata property',
 'related to chemistry',
 'related to economics',
 'related to medicine',
 'related to sport',
 #'representing a unique identifier',
 'symmetric property',
 #'to indicate a constraint',
 'to indicate a location',
 'to indicate a source',
 'transitive property',
 'with datatype string that is not an external identifier'
    # we also keep properties with no types
])

types_to_remove = types - types_to_keep

filtered_props = [p for p in filtered_props if len(set(p['types']).intersection(types_to_remove)) == 0]

# filtered_props_ids = [p['id'] for p in filtered_props]

filtered_props_dict = {p['id']:p for p in filtered_props}

with open(outfile, 'wb') as fd:
    pickle.dump(filtered_props_dict, fd)


