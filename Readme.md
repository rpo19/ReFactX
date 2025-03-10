## Download dumps
- Go to https://dumps.wikimedia.org/wikidatawiki/entities/ and download `latest-truthy.nt.bz2`.
- Go to https://dumps.wikimedia.org/enwiki/20241220/ and download `enwiki-20241220-page.sql.gz` and `enwiki-20241220-page_props.sql.gz`

## Filter Labels from the wikidata dump
The output file only has triples with label, altLabel, and description as predicate.
```
bzgrep -P '(http://www\\.w3\\.org/2000/01/rdf-schema#label|http://www\\.w3\\.org/2004/02/skos/core#altLabel|http://schema\\.org/description).*\\@en\s+.' latest-truthy.nt.bz2 | gzip -c > latest-truthy-labels-descriptions.nt.gz
```

## Create a virtualenv
```
python -m venv venv
pip install -r requirements/base.txt
pip install -r requirements/postgres.txt # for the postgres index
```

## Load labels and description into a pickle file
```
python load_labels.py latest-truthy-labels-descriptions.nt.gz wikidata-labels.pickle
```

## Create Label mappings for Wikipedia Entities

Refer to the readme in services/mariadb: [Readme.md](services/mariadb/Readme.md)

## Download property labels
Go to https://hay.toolforge.org/propbrowse/ and Download all properties as JSON.

## Filter the properties
Replace `input_props.json` with the file you just downloaded and run:
```
python filter_props.py input_props.json filtered_props.pickle
```

## Verbalize the triples using the labels
```
python verbalize_triples.py --props-mapping props_mapping --wikidata-labels wikidata_labels --wikipedia-entity-mapping wikipedia_entity_mapping wikidump.bz2 verbalized_triples.bz2 [--total-number-of-triples number]
```

## Tokenize
Choose a model from huggingface for using its tokenizer, then run:
```
python tokenize_triples.py --model-name model_name --prefix " " --end-of-triple "." --batchsize 10000 verbalized_triples.bz2 tokenized_triples.pickle.bz2 [--total-number-of-triples number]
```

## Start postgres
```
cd services/postgres
sudo docker compose up -d
```

## Choose parameters for postgres population
- Choose an integer that will represent the root of the tree. It must not be a token_id in the tokenizer vocabulary. I used 150000.
- Choose the switch parameter N. After N hops in the tree you are going to load the entire subtree in memory. You can estimate the in-memory size with `estimate_subtree_mem_usage.py`. I used 6.

## Populate postgres
```
python populate_postgres.py tokenized_triples.pickle.bz2 postgres://postgres:secret@host:port/postgres tablename rootkey batchsize switch_parameter [num_triples]
```
