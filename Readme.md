## Download dumps
- Go to https://dumps.wikimedia.org/wikidatawiki/entities/ and download `latest-truthy.nt.bz2`.
- Go to https://dumps.wikimedia.org/enwiki/20241220/ and download `enwiki-20241220-page.sql.gz` and `enwiki-20241220-page_props.sql.gz`

## Create Label mappings

Refer to the readme in services/mariadb: [Readme.md](services/mariadb/Readme.md)

## Download property labels
Go to https://hay.toolforge.org/propbrowse/ and Download all properties as JSON.

## Filter the properties
Use the notebook `filter_props.ipynb`

## Verbalize the triples using the labels
```
# TODO adapt to new label_mappings
# TODO solve unicode problem
# TODO keep description
# TODO keep wikibase-shortdesc ?
python verbalize_triples.py
```

## Tokenize
Choose a model from huggingface for using its tokenizer, then run:
```
python generate_ctrie_big.py model outfile
```

## Start postgres
```
cd services/postgres
sudo docker compose up -d
```

## Choose parameters for postgres population
- Choose an integer that will represent the root of the tree. It must not be a token_id in the tokenizer vocabulary. (The current implementations is for 2-bytes integers; max 65536). I used 60000.
- Choose the switch parameter N. After N hops in the tree you are going to load the entire subtree in memory. You can estimate the in-memory size with `estimate_subtree_mem_usage.py`. I used 8.

## Populate postgres
```
python populate_postgres.py /mnt/data/jupyterlab/rpozzi/ctrie_Phi-3-mini-128k-instruct_big.bz2 postgres://postgres:secret@10.0.0.118:5432/postgres 60000 5000000 8 709140000
```
