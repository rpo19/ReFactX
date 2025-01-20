## Download dump
Go to https://dumps.wikimedia.org/wikidatawiki/entities/ and download `latest-truthy.nt.bz2`.

## Filter labels from the dump
We filter for labels, altLabels, and descriptions

```
bzgrep -P '(http://www\\.w3\\.org/2000/01/rdf-schema#label|http://www\\.w3\\.org/2004/02/skos/core#altLabel|http://schema\\.org/description).*\\@en\s+.' latest-truthy.nt.bz2 | pv | gzip -c > /workspace/data/latest-truthy-labels-descriptions.nt.gz
```

## Load the labels into a pickle
```
python load_labels_truthy_fix.py
```

## Download property labels
Go to https://hay.toolforge.org/propbrowse/ and Download all properties as JSON.

## Filter the properties
Use the notebook `filter_props.ipynb`

## Verbalize the triples using the labels
```
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
