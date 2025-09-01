# Wikidata Prefix Tree
This file contains instruction on how to prepare the prefix tree from Wikidata dumps.

For using the 800 million facts from the paper:
- download the facts from [HuggingFace](https://huggingface.co/datasets/rpozzi/ReFactX_data)
- continue from [Tokenize and Populate](#tokenize-and-populate)

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
pip install -r requirements.txt
```

## Install flash attention
If you have compatible GPUs.
```
pip install flash-attn --no-build-isolation
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

## Tokenize and Populate

### Start postgres
```
cd services/postgres
sudo docker compose up -d
```

### Choose parameters for postgres population
- A model from huggingface for using its tokenizer; the db is tokenizer specific, thus for using another model with a different tokenizer you should re-index from scratch.
- The rootkey: an integer that will represent the root of the tree. It must not be a token_id in the tokenizer vocabulary (e.g., 300000).
- The switch parameter N. After N hops in the tree you are going to load the entire subtree in memory. You can estimate the in-memory size with `estimate_subtree_mem_usage.py`. I used 6.
- A prefix to add before the verbalized triple: remind that LLMs tokens differ when preceded by a space, so use " " as prefix if you plan to generate the triple after some tokens. Use "" for generating the triple after a newline.
- End-of-triple is appended at the end of each triple; for instance, could be " ." to resemble SPARQL or also something like `</end-of-fact>`.
- Tokenizer-batch-size is for tokenization efficiency.
- Batch-size: use the highest value that your memory allows. Explanation: since with a big number of triples it is impossible to keep the entire tree in memory, a tree is constructed for each batch, and starting from tree the db is populated with all the sequences with the available next-tokens. Since the same sequence S can appear in different batches, at search time, when looking for the possible next-tokens of S, we could get up to number-of-batches results that we need to merge. So, the higher the batch size the better is the db "organization" and consequently the lookup is faster and less disk space is required (less duplicated prefix sequences).
- `--debug` you can use a debug option to ingest only the first batch and test it with `debug_postgres.py`.

### Populate postgres
```
python populate_postgres.py $VERBALIZED_TRIPLES_BZ2 \
    --model-name $MODEL_NAME \
    --prefix " " \
    --end-of-triple ' .' \
    --tokenizer-batch-size 10000 \
    --postgres-connection postgres://postgres:secret@host:port/postgres \
    --table-name $TABLE_NAME \
    --rootkey 500000 \
    --batch-size 5000000 \
    --switch-parameter 7 \
    --total-number-of-triples 889679260
```
