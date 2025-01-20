Download from dumps.wikimedia.org:
- enwiki-20241220-page_props.sql.gz
- enwiki-20241220-page.sql.gz

Ingest in mariadb:
```
sudo ./import_script.sh enwiki-20241220-page_props.sql.gz enwiki-20241220-page.sql.gz
```

Create indexes and new table:
```
sudo ./create_idx_table.sh
```

Extract the mapping (wikidata_qid --> wikipedia_title) as a python dictionary in pickle:
```
# create virtualenv: optional
python -m venv venv
source venv/bin/activate
# install requirements
pip install requirements.txt
# run
python get_pickle_mapping.py
```
