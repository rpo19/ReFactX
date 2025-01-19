Download from dumps.wikimedia.org:
- enwiki-20241220-page_props.sql.gz
- enwiki-20241220-page.sql.gz

Ingest in mariadb:
```
sudo ./import_script.sh enwiki-20241220-page_props.sql.gz enwiki-20241220-page.sql.gz
```

Create indexes:
```
sudo ./create_indexes.sh
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


create new table with this query: select page_id,page_title, pp_value as wikidata_qid from page, page_props where page_id=pp_page and pp_propname = 'wikibase_item' limit 10;
CREATE TABLE wikidata_mapping AS
SELECT page_id, page_title, pp_value AS wikidata_qid
FROM page, page_props
WHERE page_id = pp_page AND pp_propname = 'wikibase_item'
LIMIT 10;


use mappint qid --> wikipedia_title
