download from dumps.wikimedia.org:
- page props.sql
- page.sql

ingest in mariadb with tthe script

create new table with this query: select page_id,page_title, pp_value as wikidata_qid from page, page_props where page_id=pp_page and pp_propname = 'wikibase_item' limit 10;

use mappint qid --> wikipedia_title
