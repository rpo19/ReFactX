#!/bin/bash
docker compose exec -T mariadb mariadb -u root -pexample mysql << EOF
CREATE INDEX idx_page_id ON page(page_id, page_namespace);
CREATE INDEX idx_pp_propname_page ON page_props(pp_propname, pp_page);
CREATE TABLE mappings_plus AS
SELECT
    page.page_id,
    SUBSTRING(page_props.pp_value, 1, 1) AS wikidata_prefix,  -- Extracts the initial letter 'Q'
    CAST(SUBSTRING(page_props.pp_value, 2) AS INT) AS wikidata_number,  -- Casts the numeric part to integer
    page.page_title,
    page.page_is_redirect,
    page_props_shortdesc.pp_value AS wikibase_shortdesc  -- Retrieves the short description from the second join
FROM
    page
INNER JOIN
    page_props ON page.page_id = page_props.pp_page
INNER JOIN
    page_props AS page_props_shortdesc ON page.page_id = page_props_shortdesc.pp_page
WHERE
    page_props.pp_propname = 'wikibase_item'  -- Filters for Wikidata items
    AND page_props_shortdesc.pp_propname = 'wikibase-shortdesc'  -- Filters for the short description
    AND page.page_namespace = 0;  -- Only items (namespace 0)
EOF

