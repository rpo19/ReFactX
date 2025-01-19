#!/bin/bash
docker compose exec -T mariadb mariadb -u root -pexample mysql << EOF
CREATE INDEX idx_page_id ON page(page_id);
CREATE INDEX idx_pp_propname_page ON page_props(pp_propname, pp_page);
EOF

