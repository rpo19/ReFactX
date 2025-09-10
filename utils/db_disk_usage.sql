SELECT                  
  c.relname AS table_name,
  pg_size_pretty(pg_relation_size(c.oid)) AS main_table_size,
  pg_size_pretty(pg_indexes_size(c.oid)) AS all_indexes_size,
  pg_size_pretty(pg_total_relation_size(c.oid)) AS total_size,
  pg_size_pretty(pg_relation_size(c.reltoastrelid)) AS toast_table_size
FROM pg_class c
WHERE c.relname = 'TABLENAME';