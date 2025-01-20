#!/bin/bash
(echo "SET GLOBAL foreign_key_checks = 0; SET GLOBAL unique_checks = 0; SET GLOBAL autocommit = 0; SET GLOBAL innodb_flush_log_at_trx_commit = 2;" && \
	zcat $@ && echo "SET GLOBAL foreign_key_checks = 1; SET GLOBAL unique_checks = 1; SET GLOBAL autocommit = 1; SET GLOBAL innodb_flush_log_at_trx_commit = 1;") \
	| docker compose exec -T mariadb mariadb -u root -pexample mysql

