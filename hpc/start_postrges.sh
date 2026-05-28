#!/bin/bash

if [ -f env.sh ]
then
	source env.sh
fi

if [ -f hpc/env.sh ]
then
	source hpc/env.sh
fi

# -----------------------
# CONFIG
# -----------------------
IMG=$WS_PATH/postgres.sif
WORKDIR=$PWD
PGDATA=$WS_PATH/pgdata
PGSOCK=$WS_PATH/pgsocket
PORT=5432
DB=postgres

# -----------------------
# 3. START POSTGRES
# -----------------------

exec singularity exec \
  -B "$PGDATA:/var/lib/postgresql/data" \
  -B "$PGSOCK:/pgsocket" \
  "$IMG" \
  postgres \
    -D /var/lib/postgresql/data \
    -p "$PORT" \
    -k /pgsocket \
  > "$PGDATA/logfile" 2>&1 &

