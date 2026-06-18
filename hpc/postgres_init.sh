#!/bin/bash
#SBATCH --output=logs/refactx_postgres_init_%j.out
#SBATCH --job-name=refactx_pg
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

exec 2>&1

set -euo pipefail

source env.sh

# -----------------------
# CONFIG
# -----------------------
IMG=$WS_PATH/postgres.sif
WORKDIR=$PWD
PGDATA=$WS_PATH/pgdata
PGSOCK=$WS_PATH/pgsocket
PORT=5432
DB=postgres

#echo "Removing old postgres"
#rm $IMG
#rm -rf $PGDATA $PGSOCK

echo "Init Postgres DB..."

singularity pull $WS_PATH/postgres.sif docker://postgres:18

# Initialize database only once
if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing Postgres DB..."

    mkdir -p $PGDATA
    mkdir -p $PGSOCK

    singularity exec \
        -B "$PGDATA:/var/lib/postgresql/data" \
        "$IMG" \
        initdb -D /var/lib/postgresql/data --username=postgres


    #echo "listen_addresses='*'" >> "$PGDATA/postgresql.conf"

    start_postgres() {
        echo "Starting temporary server..."

        singularity exec \
          -B "$PGDATA:/var/lib/postgresql/data" \
          -B "$PGSOCK:/pgsocket" \
          "$IMG" \
          postgres \
            -D /var/lib/postgresql/data \
            -p "$PORT" \
            -k /pgsocket \
          > "$PGDATA/logfile" 2>&1 &
        
        PG_PID=$!
        echo "Postgres PID: $PG_PID"
    }

    # ---- cleanup function ----
    cleanup() {

      echo "Stopping Postgres (PID $PG_PID)..."
      kill "$PG_PID" 2>/dev/null || true
      wait "$PG_PID" 2>/dev/null || true

      if [ "${success:-0}" -ne 0 ]; then
          echo "Program failed with exit code $success"
          teleclinotify "postgres init fail with $success"
      elif [ -n "${success+x}" ]; then
          echo "Program succeeded"
          teleclinotify "postgres init success!"
      fi

    }

    # to set the password we start postgres with no password required
    echo "local   all             all                                     trust" > "$PGDATA/pg_hba.conf"
    start_postgres

    trap cleanup EXIT INT TERM

    sleep 5

    if ! kill -0 "$PG_PID" 2>/dev/null; then
        echo "Postgres died immediately"
        exit 1
    fi

    echo "Setting postgres password..."

    PGPASSWORD="${PGPASSWORD:-$(openssl rand -base64 32)}"
    echo "Using PGPASSWORD: $PGPASSWORD"

    singularity exec \
        -B "$PGDATA:/var/lib/postgresql/data" \
        -B "$PGSOCK:/pgsocket" \
        "$IMG" \
        psql \
            -h /pgsocket \
            -p "$PORT" \
            -U postgres \
            -d postgres \
            -c "ALTER USER postgres WITH PASSWORD '$PGPASSWORD';"

    cleanup

    # gzip+base64 encoded pg_hba that simply allows only local connection (but requiring password)
    echo "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" | base64 -d | zcat > "$PGDATA/pg_hba.conf"

    success=$?
    
   # start_postgres

   # echo "Final test select version"
   # singularity exec \
   #     -B "$PGDATA:/var/lib/postgresql/data" \
   #     -B "$PGSOCK:/pgsocket" \
   #     "$IMG" \
   #     psql \
   #         -h /pgsocket \
   #         -p "$PORT" \
   #         -U postgres \
   #         -d postgres \
   #         -c "SELECT version();"


fi

teleclinotify "postgres init skipped (already initialized)"

