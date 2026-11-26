#!/bin/bash
#SBATCH --output=logs/refactx_populate_wd_%j.out
#SBATCH --job-name=refactx_pg
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
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

# -----------------------
# 3. START POSTGRES
# -----------------------

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


# ---- cleanup function ----
cleanup() {
  echo "Stopping Postgres (PID $PG_PID)..."
  kill "$PG_PID" 2>/dev/null || true
  wait "$PG_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# -----------------------
# 4. RUN YOUR PIPELINE
# -----------------------
echo "Running populate_postgres..."

cd /home/ripo631h/ReFactX
python -m utils.populate_postgres \
  $WS_PATH/ReFactX_wikidata_facts.bz2 \
  --model-name Qwen/Qwen3.6-27B \
  --prefix " " \
  --end-of-triple ' .' \
  --tokenizer-batch-size 10000 \
  --table-name qwen36 \
  --rootkey -100 \
  --batch-size 5000000 \
  --switch-parameter 7 \
  --total-number-of-triples 921000000 \
  --count-leaves && teleclinotify success || teleclinotify fail

