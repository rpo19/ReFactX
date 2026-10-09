#!/bin/bash
#SBATCH --output=/home/h7/ripo631h/ReFactX/logs/refactx_populate_fb_%j.out
#SBATCH --error=/home/h7/ripo631h/ReFactX/logs/refactx_populate_fb_%j.err
#SBATCH --job-name=refactx_pg
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source /home/h7/ripo631h/ReFactX/hpc/env.sh
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source /home/h7/ripo631h/ReFactX/hpc/postgres_utils.sh

ensure_postgres
start_postgres_watchdog
trap stop_postgres_watchdog EXIT INT TERM

cd /home/ripo631h/ReFactX

if [ -f "$SHARED_POSTGRES" ]; then
    source "$SHARED_POSTGRES"
    export INDEX_PATH="postgres://postgres:${PGPASSWORD:-postgres}@${PG_IP:-127.0.0.1}:${PG_PORT:-5432}/postgres"
    export POSTGRES_CONNECTION="$INDEX_PATH"
    export BASE_INDEX_PATH="$INDEX_PATH"
fi

echo "Running populate_postgres (POSTGRES_CONNECTION=$POSTGRES_CONNECTION)..."

python -m utils.populate_postgres \
  $WS_PATH/ReFactX_freebase_facts.bz2 \
  --model-name Qwen/Qwen3.6-27B \
  --prefix " " \
  --end-of-triple ' .' \
  --tokenizer-batch-size 10000 \
  --table-name qwen36fbwikiprops \
  --rootkey -100 \
  --batch-size 5000000 \
  --switch-parameter 7 \
  --total-number-of-triples 921000000 \
  --count-leaves && teleclinotify success || teleclinotify fail
