#!/bin/bash
#SBATCH --output=logs/refactx_populate_wd_%j.out
#SBATCH --job-name=refactx_pg
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Populate the Wikidata table into the shared seed. This is a WRITER: the
# helper takes an exclusive lock and runs Postgres directly on $WS_PATH/pgdata.
# Do not run readers while this job is alive. See hpc/postgres_overlay.sh.

exec 2>&1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"

pg_overlay_start rw
pg_overlay_install_traps

echo "Running populate_postgres (POSTGRES_CONNECTION=$POSTGRES_CONNECTION)..."

cd /home/ripo631h/ReFactX
python -m utils.populate_postgres \
  "$WS_PATH/ReFactX_wikidata_facts.bz2" \
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
