#!/bin/bash
#SBATCH --output=/home/h7/ripo631h/ReFactX/logs/refactx_populate_fb_%j.out
#SBATCH --error=/home/h7/ripo631h/ReFactX/logs/refactx_populate_fb_%j.err
#SBATCH --job-name=refactx_pg
#SBATCH -N 1
#SBATCH --time=36:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source /home/h7/ripo631h/ReFactX/hpc/env.sh

# Exclusive writer: PostgreSQL runs directly on the shared PGDATA under
# a lock; do not run readers while this is active. See hpc/postgres_overlay.sh.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start rw
pg_overlay_install_traps

cd /home/ripo631h/ReFactX

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
