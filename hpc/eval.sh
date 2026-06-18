#!/bin/bash
#SBATCH --output=logs/refactx_eval_%j.out
#SBATCH --job-name=refactx_eval
#SBATCH -N 1
#SBATCH --error=logs/refactx_eval_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# -----------------------
# WAIT FOR POSTGRES ADDR FILE
# -----------------------
ADDR_FILE=$WS_PATH/postgres.addr

echo "Waiting for Postgres address file ($ADDR_FILE)..."
for i in $(seq 1 60); do
  if [ -f "$ADDR_FILE" ]; then
    source "$ADDR_FILE"
    echo "Postgres is running at $PG_HOST:$PG_PORT (SLURM_JOB_ID=$PG_SLURM_JOB_ID)"
    break
  fi
  sleep 10
done

if [ -z "${PG_HOST:-}" ]; then
  echo "ERROR: Postgres address file not found after 10 minutes."
  exit 1
fi

# -----------------------
# RUN YOUR PIPELINE
# -----------------------
echo "Running evaluation..."

cd /home/ripo631h/ReFactX
python -m utils.eval --config configs/config_2wiki_qwen36_35b3.json
