#!/bin/bash
#SBATCH --output=logs/eval_webqsp_train_%j.out
#SBATCH --job-name=refactx_webqsp_train
#SBATCH -N 1
#SBATCH --error=logs/eval_webqsp_train_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

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
echo "Running evaluation..."

cd /home/ripo631h/ReFactX
python -m utils.eval --config configs/config_webqsp_qwen36_35b3_train.json
