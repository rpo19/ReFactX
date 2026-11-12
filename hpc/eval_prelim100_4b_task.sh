#!/bin/bash
#SBATCH --output=logs/eval_prelim100_4b_%j.out
#SBATCH --error=logs/eval_prelim100_4b_%j.err
#SBATCH --job-name=prelim100_4b
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# One preliminary 100-example 4B eval. The config is passed as $1 by
# hpc/submit_eval_prelim100_4b.sh; Slurm schedules each submitted task
# independently on any free GPU.

set -euo pipefail

CFG="${1:-}"
if [ -z "$CFG" ] || [ ! -f "$CFG" ]; then
    echo "Usage: sbatch $0 <config.json>" >&2
    exit 1
fi

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres
start_postgres_watchdog
trap stop_postgres_watchdog EXIT INT TERM

cd "$SLURM_SUBMIT_DIR"

if [ -f "$SHARED_POSTGRES" ]; then
    source "$SHARED_POSTGRES"
    export INDEX_PATH="postgres://postgres:${PGPASSWORD:-postgres}@${PG_IP:-127.0.0.1}:${PG_PORT:-5432}/postgres"
    export POSTGRES_CONNECTION="$INDEX_PATH"
    export BASE_INDEX_PATH="$INDEX_PATH"
fi

echo "Starting $CFG at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"

EXP=$(basename "$CFG" .json)
if python -m utils.eval --config "$CFG"; then
    teleclinotify "finish experiment $EXP"
else
    status=$?
    teleclinotify "experiment $EXP failed"
    exit "$status"
fi

echo "Finished $CFG at $(date)"
