#!/bin/bash
#SBATCH --output=logs/eval_mintaka_sft_%j.out
#SBATCH --error=logs/eval_mintaka_sft_%j.err
#SBATCH --job-name=sft_mintaka_test
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

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

# The config loads the adapter directly via `adapter_path`; no merge step.
echo "Starting SFT Mintaka test eval at $(date)"
python -m utils.eval --config configs/config_mintaka_sft_test.json
echo "SFT Mintaka test eval finished at $(date)"
