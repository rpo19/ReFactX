#!/bin/bash
#SBATCH --output=logs/eval_mintaka_test_%j.out
#SBATCH --job-name=refactx_mintaka_test
#SBATCH -N 1
#SBATCH --error=logs/eval_mintaka_test_%j.err
#SBATCH --time=73:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX

if [ -f "$SHARED_POSTGRES" ]; then
    source "$SHARED_POSTGRES"
    export INDEX_PATH="postgres://postgres:${PGPASSWORD:-postgres}@${PG_IP:-127.0.0.1}:${PG_PORT:-5432}/postgres"
    export POSTGRES_CONNECTION="$INDEX_PATH"
fi

python -m utils.eval --config configs/config_mintaka_qwen36_35b3_test.json

teleclinotify "refactx_mintaka_test done | SLURM_JOB_ID=$SLURM_JOB_ID"
