#!/bin/bash
#SBATCH --output=logs/test_mintaka_qwen38_27b_%j.out
#SBATCH --job-name=refactx_mintaka_qwen38
#SBATCH -N 1
#SBATCH --error=logs/test_mintaka_qwen38_27b_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

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
    export BASE_INDEX_PATH="$INDEX_PATH"
fi

python -m utils.eval --config configs/config_mintaka_qwen38_27b_train.json

teleclinotify "test_mintaka_qwen38_27b done | SLURM_JOB_ID=$SLURM_JOB_ID"
