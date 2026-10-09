#!/bin/bash
#SBATCH --output=logs/eval_webqsp_train_qwen38_27b_%j.out
#SBATCH --job-name=refactx_webqsp_train_qwen38
#SBATCH -N 1
#SBATCH --error=logs/eval_webqsp_train_qwen38_27b_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

python -m utils.eval --config configs/config_webqsp_qwen38_27b_train.json

teleclinotify "eval_webqsp_train_qwen38_27b done | SLURM_JOB_ID=$SLURM_JOB_ID"
