#!/bin/bash
#SBATCH --output=logs/eval_webqsp_grpo_checkpoint720_%j.out
#SBATCH --error=logs/eval_webqsp_grpo_checkpoint720_%j.err
#SBATCH --job-name=eval_webqsp_grpo720
#SBATCH -N 1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA="$WS_PATH/pgdata"
export SHARED_POSTGRES="$WS_PATH/postgres.addr"
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

python -m utils.eval --config configs/config_webqsp_qwen35_4b_grpo_checkpoint720_test.json

teleclinotify "WebQSP GRPO checkpoint-720 eval done | SLURM_JOB_ID=$SLURM_JOB_ID"
