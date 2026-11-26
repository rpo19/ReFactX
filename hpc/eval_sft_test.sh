#!/bin/bash
#SBATCH --output=logs/eval_sft_test_%j.out
#SBATCH --error=logs/eval_sft_test_%j.err
#SBATCH --job-name=eval_sft_test
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
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

echo "Starting 2 evals at $(date)"

MERGE_DIR=/data/horse/ws/ripo631h-quokka/sft_merged
if [ ! -d "$MERGE_DIR" ]; then
    echo "Merged model not found, merging LoRA adapters..."
    python -m utils.merge_lora
fi

CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config configs/config_mintaka_sft_test.json \
  &> logs/eval_mintaka_test_${SLURM_JOB_ID}.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config configs/config_2wiki_sft_test.json \
  &> logs/eval_2wiki_test_${SLURM_JOB_ID}.log &
PID2=$!

wait $PID1 $PID2
echo "All evals completed at $(date)"
