#!/bin/bash
#SBATCH --output=logs/eval_all_%j.out
#SBATCH --job-name=refactx_all
#SBATCH -N 1
#SBATCH --error=logs/eval_all_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:8
#SBATCH --exclusive

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Point to the populated (old) PGDATA
export PGDATA=$WS_PATH/pgdata

source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX

echo "Starting 4 evals at $(date)"

CUDA_VISIBLE_DEVICES=0,1 python -m utils.eval --config configs/config_mintaka_qwen36_35b3_train.json \
  &> logs/eval_mintaka_train_${SLURM_JOB_ID}.log &
PID1=$!

CUDA_VISIBLE_DEVICES=2,3 python -m utils.eval --config configs/config_mintaka_qwen36_35b3_test.json \
  &> logs/eval_mintaka_test_${SLURM_JOB_ID}.log &
PID2=$!

CUDA_VISIBLE_DEVICES=4,5 python -m utils.eval --config configs/config_webqsp_qwen36_35b3_train.json \
  &> logs/eval_webqsp_train_${SLURM_JOB_ID}.log &
PID3=$!

CUDA_VISIBLE_DEVICES=6,7 python -m utils.eval --config configs/config_webqsp_qwen36_35b3_test.json \
  &> logs/eval_webqsp_test_${SLURM_JOB_ID}.log &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "All evals completed at $(date)"
