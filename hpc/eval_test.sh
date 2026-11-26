#!/bin/bash
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --job-name=refactx_test
#SBATCH -N 1
#SBATCH --error=logs/eval_test_%j.err
#SBATCH --time=30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX

echo "Starting Mintaka test eval (2B model, 10 samples) at $(date)"
python -m utils.eval --config configs/config_mintaka_qwen35_2b_test.json
echo "Done at $(date)"
