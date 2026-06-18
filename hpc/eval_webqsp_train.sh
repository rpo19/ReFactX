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
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX
python -m utils.eval --config configs/config_webqsp_qwen36_35b3_train.json
