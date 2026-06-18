#!/bin/bash
#SBATCH --output=logs/refactx_eval_%j.out
#SBATCH --job-name=refactx_eval
#SBATCH -N 1
#SBATCH --error=logs/refactx_eval_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX
python -m utils.eval --config configs/config_2wiki_qwen36_35b3.json
