#!/bin/bash
#SBATCH --job-name=sft_test
#SBATCH -N 1
#SBATCH --output=logs/sft_test_%j.out
#SBATCH --error=logs/sft_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

cd /home/ripo631h/ReFactX

echo "Starting SFT test at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"

python -m utils.sft_train --config configs/sft_config_test.json

echo "SFT test finished at $(date)"
