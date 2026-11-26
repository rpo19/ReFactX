#!/bin/bash
#SBATCH --job-name=merge_lora
#SBATCH -N 1
#SBATCH --output=logs/merge_lora_%j.out
#SBATCH --error=logs/merge_lora_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd /home/ripo631h/ReFactX

echo "Starting merge at $(date)"
python -m utils.merge_lora
echo "Merge finished at $(date)"
