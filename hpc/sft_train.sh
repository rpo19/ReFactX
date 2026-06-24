#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH -N 1
#SBATCH --output=logs/sft_train_%j.out
#SBATCH --error=logs/sft_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

module load CUDA/13.0.0

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

cd /home/ripo631h/ReFactX

echo "Starting SFT training at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"

pip install -q flash-linear-attention causal-conv1d

python -m utils.sft_train --config configs/sft_config.json

echo "SFT training finished at $(date)"
