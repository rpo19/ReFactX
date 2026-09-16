#!/bin/bash
#SBATCH --job-name=sft_train_qwen35_4b
#SBATCH -N 1
#SBATCH --output=logs/sft_train_qwen35_4b_%j.out
#SBATCH --error=logs/sft_train_qwen35_4b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/ripo631h/ReFactX

echo "Starting SFT training at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"

python -m utils.sft_train --config sft_config_qwen35_4b_nomask.json --output-dir "$WS_PATH/sft_output_qwen35_4b_nomask"

echo "SFT training finished at $(date)"
