#!/bin/bash
#SBATCH --job-name=sft_kqa_08b
#SBATCH --output=logs/sft_kqa_08b_%j.out
#SBATCH --error=logs/sft_kqa_08b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd "$SLURM_SUBMIT_DIR"

DATA="${DATA:-logs/sft_training_mix_kqa.jsonl}"
if [ ! -f "$DATA" ]; then
    echo "Missing KQA SFT mixture: $DATA" >&2
    echo "Run hpc/build_kqa_sft.sh first (or set DATA)." >&2
    exit 1
fi

accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
    --config configs/sft_config_qwen35_08b_kqa.json \
    --data "$DATA" \
    --output-dir "${OUTPUT_DIR:-$WS_PATH/sft_output_08b_kqa}"

teleclinotify "sft_kqa_08b done | job=$SLURM_JOB_ID"
