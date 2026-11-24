#!/bin/bash
#SBATCH --job-name=refactx_kqa_pro_train_qwen38
#SBATCH --output=logs/kqa_pro_train_qwen38_27b_%j.out
#SBATCH --error=logs/kqa_pro_train_qwen38_27b_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd "$SLURM_SUBMIT_DIR"

KQAPRO_DIR="${KQAPRO_DIR:-$WS_PATH/kqa_pro}"
KQAPRO_INDEX="${KQAPRO_INDEX:-$KQAPRO_DIR/kqapro_qwen38.pkl}"
KQAPRO_SAMPLE_TOTAL="${KQAPRO_SAMPLE_TOTAL:-12000}"
export KQAPRO_DIR KQAPRO_INDEX

if [ ! -f "$KQAPRO_DIR/train.jsonl" ] || [ ! -f "$KQAPRO_INDEX" ]; then
    echo "Missing KQA Pro data or index under $KQAPRO_DIR" >&2
    echo "Submit hpc/prep_kqa_pro.sh first, or set KQAPRO_DIR/KQAPRO_INDEX." >&2
    exit 1
fi

if [ ! -f "$KQAPRO_DIR/train_sample_ids.json" ]; then
    python -m utils.sample_kqa_pro \
        --data-dir "$KQAPRO_DIR" \
        --output "$KQAPRO_DIR/train_sample_ids.json" \
        --total "$KQAPRO_SAMPLE_TOTAL"
fi

echo "Teacher generation started at $(date)"
python -m utils.eval --config configs/config_kqa_pro_train_qwen38_27b_local.json

teleclinotify "kqa_pro_train_qwen38_27b done | job=$SLURM_JOB_ID"
