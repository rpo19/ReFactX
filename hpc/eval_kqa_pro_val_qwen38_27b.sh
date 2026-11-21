#!/bin/bash
#SBATCH --job-name=refactx_kqa_pro_val_qwen38
#SBATCH --output=logs/kqa_pro_val_qwen38_27b_%j.out
#SBATCH --error=logs/kqa_pro_val_qwen38_27b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd "$SLURM_SUBMIT_DIR"

KQAPRO_DIR="${KQAPRO_DIR:-$WS_PATH/kqa_pro}"
KQAPRO_INDEX="${KQAPRO_INDEX:-$KQAPRO_DIR/kqapro_qwen38.pkl}"
export KQAPRO_DIR KQAPRO_INDEX

if [ ! -f "$KQAPRO_DIR/val.jsonl" ] || [ ! -f "$KQAPRO_INDEX" ]; then
    echo "Missing KQA Pro val data or index under $KQAPRO_DIR" >&2
    echo "Submit hpc/prep_kqa_pro.sh first." >&2
    exit 1
fi

python -m utils.eval --config configs/config_kqa_pro_val_qwen38_27b_local.json
teleclinotify "kqa_pro_val_qwen38_27b done | job=$SLURM_JOB_ID"
