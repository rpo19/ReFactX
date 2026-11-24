#!/bin/bash
#SBATCH --job-name=build_kqa_sft
#SBATCH --output=logs/build_kqa_sft_%j.out
#SBATCH --error=logs/build_kqa_sft_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd "$SLURM_SUBMIT_DIR"

KQA_LOG="${KQA_LOG:-logs/kqa_pro_train_eval_qwen38_27b.out}"
BASE_MIX="${BASE_MIX:-logs/sft_training_mix_v3.jsonl}"
KQA_SFT="${KQA_SFT:-logs/kqa_pro_sft.jsonl}"
OUT_MIX="${OUT_MIX:-logs/sft_training_mix_kqa.jsonl}"

if [ ! -f "$KQA_LOG" ] && [ -f "${KQA_LOG}.0" ]; then
    KQA_LOG="${KQA_LOG}.0"
fi
if [ ! -f "$KQA_LOG" ]; then
    echo "Missing teacher log: $KQA_LOG" >&2
    echo "Run hpc/eval_kqa_pro_train_qwen38_27b.sh first (or set KQA_LOG)." >&2
    exit 1
fi
if [ ! -f "$BASE_MIX" ]; then
    echo "Missing base mixture: $BASE_MIX" >&2
    exit 1
fi

python -m utils.analyze_kqa_types \
    --log "$KQA_LOG" \
    --output logs/kqa_pro_type_report.md

python -m utils.build_kqa_sft \
    --log "$KQA_LOG" \
    --output "$KQA_SFT" \
    --prompt prompts/prompt_kqa_pro_teacher.yaml \
    --max-tokens 1280 \
    --mask-mode none

python -m utils.mix_kqa_sft \
    --base "$BASE_MIX" \
    --kqa "$KQA_SFT" \
    --output "$OUT_MIX"

teleclinotify "build_kqa_sft done | job=$SLURM_JOB_ID out=$OUT_MIX"
