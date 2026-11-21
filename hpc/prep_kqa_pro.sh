#!/bin/bash
#SBATCH --job-name=kqa_pro_prep
#SBATCH --output=logs/kqa_pro_prep_%j.out
#SBATCH --error=logs/kqa_pro_prep_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

cd "$SLURM_SUBMIT_DIR"

KQAPRO_DIR="${KQAPRO_DIR:-$WS_PATH/kqa_pro}"
KQAPRO_INDEX="${KQAPRO_INDEX:-$KQAPRO_DIR/kqapro_qwen38.pkl}"
KQAPRO_TOKENIZER="${KQAPRO_TOKENIZER:-Qwen/Qwen3.8-27B}"

mkdir -p "$KQAPRO_DIR" "$SLURM_SUBMIT_DIR/logs"

echo "KQA Pro dir: $KQAPRO_DIR"
echo "Index:       $KQAPRO_INDEX"
echo "Tokenizer:   $KQAPRO_TOKENIZER"

# KQA Pro has no official live homepage; the HuggingFace mirror is the stable
# source.  Skip files that already exist so the job is idempotent.
for file in kb.json train.json val.json test.json README.md; do
    if [ ! -f "$KQAPRO_DIR/$file" ]; then
        echo "Downloading $file"
        curl -L --retry 3 --fail \
            -o "$KQAPRO_DIR/$file" \
            "https://huggingface.co/datasets/drt/kqa_pro/resolve/main/$file?download=true"
    fi
done

if [ ! -f "$KQAPRO_DIR/kqa_pro_triples.txt.gz" ]; then
    python -m utils.prep_kqa_pro \
        --data-dir "$KQAPRO_DIR" \
        --out-dir "$KQAPRO_DIR"
fi

if [ ! -f "$KQAPRO_INDEX" ]; then
    python -m utils.build_kqa_pro_index \
        "$KQAPRO_DIR/kqa_pro_triples.txt.gz" \
        --backend local \
        --output "$KQAPRO_INDEX" \
        --model-name "$KQAPRO_TOKENIZER"
fi

echo "KQA Pro prep done at $(date)"
teleclinotify "kqa_pro_prep done | job=$SLURM_JOB_ID dir=$KQAPRO_DIR"
