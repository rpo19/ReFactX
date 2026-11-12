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

# Overridable so older adapters can still be merged explicitly. The defaults
# target the supported-mix SFT run; the legacy sft_merged path is not used.
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B}"
ADAPTER="${ADAPTER:-$WS_PATH/sft_output_supported_mix}"
OUTPUT="${OUTPUT:-$WS_PATH/sft_merged_supported_mix}"

echo "Starting merge at $(date)"
echo "  base:    $BASE_MODEL"
echo "  adapter: $ADAPTER"
echo "  output:  $OUTPUT"

python -m utils.merge_lora \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --output "$OUTPUT"

echo "Merge finished at $(date)"
