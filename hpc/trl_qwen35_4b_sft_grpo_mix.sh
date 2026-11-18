#!/bin/bash
#SBATCH --job-name=trl_qwen35_4b_sft_mix
#SBATCH -N 1
#SBATCH --output=logs/trl_qwen35_4b_sft_mix_%j.out
#SBATCH --error=logs/trl_qwen35_4b_sft_mix_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

PYTHON=/software/rome/r24.04/Miniconda3/24.7.1-0/bin/python

if ! "$PYTHON" -c "import datasets, peft, trl"; then
    echo "The pinned training Python is missing datasets/peft/trl" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

# The shared DB here has qwen36fbwikipops. Some configs/scripts use
# qwen36fbwikiprops (with an r); override when that is the real table:
#   FREEBASE_TABLE=qwen36fbwikiprops sbatch ...
FREEBASE_TABLE="${FREEBASE_TABLE:-qwen36fbwikipops}"

export INDEX="$POSTGRES_CONNECTION"

echo "Postgres index: $INDEX"
echo "Freebase table: $FREEBASE_TABLE"

mkdir -p logs

# Keep the RL questions disjoint from SFT when the mixture is available.
EXCLUDE_ARGS=()
SFT_MIX=""
for candidate in \
    logs/sft_training_mix_supported.jsonl \
    logs/sft_training_mix_supported.jsonl.gz \
    sft_training_mix_supported.jsonl.gz \
    logs/sft_training_mix.jsonl; do
    if [ -f "$candidate" ]; then
        SFT_MIX="$candidate"
        break
    fi
done
if [ -n "$SFT_MIX" ]; then
    EXCLUDE_ARGS=(--exclude-from "$SFT_MIX")
else
    echo "warning: no SFT mixture found (logs/sft_training_mix_supported.jsonl[.gz]); RL may reuse SFT questions" >&2
fi

echo "Building mixed RL dataset at $(date)"
"$PYTHON" -m utils.build_rl_dataset \
    --output logs/rl_train_mix.jsonl \
    --max-per-dataset 2000 \
    --val-per-dataset 50 \
    --count-target "${COUNT_TARGET:-1400}" \
    --allow-overlap-types count \
    --freebase-table "$FREEBASE_TABLE" \
    "${EXCLUDE_ARGS[@]}"

SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter not found at $SFT_ADAPTER (configs/trl_qwen35_4b_sft_grpo_mix.json)" >&2
    exit 1
fi

echo "Starting mixed GSPO at $(date)"
"$PYTHON" utils/run_trl.py --config configs/trl_qwen35_4b_sft_grpo_mix.json

echo "Mixed GSPO finished at $(date)"
