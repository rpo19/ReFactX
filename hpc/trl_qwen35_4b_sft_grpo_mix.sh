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

# Report the job outcome to Telegram. Installed after the Postgres overlay so
# it can stop that overlay too (see the trap below).
rl_notify() {
    local status=$1
    if [ "$status" -eq 0 ]; then
        teleclinotify "rl success | SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}" || true
    else
        teleclinotify "rl error | SLURM_JOB_ID=${SLURM_JOB_ID:-unknown} exit=$status" || true
    fi
    pg_overlay_stop || true
}

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps
# Chain the overlay's EXIT cleanup with the outcome notification.
trap 'rl_notify $?' EXIT

PYTHON=/software/rome/r24.04/Miniconda3/24.7.1-0/bin/python

if ! "$PYTHON" -c "import datasets, peft, trl"; then
    echo "The pinned training Python is missing datasets/peft/trl" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

# The shared DB on this cluster exposes the Freebase table as
# qwen36fbwikiprops (with an r); override when the target DB differs:
#   FREEBASE_TABLE=qwen36fbwikipops sbatch ...
FREEBASE_TABLE="${FREEBASE_TABLE:-qwen36fbwikiprops}"

export INDEX="$POSTGRES_CONNECTION"

echo "Postgres index: $INDEX"
echo "Freebase table: $FREEBASE_TABLE"

mkdir -p logs

# The RL mixture lives in the shared workspace so it can be uploaded once with
# scp and reused by every job (see error_categorization/rl_mix_training.md).
DATASET_DIR="$WS_PATH/datasets"
DATASET_PATH="$DATASET_DIR/rl_train_mix.jsonl"
mkdir -p "$DATASET_DIR"

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

if [ -f "$DATASET_PATH" ] && [ "${REBUILD:-0}" != "1" ]; then
    echo "Using existing RL dataset: $DATASET_PATH (REBUILD=1 to rebuild)"
else
    echo "Building mixed RL dataset at $DATASET_PATH ($(date))"
    "$PYTHON" -m utils.build_rl_dataset \
        --output "$DATASET_PATH" \
        --max-per-dataset 2000 \
        --val-per-dataset 50 \
        --count-target "${COUNT_TARGET:-1400}" \
        --allow-overlap-types count \
        --freebase-table "$FREEBASE_TABLE" \
        "${EXCLUDE_ARGS[@]}"
fi

SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter not found at $SFT_ADAPTER (configs/trl_qwen35_4b_sft_grpo_mix_cluster.json)" >&2
    exit 1
fi

RL_CONFIG="${RL_CONFIG:-configs/trl_qwen35_4b_sft_grpo_mix_cluster.json}"
echo "Starting mixed GSPO at $(date) with $RL_CONFIG"
"$PYTHON" utils/run_trl.py --config "$RL_CONFIG"

echo "Mixed GSPO finished at $(date)"
