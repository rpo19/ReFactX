#!/bin/bash
#SBATCH --job-name=trl_qwen35_4b_sft
#SBATCH -N 1
#SBATCH --output=logs/trl_qwen35_4b_sft_%j.out
#SBATCH --error=logs/trl_qwen35_4b_sft_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

# Pin the base Miniconda interpreter. `python` resolves differently across
# compute nodes and may not contain the installed training dependencies.
PYTHON=/software/rome/r24.04/Miniconda3/24.7.1-0/bin/python

if ! "$PYTHON" -c "import datasets, peft, trl"; then
    echo "The pinned training Python is missing datasets/peft/trl" >&2
    exit 1
fi

# Reduce allocator fragmentation during long generation/training jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

export INDEX="$POSTGRES_CONNECTION"

echo "Postgres index: $INDEX"

# Must match the `adapter` path in configs/trl_qwen35_4b_sft_grpo.json.
SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter missing at $SFT_ADAPTER" >&2
    exit 1
fi

mkdir -p logs

echo "Starting GSPO from $SFT_ADAPTER at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"

"$PYTHON" utils/run_trl.py \
    --config configs/trl_qwen35_4b_sft_grpo.json

echo "GRPO training finished at $(date)"
