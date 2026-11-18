#!/bin/bash
#SBATCH --job-name=trl_sft_mix_smoke
#SBATCH -N 1
#SBATCH --output=logs/trl_qwen35_4b_sft_grpo_smoke_%j.out
#SBATCH --error=logs/trl_qwen35_4b_sft_grpo_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

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
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps
# Chain the overlay's EXIT cleanup with the outcome notification.
trap 'rl_notify $?' EXIT

PYTHON="$REFACTX_PYTHON"

if ! "$PYTHON" -c "import datasets, peft, trl"; then
    echo "The pinned training Python is missing datasets/peft/trl" >&2
    exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SLURM_SUBMIT_DIR"
export INDEX="$POSTGRES_CONNECTION"

SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter missing at $SFT_ADAPTER" >&2
    exit 1
fi

mkdir -p logs

echo "Starting 3-step GSPO smoke from $SFT_ADAPTER at $(date)"
echo "Postgres index: $INDEX"

rm -rf /tmp/grpo_output_qwen35_4b_smoke

"$PYTHON" utils/run_trl.py \
    --config configs/trl_qwen35_4b_sft_grpo_smoke.json

echo "GSPO smoke finished at $(date)"
