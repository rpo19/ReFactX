#!/bin/bash
#SBATCH --output=logs/eval_prelim100_%j.out
#SBATCH --error=logs/eval_prelim100_%j.err
#SBATCH --job-name=eval_prelim100
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

cd /home/ripo631h/ReFactX

SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter not found at $SFT_ADAPTER" >&2
    exit 1
fi

run_pair() {
    local vanilla_cfg="$1"
    local sft_cfg="$2"
    echo "Running $(basename "$vanilla_cfg" .json) + $(basename "$sft_cfg" .json) at $(date)"

    CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config "$vanilla_cfg" \
      &> "logs/$(basename "$vanilla_cfg" .json)_${SLURM_JOB_ID}.log" &
    local p1=$!

    CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config "$sft_cfg" \
      &> "logs/$(basename "$sft_cfg" .json)_${SLURM_JOB_ID}.log" &
    local p2=$!

    set +e
    wait "$p1"; local s1=$?
    wait "$p2"; local s2=$?
    set -e

    if [ "$s1" -ne 0 ] || [ "$s2" -ne 0 ]; then
        echo "Pair failed: $vanilla_cfg ($s1), $sft_cfg ($s2)" >&2
        return 1
    fi
}

echo "Starting preliminary 100-example evals at $(date)"
echo "Adapter: $SFT_ADAPTER"

run_pair configs/config_prelim100_mintaka_vanilla4b.json \
         configs/config_prelim100_mintaka_sft4b.json
run_pair configs/config_prelim100_2wiki_vanilla4b.json \
         configs/config_prelim100_2wiki_sft4b.json
run_pair configs/config_prelim100_webqsp_vanilla4b.json \
         configs/config_prelim100_webqsp_sft4b.json
run_pair configs/config_prelim100_cwq_vanilla4b.json \
         configs/config_prelim100_cwq_sft4b.json

echo "All preliminary evals completed at $(date)"
