#!/bin/bash
#SBATCH --output=logs/eval_sft_test_%j.out
#SBATCH --error=logs/eval_sft_test_%j.err
#SBATCH --job-name=eval_sft_supported_mix
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

# Evals load the adapter directly via each config's `adapter_path`; no merge.
SFT_ADAPTER="$WS_PATH/sft_output_supported_mix"
if [ ! -f "$SFT_ADAPTER/adapter_config.json" ]; then
    echo "SFT adapter not found at $SFT_ADAPTER" >&2
    exit 1
fi

echo "Starting 4 SFT sample evals (200 examples each) at $(date)"
echo "Adapter: $SFT_ADAPTER"

# Wikidata index: Mintaka + 2Wiki.
CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config configs/config_mintaka_sft_test.json \
  &> logs/eval_mintaka_sft_${SLURM_JOB_ID}.log &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config configs/config_2wiki_sft_test.json \
  &> logs/eval_2wiki_sft_${SLURM_JOB_ID}.log &
PID2=$!

wait $PID1 $PID2
echo "Wikidata evals completed at $(date)"

# Freebase index: WebQSP + CWQ.
CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config configs/config_webqsp_sft_test.json \
  &> logs/eval_webqsp_sft_${SLURM_JOB_ID}.log &
PID3=$!

CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config configs/config_cwq_sft_test.json \
  &> logs/eval_cwq_sft_${SLURM_JOB_ID}.log &
PID4=$!

wait $PID3 $PID4
echo "All SFT evals completed at $(date)"
