#!/bin/bash
#SBATCH --output=logs/test_mintaka_qwen35_4b_sft_%j.out
#SBATCH --job-name=mtk_test_dora4b
#SBATCH -N 1
#SBATCH --error=logs/test_mintaka_qwen35_4b_sft_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

cd /home/ripo631h/ReFactX

python -m utils.eval --config configs/config_mintaka_qwen35_4b_sft_test.json

teleclinotify "test_mintaka_qwen35_4b_sft done | SLURM_JOB_ID=$SLURM_JOB_ID"
