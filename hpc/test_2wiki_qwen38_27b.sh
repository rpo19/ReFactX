#!/bin/bash
#SBATCH --output=logs/test_2wiki_dev_qwen38_27b_%j.out
#SBATCH --job-name=refactx_2wiki_dev_qwen38
#SBATCH -N 1
#SBATCH --error=logs/test_2wiki_dev_qwen38_27b_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:4

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

cd /home/ripo631h/ReFactX

# 2Wiki test is unlabeled; the benchmark is the dev split (validation).
PYTHON="${REFACTX_PYTHON:-python}"
"$PYTHON" -m utils.eval --config configs/config_2wiki_qwen38_27b_dev.json

teleclinotify "test_2wiki_dev_qwen38_27b done | SLURM_JOB_ID=$SLURM_JOB_ID"
