#!/bin/bash
#SBATCH --output=logs/eval_test_%j.out
#SBATCH --job-name=refactx_test
#SBATCH -N 1
#SBATCH --error=logs/eval_test_%j.err
#SBATCH --time=30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
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

echo "Starting Mintaka test eval (2B model, 10 samples) at $(date)"
python -m utils.eval --config configs/config_mintaka_qwen35_2b_test.json
echo "Done at $(date)"
