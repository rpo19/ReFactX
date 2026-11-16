#!/bin/bash
#SBATCH --output=logs/eval_2wiki_sft_%j.out
#SBATCH --error=logs/eval_2wiki_sft_%j.err
#SBATCH --job-name=sft_2wiki_test
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
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

# The config loads the adapter directly via `adapter_path`; no merge step.
echo "Starting SFT 2Wiki test eval at $(date)"
python -m utils.eval --config configs/config_2wiki_sft_test.json
echo "SFT 2Wiki test eval finished at $(date)"
