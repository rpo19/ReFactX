#!/bin/bash
#SBATCH --output=logs/refactx_eval_%j.out
#SBATCH --job-name=refactx_eval
#SBATCH -N 1
#SBATCH --error=logs/refactx_eval_%j.err
#SBATCH --time=73:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
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

python -m utils.eval --config configs/config_2wiki_qwen36_35b3.json

teleclinotify "refactx_eval (twowikimultihop) done | SLURM_JOB_ID=$SLURM_JOB_ID"
