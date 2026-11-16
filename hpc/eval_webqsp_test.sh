#!/bin/bash
#SBATCH --output=logs/eval_webqsp_test_%j.out
#SBATCH --job-name=refactx_webqsp_test
#SBATCH -N 1
#SBATCH --error=logs/eval_webqsp_test_%j.err
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

python -m utils.eval --config configs/config_webqsp_qwen36_35b3_test.json

teleclinotify "refactx_webqsp_test done | SLURM_JOB_ID=$SLURM_JOB_ID"
