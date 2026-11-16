#!/bin/bash
#SBATCH --output=logs/eval_mintaka_grpo_checkpoint720_%j.out
#SBATCH --error=logs/eval_mintaka_grpo_checkpoint720_%j.err
#SBATCH --job-name=eval_mtk_grpo720
#SBATCH -N 1
#SBATCH --time=02:00:00
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

cd "$SLURM_SUBMIT_DIR"

python -m utils.eval --config configs/config_mintaka_qwen35_4b_grpo_checkpoint720_test.json

teleclinotify "Mintaka GRPO checkpoint-720 eval done | SLURM_JOB_ID=$SLURM_JOB_ID"
