#!/bin/bash
#SBATCH --output=logs/eval_prelim100_4b_%j.out
#SBATCH --error=logs/eval_prelim100_4b_%j.err
#SBATCH --job-name=prelim100_4b
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# One preliminary 100-example 4B eval. The config is passed as $1 by
# hpc/submit_eval_prelim100_4b.sh; Slurm schedules each submitted task
# independently on any free GPU.

set -euo pipefail

CFG="${1:-}"
if [ -z "$CFG" ] || [ ! -f "$CFG" ]; then
    echo "Usage: sbatch $0 <config.json>" >&2
    exit 1
fi

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
# Each attempt gets a fresh instance: a crashed server cannot poison a retry.
pg_overlay_install_traps

cd "$SLURM_SUBMIT_DIR"

echo "Starting $CFG at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID  Node: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-not set}"

EXP=$(basename "$CFG" .json)
attempt=1
max_attempts=3
while true; do
    pg_overlay_start ro
    if python -m utils.eval --config "$CFG"; then
        pg_overlay_stop
        teleclinotify "finish experiment $EXP"
        break
    else
        status=$?
        pg_overlay_stop
        if [ "$attempt" -ge "$max_attempts" ]; then
            teleclinotify "experiment $EXP failed"
            exit "$status"
        fi
        echo "Attempt $attempt failed (exit $status); retrying in 60s..." >&2
        attempt=$((attempt + 1))
        sleep 60
    fi
done

echo "Finished $CFG at $(date)"
