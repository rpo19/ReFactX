#!/bin/bash
#SBATCH --output=logs/test_mintaka_qwen35_4b_%j.out
#SBATCH --job-name=mtk_test_qwen4b
#SBATCH -N 1
#SBATCH --error=logs/test_mintaka_qwen35_4b_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres
export POSTGRES_ADDR_FILE="$SHARED_POSTGRES"

postgres_watchdog() {
    while true; do
        if [ -f "$SHARED_POSTGRES" ]; then
            # shellcheck disable=SC1090
            source "$SHARED_POSTGRES"
        fi
        if [ -n "${PG_IP:-}" ] && [ -n "${PG_PORT:-}" ] \
            && timeout 2 bash -c "echo >/dev/tcp/$PG_IP/$PG_PORT" 2>/dev/null; then
            sleep 10
            continue
        fi
        echo "Postgres is unavailable; attempting recovery at $(date)" >&2
        ensure_postgres || echo "Postgres recovery attempt failed" >&2
    done
}

postgres_watchdog &
POSTGRES_WATCHDOG_PID=$!
cleanup_postgres_watchdog() {
    kill "$POSTGRES_WATCHDOG_PID" 2>/dev/null || true
    wait "$POSTGRES_WATCHDOG_PID" 2>/dev/null || true
}
trap cleanup_postgres_watchdog EXIT INT TERM

cd /home/ripo631h/ReFactX

if [ -f "$SHARED_POSTGRES" ]; then
    source "$SHARED_POSTGRES"
    export INDEX_PATH="postgres://postgres:${PGPASSWORD:-postgres}@${PG_IP:-127.0.0.1}:${PG_PORT:-5432}/postgres"
    export POSTGRES_CONNECTION="$INDEX_PATH"
    export BASE_INDEX_PATH="$INDEX_PATH"
fi

python -m utils.eval --config configs/config_mintaka_qwen35_4b_test.json

teleclinotify "test_mintaka_qwen35_4b_sft done | SLURM_JOB_ID=$SLURM_JOB_ID"
