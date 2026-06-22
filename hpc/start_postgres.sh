#!/bin/bash
#SBATCH --output=logs/postgres_%j.out
#SBATCH --job-name=postgres
#SBATCH -N 1
#SBATCH --error=logs/postgres_%j.err
#SBATCH --time=73:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$(dirname "$0")"
source env.sh
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source postgres_utils.sh

ensure_postgres

echo ""
echo "============================================"
echo "  Postgres running at:"
echo "  Host: ${PG_HOST:-localhost}"
echo "  IP:   ${PG_IP:-127.0.0.1}"
echo "  Port: ${PG_PORT:-5432}"
echo "  PID:  ${PG_PID:-unknown}"
echo "============================================"
echo ""
echo "Keeps Postgres alive for the full wall time."
echo "scancel $SLURM_JOB_ID to stop."

sleep infinity
