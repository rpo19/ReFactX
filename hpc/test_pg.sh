#!/bin/bash
#SBATCH --output=logs/test_pg_%j.out
#SBATCH --job-name=test_pg
#SBATCH -N 1
#SBATCH --error=logs/test_pg_%j.err
#SBATCH --time=10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

NODE=$(hostname -A | awk '{print $1}')
echo "=== Node: $NODE ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

echo ""
echo "--- ensure_postgres ---"
ensure_postgres

echo ""
echo "--- Testing connection (TCP) ---"
singularity exec \
  -B "$PGDATA:/var/lib/postgresql/data" \
  "$IMG" \
  psql -h "$PG_IP" -p "$PG_PORT" -U postgres -d postgres \
  -c "SELECT 'Connection OK from $NODE' AS message;"

echo ""
echo "=== TEST PASSED on $NODE ==="
