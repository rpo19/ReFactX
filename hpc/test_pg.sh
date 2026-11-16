#!/bin/bash
#SBATCH --output=logs/test_pg_%j.out
#SBATCH --job-name=test_pg
#SBATCH -N 1
#SBATCH --error=logs/test_pg_%j.err
#SBATCH --time=10:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

# Smoke test for the per-job PostgreSQL helper. Starts a read-only instance on
# a copy-on-write view of the shared PGDATA, connects to it over the private
# socket, and shuts it down. See hpc/postgres_overlay.sh for the mechanism.

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"

pg_overlay_start ro
pg_overlay_install_traps

NODE=$(hostname -A | awk '{print $1}')
echo "=== Node: $NODE ==="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Time: $(date)"
echo "Socket dir: $PGHOST"
echo "Index URL : $BASE_INDEX_PATH"

python3 - <<'PY'
import psycopg

with psycopg.connect("postgresql:///postgres") as conn:
    with conn.cursor() as cur:
        cur.execute("select current_user, current_database(), version();")
        user, db, version = cur.fetchone()
        print(f"connected as {user} to {db}")
        print(version.split(",")[0])
        cur.execute("select count(*) from pg_class;")
        print("pg_class rows:", cur.fetchone()[0])
PY

# Exercise the same code path as utils/eval.py: BASE_INDEX_PATH gets
# "?tablename=..." appended and libpq reads host/port from PGHOST/PGPORT.
python3 - <<'PY'
import os
import refactx

url = os.environ["BASE_INDEX_PATH"] + "?tablename=qwen36"
index = refactx.load_index(url)
print(f"refactx index loaded: rootkey={index.rootkey} switch_parameter={index.switch_parameter}")
PY

echo "=== TEST PASSED on $NODE ==="
