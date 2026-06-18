#!/bin/bash
#SBATCH --output=logs/postgres_%j.out
#SBATCH --job-name=postgres
#SBATCH -N 1
#SBATCH --error=logs/postgres_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$(dirname "$0")"
source env.sh

# -----------------------
# CONFIG
# -----------------------
IMG=$WS_PATH/postgres.sif
PGDATA=$WS_PATH/pgdata
PGSOCK=$WS_PATH/pgsocket
PORT=5432
ADDR_FILE=$WS_PATH/postgres.addr

# -----------------------
# GET NODE IP
# -----------------------
NODE_HOST=$(hostname -A | awk '{print $1}')
NODE_IP=$(hostname -I | awk '{print $1}')
echo "Node hostname: $NODE_HOST"
echo "Node IP: $NODE_IP"

# -----------------------
# START POSTGRES
# -----------------------
echo "Starting Postgres on port $PORT..."

singularity exec \
  -B "$PGDATA:/var/lib/postgresql/data" \
  -B "$PGSOCK:/pgsocket" \
  "$IMG" \
  postgres \
    -D /var/lib/postgresql/data \
    -p "$PORT" \
    -k /pgsocket \
    -c listen_addresses='*' \
  > "$PGDATA/logfile" 2>&1 &

PG_PID=$!
echo "Postgres PID: $PG_PID"

# ---- cleanup ----
cleanup() {
  echo "Stopping Postgres (PID $PG_PID)..."
  rm -f "$ADDR_FILE"
  kill "$PG_PID" 2>/dev/null || true
  wait "$PG_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

# -----------------------
# WAIT FOR POSTGRES
# -----------------------
echo "Waiting for Postgres to be ready..."
for i in $(seq 1 30); do
  if singularity exec \
    -B "$PGDATA:/var/lib/postgresql/data" \
    -B "$PGSOCK:/pgsocket" \
    "$IMG" \
    psql -h /pgsocket -p "$PORT" -U postgres -d postgres -c "SELECT 1;" 2>/dev/null; then
    echo "Postgres is ready!"
    break
  fi
  sleep 2
done

# -----------------------
# WRITE ADDRESS FILE
# -----------------------
cat > "$ADDR_FILE" <<EOF
# Postgres connection info - written by start_postgres.sh (SLURM_JOB_ID=$SLURM_JOB_ID)
# This file exists only while Postgres is running. Remove to stop.
PG_HOST=$NODE_HOST
PG_IP=$NODE_IP
PG_PORT=$PORT
PGPASSWORD=${PGPASSWORD:-}
PG_SLURM_JOB_ID=$SLURM_JOB_ID
PG_PID=$PG_PID
EOF
echo "Wrote $ADDR_FILE"

echo ""
echo "============================================"
echo "  Postgres running at:"
echo "  Host: $NODE_HOST"
echo "  IP:   $NODE_IP"
echo "  Port: $PORT"
echo "  JDBC: jdbc:postgresql://$NODE_IP:$PORT/postgres"
echo "  PG:   psql -h $NODE_IP -p $PORT -U postgres -d postgres"
echo "============================================"
echo ""
echo "Job will keep Postgres alive for the full wall time."
echo "Cancel the job (scancel $SLURM_JOB_ID) when done."

# Keep alive
sleep infinity
