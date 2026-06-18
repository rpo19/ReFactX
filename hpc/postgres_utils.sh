# shellcheck shell=bash
# Shared Postgres utilities - source from sbatch scripts.
# Usage: source postgres_utils.sh && ensure_postgres
#
# Starts Postgres only if not already running (checked via addr file + PID).
# Writes conn info to $ADDR_FILE for other jobs to reuse.
# Cleans up (kills Postgres, removes addr file) only if THIS process started it.

: "${PGPORT:=5432}"
: "${ADDR_FILE:=$WS_PATH/postgres.addr}"
: "${IMG:=$WS_PATH/postgres.sif}"
_NODE=$(hostname -s 2>/dev/null || echo "unknown")
: "${PGDATA:=$WS_PATH/pgdata-$_NODE}"
: "${PGSOCK:=$WS_PATH/pgsocket-$_NODE}"

ensure_postgres() {
  # 1) Reuse running Postgres if addr file + PID are alive
  if [ -f "$ADDR_FILE" ]; then
    # shellcheck disable=SC1090
    source "$ADDR_FILE"
    if [ -n "${PG_PID:-}" ] && kill -0 "$PG_PID" 2>/dev/null; then
      echo "Reusing Postgres (PID $PG_PID) at ${PG_HOST:-localhost}:${PG_PORT:-5432}"
      export PG_HOST PG_IP PG_PORT PGPASSWORD PG_SLURM_JOB_ID PG_PID
      return 0
    fi
    echo "Stale addr file (PID ${PG_PID:-unknown}). Starting fresh..."
    rm -f "$ADDR_FILE"
  fi

  # 2) Get node network info
  NODE_HOST=$(hostname -A | awk '{print $1}')
  NODE_IP=$(hostname -I | awk '{print $1}')

  # 3) Init PGDATA if needed
  mkdir -p "$PGDATA" "$PGSOCK"

  if [ ! -f "$PGDATA/PG_VERSION" ]; then
    echo "Initializing Postgres DB in $PGDATA..."
    singularity exec \
      -B "$PGDATA:/var/lib/postgresql/data" \
      "$IMG" \
      initdb -D /var/lib/postgresql/data --username=postgres
  fi

  # Always rewrite pg_hba.conf so stale PGDATA gets updated rules
  cat > "$PGDATA/pg_hba.conf" <<- EOF
local   all             all                                     trust
host    all             all             127.0.0.1/32            trust
host    all             all             ::1/128                 trust
host    all             all             0.0.0.0/0               trust
host    all             all             ::/0                    trust
EOF

  # 4) Start Postgres
  echo "Starting Postgres on port ${PGPORT}..."

  singularity exec \
    -B "$PGDATA:/var/lib/postgresql/data" \
    -B "$PGSOCK:/pgsocket" \
    "$IMG" \
    postgres \
      -D /var/lib/postgresql/data \
      -p "$PGPORT" \
      -k /pgsocket \
      -c listen_addresses='*' \
    > "$PGDATA/logfile" 2>&1 &

  PG_PID=$!

  # 4) Wait for Postgres to accept connections
  for i in $(seq 1 30); do
    if singularity exec \
      -B "$PGDATA:/var/lib/postgresql/data" \
      -B "$PGSOCK:/pgsocket" \
      "$IMG" \
      psql -h /pgsocket -p "$PGPORT" -U postgres -d postgres -c "SELECT 1;" 2>/dev/null; then
      echo "Postgres is ready (PID $PG_PID)"
      break
    fi
    sleep 2
  done

  # 4b) Fallback: if Postgres died (race with another starter), reuse theirs
  if ! kill -0 "$PG_PID" 2>/dev/null; then
    echo "Postgres died during startup (likely another job claimed PGDATA)."
    if [ -f "$ADDR_FILE" ]; then
      source "$ADDR_FILE"
      if [ -n "${PG_PID:-}" ] && kill -0 "$PG_PID" 2>/dev/null; then
        echo "Reusing Postgres (PID $PG_PID) started by another job."
        export PG_HOST PG_IP PG_PORT PGPASSWORD PG_SLURM_JOB_ID PG_PID
        return 0
      fi
    fi
    echo "FATAL: No running Postgres found. Check $PGDATA/logfile"
    return 1
  fi

  # 5) Write addr file for other jobs
  cat > "$ADDR_FILE" <<EOF
# Postgres connection info - started by SLURM_JOB_ID=$SLURM_JOB_ID
PG_HOST=$NODE_HOST
PG_IP=$NODE_IP
PG_PORT=$PGPORT
PGPASSWORD=${PGPASSWORD:-}
PG_SLURM_JOB_ID=$SLURM_JOB_ID
PG_PID=$PG_PID
EOF
  echo "Wrote $ADDR_FILE"

  # 6) Export conn vars for caller
  export PG_HOST="$NODE_HOST"
  export PG_IP="$NODE_IP"
  export PG_PORT="$PGPORT"
  export PG_PID

  # 7) Cleanup: kill Postgres only when the starter exits
  _pg_cleanup() {
    echo "Shutting down Postgres (PID $PG_PID)..."
    rm -f "$ADDR_FILE"
    kill "$PG_PID" 2>/dev/null || true
    wait "$PG_PID" 2>/dev/null || true
  }
  trap _pg_cleanup EXIT INT TERM
}
