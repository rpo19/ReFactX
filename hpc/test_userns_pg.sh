#!/bin/bash
#SBATCH --job-name=userns_pg_probe
#SBATCH --output=logs/userns_pg_probe_%j.out
#SBATCH --error=logs/userns_pg_probe_%j.err
#SBATCH -N 1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Single-instance probe with copy-up instrumentation:
#   unshare -U -m --map-user=999 --keep-caps
#     -> host uid maps to uid 999, so PGDATA is owned by 999 (Postgres ok)
#     -> CAP_SYS_ADMIN retained, so fuse-overlayfs can mount the Lustre lower
#     -> geteuid0.so makes libfuse use direct mount(2) (no fusermount3)
#     -> postgres runs as uid 999 in the same mount namespace
#
# Measures upper-layer growth after mount / startup / EXPLAIN / indexed query.

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then ROOT="$SLURM_SUBMIT_DIR"; else ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; fi
if [ -z "${PS1LOGIN:-}" ]; then
    if [ -d "/data/horse/ws/${USER}-quokka" ]; then PS1LOGIN="alpha"
    elif [ -d "/data/cat/ws/${USER}-horse" ]; then PS1LOGIN="capella"; fi
fi
# shellcheck disable=SC1091
source "$ROOT/hpc/env.sh"
if [ ! -d "$WS_PATH" ]; then
    for candidate in "/data/horse/ws/${USER}-quokka" "/data/cat/ws/${USER}-horse"; do
        if [ -d "$candidate" ]; then WS_PATH="$candidate"; export WS_PATH; break; fi
    done
fi
export WS_PATH

UPPER_ROOT="${UPPER_ROOT:-/tmp}"
WORK="$UPPER_ROOT/userns_pg_${SLURM_JOB_ID}"
mkdir -p "$WORK"

cat >"$WORK/inner.sh" <<'INNER'
#!/bin/sh
set -u
B="$1"; lower="$2"; fo="$3"; img="$4"; tbl="$5"; preload="$6"; SHIMDIR="$7"
mkdir -p "$B/upper" "$B/work" "$B/merged" "$B/sock"

upper_size() { du -sh "$B/upper" 2>/dev/null | cut -f1; }

echo "inside id: $(id)"
echo "golden owner: $(stat -c '%u:%g %a' "$lower")"
echo "base: $B"

echo "--- fuse-overlayfs (direct mount via geteuid0.so) ---"
LD_PRELOAD="$preload" "$fo" \
    -o "lowerdir=$lower,upperdir=$B/upper,workdir=$B/work" "$B/merged"
echo "RESULT fuse_mount_rc=$?"
ls "$B/merged" | head -3
chmod 700 "$B/merged" 2>&1 || true

# The mount helper for fuse-overlayfs is not installed, so `mount -o remount`
# fails. Remount through mount(2) directly (we hold CAP_SYS_ADMIN here).
# noatime is essential: without it, reading one page of a huge relation updates
# its atime and fuse-overlayfs copies the whole file into the upper layer.
python3 - "$B/merged" <<'PYEOF'
import ctypes, os, sys
path = sys.argv[1].encode()
MS_REMOUNT, MS_NOSUID, MS_NODEV, MS_NOATIME = 32, 2, 4, 1024
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_void_p]
libc.mount.restype = ctypes.c_int
rc = libc.mount(None, path, None, MS_REMOUNT | MS_NOSUID | MS_NODEV | MS_NOATIME, None)
if rc != 0:
    print("RESULT remount_noatime=fail", os.strerror(ctypes.get_errno()))
    sys.exit(1)
print("RESULT remount_noatime=ok")
PYEOF
grep " $B/merged " /proc/mounts || true
echo "merged owner: $(stat -c '%u:%g %a' "$B/merged")"
echo "RESULT upper_after_mount=$(upper_size)"

SING="singularity exec -B $B/merged:/var/lib/postgresql/data -B $B/sock:/pgsocket -B $SHIMDIR:/pgshim:ro $img"

echo "--- container identity ---"
$SING sh -c 'id; stat -c "pgdata %u:%g %a" /var/lib/postgresql/data' 2>&1 | sed 's/^/  /'

echo "--- starting postgres ---"
$SING sh -c 'PGDATA=/var/lib/postgresql/data PGOVERLAY_LAZYRW_DEBUG=1 \
    LD_PRELOAD=/pgshim/liblazyrw-glibc-x86_64.so exec postgres -D /var/lib/postgresql/data \
    -p 5432 -k /pgsocket -c listen_addresses="" -c autovacuum=off -c track_counts=off \
    -c default_transaction_read_only=on -c full_page_writes=off \
    -c wal_level=minimal -c max_wal_senders=0 -c fsync=off \
    -c recovery_init_sync_method=syncfs -c io_method=worker' \
    >"$B/pg.log" 2>&1 &
PG=$!

ok=0
for i in $(seq 1 120); do
    if $SING psql -h /pgsocket -p 5432 -U postgres -d postgres -tAc 'select 1' 2>/dev/null | grep -qx 1; then
        ok=1
        break
    fi
    if ! kill -0 "$PG" 2>/dev/null; then
        break
    fi
    sleep 1
done

if [ "$ok" != 1 ]; then
    echo "RESULT postgres_start=fail"
    tail -40 "$B/pg.log"
    umount "$B/merged" 2>/dev/null || true
    exit 1
fi
echo "RESULT postgres_start=ok"
echo "RESULT upper_after_start=$(upper_size)"

echo "--- indexes on $tbl ---"
$SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
    -c "select indexname from pg_indexes where tablename='$tbl';" 2>&1 | tail -8

echo "--- EXPLAIN (default planner) ---"
$SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
    -c "explain (costs off) select id, numleaves from $tbl where key = ARRAY[-200];" 2>&1 | tail -12
echo "RESULT upper_after_explain=$(upper_size)"

echo "--- indexed query (enable_seqscan=off) ---"
$SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
    -c "set enable_seqscan=off;" \
    -c "explain (costs off) select id, numleaves from $tbl where key = ARRAY[-200];" \
    -c "select id, numleaves from $tbl where key = ARRAY[-200];" 2>&1 | tail -16
echo "RESULT upper_after_query=$(upper_size)"

echo "--- graceful stop ---"
$SING pg_ctl -D /var/lib/postgresql/data -m fast -w stop 2>&1 | tail -3
echo "RESULT graceful_stop_rc=$?"
echo "RESULT upper_after_stop=$(upper_size)"

kill "$PG" 2>/dev/null || true
echo "--- upper tree (top) ---"
du -sh "$B/upper"/* 2>/dev/null | sort -h | tail -8
umount "$B/merged" 2>/dev/null || true
INNER
chmod +x "$WORK/inner.sh"

echo "host=$(hostname) job=$SLURM_JOB_ID WS_PATH=$WS_PATH UPPER_ROOT=$UPPER_ROOT"
unshare -U -m --map-user=999 --map-group=999 --keep-caps \
    /bin/sh "$WORK/inner.sh" \
    "$WORK/ovl" "$WS_PATH/pgdata" "$WS_PATH/bin/fuse-overlayfs" \
    "$WS_PATH/postgres.sif" qwen36 "$WS_PATH/bin/geteuid0.so" "$WS_PATH/bin"
echo "outer_rc=$?"
echo "== done at $(date) =="
