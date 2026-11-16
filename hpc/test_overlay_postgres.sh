#!/bin/bash
#SBATCH --job-name=pg_overlay_test
#SBATCH --output=logs/pg_overlay_test_%j.out
#SBATCH --error=logs/pg_overlay_test_%j.err
#SBATCH -N 1
#SBATCH --time=00:40:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Concurrent copy-on-write Postgres instances over one shared, read-only PGDATA.
#
# Each instance:
#   unshare -U -m --map-user=999 --keep-caps
#     -> host uid maps to 999, so PGDATA is owned by a non-root uid
#     -> CAP_SYS_ADMIN is retained, so fuse-overlayfs can mount the Lustre lower
#     -> geteuid0.so makes libfuse take the direct mount(2) path (no fusermount3)
#     -> liblazyrw opens relation files read-only and upgrades on first write,
#        so reads never copy relation files into the per-instance upper layer
#     -> postgres runs as uid 999 in that mount namespace
#
# Instance 1 runs writable and creates an isolation_probe table; the others are
# read-only, must not see it, and must reject DDL.
#
# Usage:
#   sbatch hpc/test_overlay_postgres.sh
# Overrides:
#   INSTANCES=<n>     default 4
#   UPPER_ROOT=<dir>  default /tmp (node-local); use Lustre to be extra safe
#   OVL_TABLE=<name>  default qwen36

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

IMG="${IMG:-$WS_PATH/postgres.sif}"
GOLDEN="${GOLDEN:-$WS_PATH/pgdata}"
TABLENAME="${OVL_TABLE:-qwen36}"
INSTANCES="${INSTANCES:-4}"
UPPER_ROOT="${UPPER_ROOT:-/tmp}"
BIN="$WS_PATH/bin"
BASE="$UPPER_ROOT/pg_overlay_${SLURM_JOB_ID}"

mkdir -p "$BASE"

echo "============================================================"
echo "Concurrent overlay Postgres test"
echo "date        : $(date)"
echo "host        : $(hostname)"
echo "job         : ${SLURM_JOB_ID:-none}"
echo "GPU         : ${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi -L 2>/dev/null | head -1 || true
echo "GOLDEN      : $GOLDEN"
echo "INSTANCES   : $INSTANCES"
echo "UPPER_ROOT  : $UPPER_ROOT"
echo "BASE        : $BASE"
echo "============================================================"
df -h "$UPPER_ROOT" "$GOLDEN" 2>&1 | sed 's/^/  /'
ls -l "$BIN/fuse-overlayfs" "$BIN/geteuid0.so" "$BIN/liblazyrw-glibc-x86_64.so" 2>&1 | sed 's/^/  /'

cat >"$BASE/inner.sh" <<'INNER'
#!/bin/sh
set -u
B="$1"; lower="$2"; fo="$3"; img="$4"; tbl="$5"; bin="$6"; mode="$7"; idx="$8"
mkdir -p "$B/upper" "$B/work" "$B/merged" "$B/sock"

upper_size() { du -sh "$B/upper" 2>/dev/null | cut -f1; }
trap 'umount "$B/merged" 2>/dev/null || true' EXIT INT TERM

echo "inst$idx: id=$(id -u) mode=$mode"

LD_PRELOAD="$bin/geteuid0.so" "$fo" \
    -o "lowerdir=$lower,upperdir=$B/upper,workdir=$B/work" "$B/merged" || {
    echo "RESULT inst$idx fuse_mount=fail"; exit 1;
}
chmod 700 "$B/merged" 2>/dev/null || true

# noatime via mount(2): the mount.fuse-overlayfs helper is not installed
python3 - "$B/merged" <<'PYEOF' || { echo "RESULT inst$idx remount=fail"; exit 1; }
import ctypes, os, sys
path = sys.argv[1].encode()
MS_REMOUNT, MS_NOSUID, MS_NODEV, MS_NOATIME = 32, 2, 4, 1024
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_void_p]
libc.mount.restype = ctypes.c_int
if libc.mount(None, path, None, MS_REMOUNT | MS_NOSUID | MS_NODEV | MS_NOATIME, None) != 0:
    print("remount failed:", os.strerror(ctypes.get_errno()), file=sys.stderr)
    sys.exit(1)
PYEOF

SING="singularity exec -B $B/merged:/var/lib/postgresql/data -B $B/sock:/pgsocket -B $bin:/pgshim:ro $img"
SHIM=/pgshim/liblazyrw-glibc-x86_64.so

probe=$($SING sh -c "PGDATA=/var/lib/postgresql/data PGOVERLAY_LAZYRW_DEBUG=1 LD_PRELOAD=$SHIM postgres -V" 2>&1)
if ! printf '%s\n' "$probe" | grep -q '\[lazyrw\] active'; then
    echo "RESULT inst$idx lazyrw=inactive"
    printf '%s\n' "$probe"
    exit 1
fi
echo "RESULT inst$idx lazyrw=active"
echo "RESULT inst$idx upper_after_mount=$(upper_size)"

RO=""
[ "$mode" = "ro" ] && RO="-c default_transaction_read_only=on"

$SING sh -c "PGDATA=/var/lib/postgresql/data LD_PRELOAD=$SHIM exec postgres \
    -D /var/lib/postgresql/data -p 5432 -k /pgsocket -c listen_addresses='' \
    -c autovacuum=off -c track_counts=off -c full_page_writes=off \
    -c wal_level=minimal -c max_wal_senders=0 -c fsync=off \
    -c recovery_init_sync_method=syncfs -c io_method=worker $RO" \
    >"$B/pg.log" 2>&1 &
PG=$!

ok=0
for i in $(seq 1 120); do
    if $SING psql -h /pgsocket -p 5432 -U postgres -d postgres -tAc 'select 1' 2>/dev/null | grep -qx 1; then
        ok=1; break
    fi
    kill -0 "$PG" 2>/dev/null || break
    sleep 1
done
if [ "$ok" != 1 ]; then
    echo "RESULT inst$idx postgres_start=fail"
    tail -30 "$B/pg.log"
    exit 1
fi
echo "RESULT inst$idx postgres_start=ok"
echo "RESULT inst$idx upper_after_start=$(upper_size)"

echo "inst$idx: indexed read of $tbl"
$SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
    -c "select id, numleaves from $tbl where key = ARRAY[-200];" 2>&1 | tail -5

if [ "$mode" = "rw" ]; then
    echo "inst$idx: write isolation probe"
    $SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
        -c "create table isolation_probe(i int); insert into isolation_probe values (42); select * from isolation_probe;" \
        2>&1 | tail -6
else
    echo "inst$idx: verify isolation_probe is absent"
    $SING psql -h /pgsocket -p 5432 -U postgres -d postgres -tAc \
        "select coalesce(to_regclass('public.isolation_probe')::text, '<absent>');" 2>&1 | tail -3
    echo "inst$idx: verify DDL is rejected"
    $SING psql -h /pgsocket -p 5432 -U postgres -d postgres \
        -c "create table should_fail(i int);" 2>&1 | tail -3
fi

echo "RESULT inst$idx upper_after_query=$(upper_size)"
echo "RESULT inst$idx upper_base=$(du -sh "$B/upper/base" 2>/dev/null | cut -f1)"

$SING pg_ctl -D /var/lib/postgresql/data -m fast -w stop >/dev/null 2>&1
echo "RESULT inst$idx stop_rc=$?"
echo "RESULT inst$idx upper_after_stop=$(upper_size)"
kill "$PG" 2>/dev/null || true
exit 0
INNER
chmod +x "$BASE/inner.sh"

echo ""
echo "== launching $INSTANCES instances concurrently =="
for idx in $(seq 1 "$INSTANCES"); do
    mode="ro"
    [ "$idx" = "1" ] && mode="rw"
    (
        unshare -U -m --map-user=999 --map-group=999 --keep-caps \
            /bin/sh "$BASE/inner.sh" \
            "$BASE/inst$idx" "$GOLDEN" "$BIN/fuse-overlayfs" "$IMG" "$TABLENAME" "$BIN" "$mode" "$idx" \
            >"$BASE/inst$idx.log" 2>&1
        echo $? >"$BASE/inst$idx.rc"
    ) &
done
wait

echo ""
FAIL=0
for idx in $(seq 1 "$INSTANCES"); do
    echo "---------------- inst$idx (rc=$(cat "$BASE/inst$idx.rc" 2>/dev/null || echo missing)) ----------------"
    cat "$BASE/inst$idx.log"
    rc=$(cat "$BASE/inst$idx.rc" 2>/dev/null || echo 1)
    [ "$rc" = "0" ] || FAIL=1
    grep -q 'RESULT inst'"$idx"' lazyrw=active' "$BASE/inst$idx.log" || FAIL=1
    if [ "$idx" = "1" ]; then
        grep -q '^ *42$' "$BASE/inst$idx.log" || FAIL=1
    else
        grep -q '<absent>' "$BASE/inst$idx.log" || FAIL=1
        grep -q 'read-only transaction' "$BASE/inst$idx.log" || FAIL=1
    fi
done

echo ""
echo "============================================================"
echo "== summary =="
for idx in $(seq 1 "$INSTANCES"); do
    echo "  inst$idx: $(grep -E 'RESULT inst'"$idx"' (lazyrw|upper_after_start|upper_after_query|upper_base|stop_rc)=' "$BASE/inst$idx.log" | tr '\n' ' ')"
done
echo "  golden postmaster.pid: $(ls "$GOLDEN/postmaster.pid" 2>/dev/null || echo absent)"
if [ "$FAIL" = 0 ]; then
    echo "RESULT concurrent_overlay_test=PASS"
else
    echo "RESULT concurrent_overlay_test=FAIL"
fi
echo "== finished at $(date) =="
echo "============================================================"

rm -rf "$BASE"
exit "$FAIL"
