# shellcheck shell=bash
# ===========================================================================
# Per-job, disposable PostgreSQL for ReFactX Slurm tasks
# ===========================================================================
#
# Every task that needs the KG index starts its own PostgreSQL on a
# copy-on-write view of the shared, read-only PGDATA. Nothing is shared
# between jobs except the immutable seed and the binaries below.
#
#   source hpc/postgres_overlay.sh
#   pg_overlay_start ro          # or: rw (writers only)
#   pg_overlay_install_traps
#   ... python -m utils.eval ...  # BASE_INDEX_PATH/POSTGRES_CONNECTION are set
#
# Why not one shared server?
# --------------------------
# PGDATA is single-writer: two postmasters on the same data directory corrupt
# it (we hit "invalid resource manager ID in checkpoint record"). The previous
# postgres_utils.sh tried to elect one owner per node through
# $WS_PATH/postgres.addr, with a watchdog in every job that restarted it. That
# race is what corrupted the cluster, and it made every task depend on a
# long-lived service job. One private server per Slurm task removes the race
# entirely.
#
# Why not mount PGDATA read-only?
# -------------------------------
# PostgreSQL cannot start on a read-only data directory. Even a cleanly shut
# down cluster needs to create $PGDATA/postmaster.pid, and a running primary
# writes pg_control, WAL, pg_xact (hint bits), pg_stat, temp files, ...
# `default_transaction_read_only=on` only rejects client DML; it does not make
# the storage read-only. So each job gets a writable *view*.
#
# The copy-on-write stack, and the workarounds needed on this cluster
# ------------------------------------------------------------------
#   lower : $WS_PATH/pgdata (immutable seed, 307 GB, on Lustre)
#   upper : /tmp/pg_overlay.<job>.<mode>.XXXX (node-local, per job)
#   merged: PGDATA seen by PostgreSQL
#
# 1. Kernel overlayfs rejects the Lustre lower ("overlay: filesystem on
#    /lower not supported"), so we use fuse-overlayfs.
# 2. The compute nodes have neither fuse-overlayfs nor fusermount3. We ship
#    both workarounds in $WS_PATH/bin:
#      - fuse-overlayfs  (copied from the login node; same EL9 userland)
#      - geteuid0.so     (LD_PRELOAD only for fuse-overlayfs; libfuse only
#                         takes the direct mount(2) path when geteuid()==0,
#                         and fusermount3 is missing. getuid() is untouched
#                         because libfuse passes it as the mount's user_id.)
# 3. To mount FUSE without fusermount3 and without being root, each instance
#    runs in `unshare -U -m --map-user=999 --map-group=999 --keep-caps`.
#    The host uid maps to uid 999, so PostgreSQL sees PGDATA owned by 999 and
#    passes its "refuse to run as root" check, while CAP_SYS_ADMIN is kept in
#    the namespace so FUSE can mount.
# 4. fuse-overlayfs reports the merged root as 0755; PostgreSQL requires
#    0700/0750, so the inner script chmods the mount point. It also remounts
#    with noatime via mount(2) (the mount.fuse-overlayfs helper is absent).
# 5. Reads would still copy whole relations: PostgreSQL opens every relation
#    segment O_RDWR even to read it (md.c), and overlayfs copies a lower file
#    whole on a read-write open. We use pgoverlay's `liblazyrw` shim
#    (Apache-2.0, https://github.com/abd-ulbasit/pgoverlay, internal/cow/
#    lazyrw): it opens relation files O_RDONLY and reopens one O_RDWR on its
#    first actual write. With the seed frozen this copies nothing on reads.
# 6. `-c recovery_init_sync_method=syncfs` (PG >= 14) avoids the pre-recovery
#    per-file O_RDWR fsync pass, which would copy the whole data directory.
# 7. `-c io_method=worker` (PG 18) keeps relation I/O in libc calls; the shim
#    cannot see io_uring submissions.
#
# The seed must be cleanly shut down and ideally frozen (ReFactX populates
# with COPY ... FREEZE). Without that, the first read may set hint bits and
# lazily upgrade a relation file. A one-off VACUUM (FREEZE, ANALYZE) settles
# it, as pgoverlay's seed-settle step does.
#
# Writable tasks (populate_*.sh) use `pg_overlay_start rw`: the same namespace
# trick, but PostgreSQL runs directly on $WS_PATH/pgdata with no overlay. A
# lock directory ($WS_PATH/pgdata.write.lock) keeps writers exclusive, and
# read-only tasks refuse to start while it exists. Never run a writer while
# readers are live: the lower must be immutable.
#
# Binaries required in $WS_PATH/bin:
#   fuse-overlayfs            copied from the login node (same EL9 userland)
#   geteuid0.so               gcc -shared -fPIC -O2 -o "$WS_PATH/bin/geteuid0.so" \
#                                 hpc/geteuid0.c -ldl
#   liblazyrw-glibc-x86_64.so from pgoverlay internal/cow/lazyrw/dist (Apache-2.0),
#                             checksum-verified against its SHA256SUMS
# Overrides: PG_OVERLAY_IMAGE, PG_OVERLAY_GOLDEN, PG_OVERLAY_BIN,
#            PG_OVERLAY_UPPER_ROOT (default /tmp), PG_OVERLAY_KEEP=1.
# ===========================================================================

if [ -z "${WS_PATH:-}" ]; then
    _pg_overlay_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck disable=SC1091
    source "$_pg_overlay_here/env.sh"
fi

PG_OVERLAY_ACTIVE="${PG_OVERLAY_ACTIVE:-0}"
PG_OVERLAY_PID=""
PG_OVERLAY_BASE=""
PG_OVERLAY_LOCK=""
PG_OVERLAY_MODE=""
PG_OVERLAY_BIN="${PG_OVERLAY_BIN:-$WS_PATH/bin}"
PG_OVERLAY_IMAGE="${PG_OVERLAY_IMAGE:-$WS_PATH/postgres.sif}"
PG_OVERLAY_GOLDEN="${PG_OVERLAY_GOLDEN:-$WS_PATH/pgdata}"
PG_OVERLAY_UPPER_ROOT="${PG_OVERLAY_UPPER_ROOT:-/tmp}"

pg_overlay_start() {
    local mode="${1:-ro}"
    local job="${SLURM_JOB_ID:-$$}"
    local pgmajor=""

    if [ "$PG_OVERLAY_ACTIVE" = "1" ]; then
        echo "[pg_overlay] already running ($PG_OVERLAY_MODE)" >&2
        return 0
    fi
    case "$mode" in
        ro|rw) ;;
        *) echo "[pg_overlay] unknown mode '$mode' (expected ro or rw)" >&2; return 2 ;;
    esac

    local missing=0 f
    for f in "$PG_OVERLAY_IMAGE"; do
        [ -f "$f" ] || { echo "[pg_overlay] missing $f" >&2; missing=1; }
    done
    if [ "$mode" = "ro" ]; then
        for f in "$PG_OVERLAY_BIN/fuse-overlayfs" "$PG_OVERLAY_BIN/geteuid0.so" \
                 "$PG_OVERLAY_BIN/liblazyrw-glibc-x86_64.so"; do
            [ -f "$f" ] || { echo "[pg_overlay] missing $f" >&2; missing=1; }
        done
    fi
    [ -f "$PG_OVERLAY_GOLDEN/PG_VERSION" ] || {
        echo "[pg_overlay] no seed at $PG_OVERLAY_GOLDEN (run hpc/postgres_init.sh once)" >&2
        missing=1
    }
    [ "$missing" = "0" ] || return 1

    # Writers are exclusive; readers refuse to start while a writer holds the
    # lock. (The reverse race is tiny: a reader that checks just before the
    # writer creates the lock. Population is a deliberate, manual operation.)
    if [ "$mode" = "rw" ]; then
        PG_OVERLAY_LOCK="$PG_OVERLAY_GOLDEN.write.lock"
        if ! mkdir "$PG_OVERLAY_LOCK" 2>/dev/null; then
            echo "[pg_overlay] another writer holds $PG_OVERLAY_LOCK" >&2
            PG_OVERLAY_LOCK=""
            return 1
        fi
    elif [ -d "$PG_OVERLAY_GOLDEN.write.lock" ]; then
        echo "[pg_overlay] $PG_OVERLAY_GOLDEN is being written ($PG_OVERLAY_GOLDEN.write.lock); refusing read-only start" >&2
        return 1
    fi

    PG_OVERLAY_BASE="$(mktemp -d "$PG_OVERLAY_UPPER_ROOT/pg_overlay.${job}.${mode}.XXXXXX")" || {
        [ -n "$PG_OVERLAY_LOCK" ] && rmdir "$PG_OVERLAY_LOCK" 2>/dev/null
        PG_OVERLAY_LOCK=""
        return 1
    }
    PG_OVERLAY_MODE="$mode"

    pgmajor="$(tr -cd '0-9' < "$PG_OVERLAY_GOLDEN/PG_VERSION" 2>/dev/null || true)"
    [ -n "$pgmajor" ] || pgmajor=0

    cat > "$PG_OVERLAY_BASE/inner.sh" <<'INNER'
#!/bin/sh
set -u
B="$1"; mode="$2"; golden="$3"; img="$4"; bin="$5"; pgmajor="$6"
mkdir -p "$B/upper" "$B/work" "$B/merged" "$B/sock"

SING=""
PRELOAD=""
EXTRA=""

if [ "$mode" = "ro" ]; then
    LD_PRELOAD="$bin/geteuid0.so" "$bin/fuse-overlayfs" \
        -o "lowerdir=$golden,upperdir=$B/upper,workdir=$B/work" "$B/merged" || {
        echo "[pg_overlay] fuse-overlayfs mount failed" >&2
        echo "[pg_overlay] if this says 'cannot mount', check /dev/fuse and geteuid0.so" >&2
        exit 1
    }
    chmod 700 "$B/merged" 2>/dev/null || true
    # noatime through mount(2): the mount.fuse-overlayfs helper is not installed
    python3 - "$B/merged" <<'PYEOF' || exit 1
import ctypes, os, sys
path = sys.argv[1].encode()
MS_REMOUNT, MS_NOSUID, MS_NODEV, MS_NOATIME = 32, 2, 4, 1024
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                       ctypes.c_ulong, ctypes.c_void_p]
libc.mount.restype = ctypes.c_int
if libc.mount(None, path, None,
              MS_REMOUNT | MS_NOSUID | MS_NODEV | MS_NOATIME, None) != 0:
    print("noatime remount failed:", os.strerror(ctypes.get_errno()), file=sys.stderr)
    sys.exit(1)
PYEOF
    # The seed may have been copied after a crash; these must not stop us.
    rm -f "$B/merged/postmaster.pid" "$B/merged/standby.signal" "$B/merged/recovery.signal"
    SING="singularity exec -B $B/merged:/var/lib/postgresql/data -B $B/sock:/pgsocket -B $bin:/pgshim:ro $img"
    PRELOAD="LD_PRELOAD=/pgshim/liblazyrw-glibc-x86_64.so"
    # A read-only workload only needs its own WAL/xact/stat writes; relation
    # writes are hint bits and are safe to disable.
    EXTRA="-c default_transaction_read_only=on -c full_page_writes=off -c wal_level=minimal -c fsync=off"
else
    # Writer: no overlay, write the seed directly. Exclusive by the outer lock.
    if [ -f "$golden/postmaster.pid" ]; then
        echo "[pg_overlay] $golden/postmaster.pid exists; refusing to start a writer" >&2
        exit 1
    fi
    SING="singularity exec -B $golden:/var/lib/postgresql/data -B $B/sock:/pgsocket $img"
fi

if [ "$pgmajor" -ge 14 ] 2>/dev/null; then
    EXTRA="$EXTRA -c recovery_init_sync_method=syncfs"
fi
if [ "$pgmajor" -ge 18 ] 2>/dev/null; then
    EXTRA="$EXTRA -c io_method=worker"
fi

cleanup() {
    trap - INT TERM
    $SING pg_ctl -D /var/lib/postgresql/data -m fast -w stop >/dev/null 2>&1 || true
    if [ "$mode" = "ro" ]; then
        umount "$B/merged" 2>/dev/null || true
    fi
}
trap cleanup INT TERM
trap 'cleanup; exit 0' EXIT

$SING sh -c "PGDATA=/var/lib/postgresql/data ${PRELOAD:+$PRELOAD }exec postgres \
    -D /var/lib/postgresql/data -p 5432 -k /pgsocket -c listen_addresses='' \
    -c autovacuum=off -c max_wal_senders=0 -c archive_mode=off -c ssl=off $EXTRA" \
    >"$B/pg.log" 2>&1 &
SING_PID=$!
wait "$SING_PID"
INNER
    chmod 700 "$PG_OVERLAY_BASE/inner.sh"

    # The Unix socket lives on the shared filesystem, so the task shell (in the
    # host mount namespace) can connect to the server started inside the
    # namespace: PGHOST points at the host directory.
    export PGHOST="$PG_OVERLAY_BASE/sock"
    export PGPORT=5432
    export PGUSER=postgres
    # No query string here: utils/eval.py appends "?tablename=...".
    export INDEX_PATH="postgresql:///postgres"
    export BASE_INDEX_PATH="$INDEX_PATH"
    export POSTGRES_CONNECTION="$INDEX_PATH"
    export INDEX="$INDEX_PATH"

    unshare -U -m --map-user=999 --map-group=999 --keep-caps \
        /bin/sh "$PG_OVERLAY_BASE/inner.sh" \
        "$PG_OVERLAY_BASE" "$mode" "$PG_OVERLAY_GOLDEN" "$PG_OVERLAY_IMAGE" \
        "$PG_OVERLAY_BIN" "$pgmajor" \
        >"$PG_OVERLAY_BASE/outer.log" 2>&1 &
    PG_OVERLAY_PID=$!
    PG_OVERLAY_ACTIVE=1

    local i ready=0
    for i in $(seq 1 240); do
        if python3 -c 'import psycopg; psycopg.connect("postgresql:///postgres", connect_timeout=2).close()' >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$PG_OVERLAY_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [ "$ready" != "1" ]; then
        echo "[pg_overlay] $mode PostgreSQL failed to become ready" >&2
        tail -30 "$PG_OVERLAY_BASE/pg.log" >&2 2>/dev/null || true
        tail -20 "$PG_OVERLAY_BASE/outer.log" >&2 2>/dev/null || true
        pg_overlay_stop
        return 1
    fi

    echo "[pg_overlay] $mode PostgreSQL ready (socket $PGHOST, upper $PG_OVERLAY_BASE/upper)"
    return 0
}

pg_overlay_stop() {
    [ "${PG_OVERLAY_ACTIVE:-0}" = "1" ] || {
        [ -n "${PG_OVERLAY_LOCK:-}" ] && rmdir "$PG_OVERLAY_LOCK" 2>/dev/null
        PG_OVERLAY_LOCK=""
        return 0
    }
    PG_OVERLAY_ACTIVE=0

    local pid="${PG_OVERLAY_PID:-}"
    local i
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        # The inner shell traps TERM, stops Postgres with pg_ctl -m fast, and
        # unmounts the overlay.
        kill -TERM "$pid" 2>/dev/null || true
        for i in $(seq 1 60); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true

    if [ -n "${PG_OVERLAY_LOCK:-}" ]; then
        rmdir "$PG_OVERLAY_LOCK" 2>/dev/null || true
    fi
    PG_OVERLAY_LOCK=""

    if [ "${PG_OVERLAY_KEEP:-0}" = "1" ]; then
        echo "[pg_overlay] kept $PG_OVERLAY_BASE (PG_OVERLAY_KEEP=1)" >&2
    else
        rm -rf "$PG_OVERLAY_BASE"
    fi
    unset PGHOST PGPORT PGUSER
    PG_OVERLAY_PID=""
    PG_OVERLAY_BASE=""
    PG_OVERLAY_MODE=""
}

pg_overlay_install_traps() {
    trap 'pg_overlay_stop' EXIT
    trap 'pg_overlay_stop; exit 130' INT TERM
}
