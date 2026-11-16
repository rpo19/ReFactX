/*
 * LD_PRELOAD shim for fuse-overlayfs.
 *
 * libfuse3 only calls mount(2) directly when geteuid() == 0; otherwise it
 * execs fusermount3, which is not installed on the compute nodes. Inside a
 * user namespace created with `unshare --map-user=999 --keep-caps` we do have
 * CAP_SYS_ADMIN but euid is 999, so we make libfuse take the direct path.
 *
 * getuid() MUST stay untouched: libfuse passes it to the kernel as the FUSE
 * mount's user_id, and uid 0 is not mapped in this namespace.
 *
 * Only preloaded into fuse-overlayfs, never into Postgres.
 */
#include <sys/types.h>
#include <unistd.h>

uid_t geteuid(void) { return 0; }
