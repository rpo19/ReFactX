#!/bin/bash
# Build a reusable Python environment on the workspace filesystem.
#
# The training dependencies currently live in
# ~/.local/lib/pythonX.Y/site-packages on the NFS home. Importing torch from
# there is very slow on compute nodes (cold reads of hundreds of MB of shared
# libraries), which delays every job by several minutes. This script mirrors
# that user site-packages tree onto the Lustre workspace and creates a venv
# that reads from the mirror, so imports come from $WS_PATH instead.
#
# Re-run it after installing new packages with `pip --user` to sync the mirror.
#
# Usage: bash hpc/setup_venv.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=hpc/env.sh
source "$HERE/env.sh"

BASE_PY="${BASE_PY:-/software/rome/r24.04/Miniconda3/24.7.1-0/bin/python}"
PYVER="$("$BASE_PY" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
SRC_SITE="$HOME/.local/lib/python${PYVER}/site-packages"
MIRROR="$WS_PATH/pyenv/site-packages"
VENV="$WS_PATH/venvs/refactx"

if [ ! -d "$SRC_SITE" ]; then
    echo "No user site-packages at $SRC_SITE" >&2
    exit 1
fi

echo "Mirroring $SRC_SITE -> $MIRROR"
mkdir -p "$MIRROR"
if command -v rsync >/dev/null 2>&1; then
    # No --delete: never remove files here, and a re-run resumes/syncs.
    rsync -aH --info=progress2 "$SRC_SITE"/ "$MIRROR"/
else
    cp -a "$SRC_SITE"/. "$MIRROR"/
fi

if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating venv at $VENV"
    "$BASE_PY" -m venv --system-site-packages "$VENV"
fi

VENV_SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
printf '%s\n' "$MIRROR" > "$VENV_SITE/zz_refactx_mirror.pth"

echo "Verifying imports with $VENV/bin/python ..."
"$VENV/bin/python" -c \
    'import torch, datasets, peft, trl, transformers, wandb; print("OK", "torch", torch.__version__, "trl", trl.__version__)'
echo "Done. Point the HPC scripts at $VENV/bin/python (env.sh does this automatically)."
