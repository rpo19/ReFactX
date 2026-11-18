PS1LOGIN="${PS1LOGIN:-$(hostname -A)}"
# pass PS1LOGIN to sbatch: > sbatch "--export=ALL,PS1LOGIN=$PS1" populate.sh
echo $PS1LOGIN
WS_PATH=""
_FALLBACK_WS=""
case "$PS1LOGIN" in
    *capella*)
        WS_PATH="/data/cat/ws/$USER-horse"
        _FALLBACK_WS="/data/horse/ws/$USER-quokka"
	echo capella
        ;;
    *alpha*)
        WS_PATH="/data/horse/ws/$USER-quokka"
        _FALLBACK_WS="/data/cat/ws/$USER-horse"
	echo alpha
        ;;
    *)
        echo "ERROR: unrecognized host."
        ;;
esac

# $HOME is shared between the two clusters, but each node only mounts/writes
# one of the two workspace paths. The login-node name and the mounted
# workspace do not always agree (compute nodes here have "capella" in
# hostname -A but the data lives on the horse mount). Prefer whichever
# workspace actually exists AND is writable. Checking writability matters:
# a path can exist on the shared mount yet be read-only from the other
# cluster, and using it breaks any tool that creates directories there.
# Without this, set -e in the calling task aborts on the mkdir below.
if [ -z "$WS_PATH" ] || [ ! -d "$WS_PATH" ] || [ ! -w "$WS_PATH" ]; then
    if [ -n "$_FALLBACK_WS" ] && [ -d "$_FALLBACK_WS" ] && [ -w "$_FALLBACK_WS" ]; then
        WS_PATH="$_FALLBACK_WS"
        echo "workspace fallback: $WS_PATH"
    fi
fi

if [ -z "$WS_PATH" ]; then
    echo "ERROR: no writable workspace found (tried the host-specific path and its fallback)." >&2
    return 1 2>/dev/null || exit 1
fi

export WS_PATH

# Prefer the workspace venv built by hpc/setup_venv.sh (reads torch and the
# other deps from $WS_PATH/Lustre instead of the slow NFS home). Fall back to
# the pinned Miniconda interpreter when the venv has not been built yet.
REFACTX_PYTHON="$WS_PATH/venvs/refactx/bin/python"
if [ ! -x "$REFACTX_PYTHON" ]; then
    REFACTX_PYTHON=/software/rome/r24.04/Miniconda3/24.7.1-0/bin/python
fi
export REFACTX_PYTHON

export HF_HOME="$WS_PATH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME" \
         "$HF_HUB_CACHE" \
         "$TRANSFORMERS_CACHE" \
         "$HF_DATASETS_CACHE"

# opencode (and other XDG-aware tools) create $XDG_CACHE_HOME/opencode at
# startup and abort with EACCES if that path is not writable. Only redirect
# the cache into the workspace when the mkdir succeeds; otherwise leave
# XDG_CACHE_HOME unset so tools fall back to the default ~/.cache.
export XDG_CACHE_HOME="$WS_PATH/.cache"
if ! mkdir -p "$XDG_CACHE_HOME"; then
    echo "WARNING: cannot create $XDG_CACHE_HOME; leaving XDG_CACHE_HOME unset." >&2
    unset XDG_CACHE_HOME
fi
