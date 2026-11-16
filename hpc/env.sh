PS1LOGIN="${PS1LOGIN:-$(hostname -A)}"
# pass PS1LOGIN to sbatch: > sbatch "--export=ALL,PS1LOGIN=$PS1" populate.sh
echo $PS1LOGIN
WS_PATH=""
case "$PS1LOGIN" in
    *capella*)
        WS_PATH="/data/cat/ws/$USER-horse"
	echo capella
        ;;
    *alpha*)
        WS_PATH="/data/horse/ws/$USER-quokka"
	echo alpha
        ;;
    *)
        echo "ERROR: unrecognized host."
        ;;
esac

# The login-node name and the mounted workspace do not always agree (compute
# nodes here have "capella" in hostname -A but the data lives on the horse
# mount). Prefer whichever workspace actually exists; this keeps the login
# name only as a hint. Without it, set -e in the calling task aborts on the
# mkdir below.
if [ ! -d "$WS_PATH" ]; then
    for _ws_candidate in "/data/horse/ws/$USER-quokka" "/data/cat/ws/$USER-horse"; do
        if [ -d "$_ws_candidate" ]; then
            WS_PATH="$_ws_candidate"
            echo "workspace fallback: $WS_PATH"
            break
        fi
    done
fi

export WS_PATH
export HF_HOME="$WS_PATH/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

mkdir -p "$HF_HOME" \
         "$HF_HUB_CACHE" \
         "$TRANSFORMERS_CACHE" \
         "$HF_DATASETS_CACHE"

export XDG_CACHE_HOME="$WS_PATH/.cache"
mkdir -p "$XDG_CACHE_HOME"

