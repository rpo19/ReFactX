PS1LOGIN="${PS1LOGIN:-$(hostname -A)}"
# pass PS1LOGIN to sbatch: > sbatch "--export=ALL,PS1LOGIN=$PS1" populate.sh
echo $PS1LOGIN
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

