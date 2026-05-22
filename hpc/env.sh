if [ -z "${WS_PATH:-}" ]; then
    # pass PS1LOGIN to sbatch
    case "$PS1LOGIN" in
        *capella*)
            WS_PATH="/data/cat/ws/$USER-horse"
            ;;
        *alpha*)
            WS_PATH="/data/horse/ws/$USER-quokka"
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

fi
