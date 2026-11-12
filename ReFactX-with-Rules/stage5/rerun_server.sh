#!/bin/bash
# Run the tokenizer preflight, greedy smoke test, or sampled Stage 5 benchmark.
set -euo pipefail

mode="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
root="$(cd "$(dirname "$0")/.." && pwd)"
python="${REFACTX_PYTHON:-$HOME/refactx_env/bin/python}"
model="${REFACTX_MODEL:-Qwen/Qwen3.5-4B}"
device="${REFACTX_DEVICE:-cpu}"
dtype="${REFACTX_DTYPE:-float32}"
cd "$root"

case "$mode" in
    preflight)
        exec "$python" -u stage5/run_benchmark.py --preflight --model "$model" "$@"
        ;;
    smoke)
        "$python" -u stage5/run_benchmark.py --preflight --model "$model" "$@"
        exec "$python" -u stage5/run_benchmark.py --model "$model" \
            --device "$device" --dtype "$dtype" --n-samples 1 --no-sample --check-masks "$@"
        ;;
    benchmark)
        exec "$python" -u stage5/run_benchmark.py --model "$model" \
            --device "$device" --dtype "$dtype" --n-samples 3 "$@"
        ;;
    *)
        echo "Usage: $0 [preflight|smoke|benchmark] [benchmark options]" >&2
        exit 2
        ;;
esac
