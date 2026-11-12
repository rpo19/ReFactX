#!/bin/bash
# Submit the 8 preliminary 100-example 4B evals as independent jobs.
# Each job requests one GPU; Slurm decides when and where they run, so they can
# execute in parallel subject to the cluster's free resources.

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIGS=(
    configs/config_prelim100_mintaka_vanilla4b.json
    configs/config_prelim100_mintaka_sft4b.json
    configs/config_prelim100_2wiki_vanilla4b.json
    configs/config_prelim100_2wiki_sft4b.json
    configs/config_prelim100_webqsp_vanilla4b.json
    configs/config_prelim100_webqsp_sft4b.json
    configs/config_prelim100_cwq_vanilla4b.json
    configs/config_prelim100_cwq_sft4b.json
)

for cfg in "${CONFIGS[@]}"; do
    if [ ! -f "$cfg" ]; then
        echo "Missing config: $cfg" >&2
        exit 1
    fi
    job_name=$(basename "$cfg" .json)
    job_name="${job_name#config_}"
    job_id=$(sbatch --parsable --job-name="$job_name" \
        hpc/eval_prelim100_4b_task.sh "$cfg")
    echo "submitted $cfg -> job $job_id"
done

echo "Submitted ${#CONFIGS[@]} jobs. Check with: squeue -u $USER"
