#!/bin/bash
#SBATCH --job-name=trl_qwen35_4b_sft
#SBATCH -N 1
#SBATCH --output=logs/trl_qwen35_4b_sft_%j.out
#SBATCH --error=logs/trl_qwen35_4b_sft_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA="$WS_PATH/pgdata"
export SHARED_POSTGRES="$WS_PATH/postgres.addr"
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

# Reuse the shared PostgreSQL service, or start one and publish its address.
ensure_postgres

# Use the Python from PATH (base Miniconda). A dedicated /opt/conda/envs/trl
# conda environment does not exist on this cluster.
PYTHON=python

# Reduce allocator fragmentation during long generation/training jobs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

# run_trl reads INDEX from .env by default. Override it with the live service
# discovered by ensure_postgres so localhost is not used on another node.
export INDEX="$POSTGRES_CONNECTION"

echo "Postgres index: $INDEX"

if [ ! -d "/data/horse/ws/ripo631h-quokka/sft_output_qwen35_4b" ]; then
    echo "SFT adapter directory is missing" >&2
    exit 1
fi

mkdir -p logs

echo "Starting GRPO from the Qwen3.5-4B SFT adapter at $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"

"$PYTHON" utils/run_trl.py \
    --config configs/trl_qwen35_4b_sft_grpo.json

echo "GRPO training finished at $(date)"
