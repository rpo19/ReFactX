#!/bin/bash
#SBATCH --output=logs/eval_2wiki_sft_%j.out
#SBATCH --error=logs/eval_2wiki_sft_%j.err
#SBATCH --job-name=sft_2wiki_test
#SBATCH -N 1
#SBATCH --time=72:00:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"
export PGDATA=$WS_PATH/pgdata
export SHARED_POSTGRES=$WS_PATH/postgres.addr
source "$SLURM_SUBMIT_DIR/hpc/postgres_utils.sh"

ensure_postgres

cd /home/ripo631h/ReFactX

if [ -f "$SHARED_POSTGRES" ]; then
    source "$SHARED_POSTGRES"
    export INDEX_PATH="postgres://postgres:${PGPASSWORD:-postgres}@${PG_IP:-127.0.0.1}:${PG_PORT:-5432}/postgres"
    export POSTGRES_CONNECTION="$INDEX_PATH"
fi

MERGE_DIR=/data/horse/ws/ripo631h-quokka/sft_merged
if [ ! -d "$MERGE_DIR" ]; then
    echo "Merging LoRA adapters..."
    python -m utils.merge_lora
fi

echo "Starting SFT 2Wiki test eval at $(date)"
python -m utils.eval --config configs/config_2wiki_sft_test.json
echo "SFT 2Wiki test eval finished at $(date)"
