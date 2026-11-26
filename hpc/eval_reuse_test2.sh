#!/bin/bash
#SBATCH --output=logs/eval_reuse2_%j.out
#SBATCH --job-name=refactx_10
#SBATCH -N 1
#SBATCH --error=logs/eval_reuse2_%j.err
#SBATCH --time=30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

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

cat > /tmp/config_mintaka_35b3_test_n10.json << 'CFG'
{
    "experiment_name": "mintaka_test_qwen36_35b_n10",
    "output": null,
    "log_dir": "logs",
    "tablename": "qwen36",
    "model_name": "Qwen/Qwen3.6-35B-A3B",
    "model_dtype": "bfloat16",
    "dataset": "Rexhaif/mintaka-qa-en",
    "dataset_split": "test",
    "n": 10,
    "prompt": "prompts/prompt_qwen36_mini2.txt",
    "wandb": false,
    "unconstrained_generation": false,
    "debug": false,
    "debug_states": false,
    "continue": false,
    "http_rootcert": null,
    "device": "auto",
    "thinking": false,
    "num_beams": 1,
    "batch_size": 1,
    "generation_config": {"max_length": 1024, "do_sample": false}
}
CFG

echo "STARTED at $(date) on $(hostname) INDEX_PATH=$INDEX_PATH"
python -m utils.eval --config /tmp/config_mintaka_35b3_test_n10.json
echo "DONE at $(date)"
