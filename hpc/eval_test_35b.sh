#!/bin/bash
#SBATCH --output=logs/eval_test_35b_%j.out
#SBATCH --job-name=refactx_test35b
#SBATCH -N 1
#SBATCH --error=logs/eval_test_35b_%j.err
#SBATCH --time=30:00
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2

set -euo pipefail

source "$SLURM_SUBMIT_DIR/hpc/env.sh"

# Per-job disposable PostgreSQL on a read-only copy-on-write view of the
# shared PGDATA. Replaces the old shared-server/addr-file/watchdog logic;
# see hpc/postgres_overlay.sh for the mechanism and workarounds.
source "$SLURM_SUBMIT_DIR/hpc/postgres_overlay.sh"
pg_overlay_start ro
pg_overlay_install_traps

cd /home/ripo631h/ReFactX

# Write a temp config with n=10 and no wandb
cat > /tmp/config_mintaka_35b3_test_n10.json << 'CFG'
{
    "experiment_name": "mintaka_test_eval_qwen36_35b",
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
    "continue": true,
    "http_rootcert": null,
    "device": "auto",
    "thinking": false,
    "num_beams": 1,
    "batch_size": 1,
    "generation_config": {"max_length": 1024, "do_sample": false}
}
CFG

echo "Starting Mintaka 35b3 test (10 samples) at $(date)"
python -m utils.eval --config /tmp/config_mintaka_35b3_test_n10.json
echo "Done at $(date)"
