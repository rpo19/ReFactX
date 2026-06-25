#!/bin/bash
#SBATCH --job-name=prep_eval
#SBATCH -N 1
#SBATCH --output=logs/prep_eval_%j.out
#SBATCH --error=logs/prep_eval_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd /home/ripo631h/ReFactX
python -m utils.prep_eval_data
echo "Done"
