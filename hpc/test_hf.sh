#!/bin/bash
#SBATCH --job-name=test_hf
#SBATCH -N 1
#SBATCH --output=logs/test_hf_%j.out
#SBATCH --error=logs/test_hf_%j.err
#SBATCH --time=00:05:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

set -euo pipefail
source "$SLURM_SUBMIT_DIR/hpc/env.sh"
cd /home/ripo631h/ReFactX

echo "=== AmazonScience/mintaka (en, test) ==="
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('AmazonScience/mintaka', 'en', split='test')
    print(f'OK: {len(ds)} samples')
except Exception as e:
    print(f'FAIL: {e}')
"

echo "=== Rexhaif/mintaka-qa-en (test) ==="
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('Rexhaif/mintaka-qa-en', split='test')
    print(f'OK: {len(ds)} samples, keys: {list(ds[0].keys())}')
except Exception as e:
    print(f'FAIL: {e}')
"

echo "=== xanhho/2WikiMultihopQA (validation, parquet) ==="
python -c "
from datasets import load_dataset
try:
    ds = load_dataset('xanhho/2WikiMultihopQA', revision='refs/convert/parquet', split='validation')
    print(f'OK: {len(ds)} samples, keys: {list(ds[0].keys())}')
except Exception as e:
    print(f'FAIL: {e}')
"

echo "Done"
