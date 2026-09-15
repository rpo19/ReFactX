#!/bin/bash
#SBATCH --output=/home/h7/ripo631h/ReFactX/test_dataset_%j.out
#SBATCH --job-name=test_en
#SBATCH -N 1
#SBATCH --error=/home/h7/ripo631h/ReFactX/test_dataset_%j.err
#SBATCH --time=10:00
#SBATCH --gres=gpu:1

set -euo pipefail

source "/home/h7/ripo631h/ReFactX/hpc/env.sh"

python3 -c "
from datasets import load_dataset
ds = load_dataset('AmazonScience/mintaka', 'en', split='test', revision='refs/convert/parquet')
print(f'English test examples: {len(ds)}')
print(f'First lang: {ds[0][\"lang\"]}')
"
