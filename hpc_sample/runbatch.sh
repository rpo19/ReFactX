#!/bin/bash
#SBATCH --job-name=jobname
#SBATCH --nodes 1
#SBATCH --time=01:00:00
#SBATCH --partition=part1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=job_output.out      # Standard output
#SBATCH --error=job_error.out        # Standard error

module load module1

source /opt/share/conda.sh # load conda

python eval.py --index phi4_index --model phi4_model --dataset mintaka_sample_dev mdz-hpc-phi4-mintaka-sample-dev mdz-hpc-phi4-mintaka-sample-dev.out

