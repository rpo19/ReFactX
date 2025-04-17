#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=throughput
#SBATCH --output=logs/throughput_%j_out.log
#SBATCH --error=logs/throughput_%j_err.log

source env.sh

MODEL=$1
INDEX=$2

cd ..

python throughput.py --model $MODEL --index $INDEX --max-tokens 10 --output throughput.unconstrained.10.$MODEL.log --unconstrained-generation
python throughput.py --model $MODEL --index $INDEX --max-tokens 100 --output throughput.unconstrained.100.$MODEL.log --unconstrained-generation
python throughput.py --model $MODEL --index $INDEX --max-tokens 500 --output throughput.unconstrained.500.$MODEL.log --unconstrained-generation
python throughput.py --model $MODEL --index $INDEX --max-tokens 1000 --output throughput.unconstrained.1000.$MODEL.log --unconstrained-generation

python throughput.py --model $MODEL --index $INDEX --max-tokens 10 --output throughput.constrained.10.$MODEL.log
python throughput.py --model $MODEL --index $INDEX --max-tokens 100 --output throughput.constrained.100.$MODEL.log
python throughput.py --model $MODEL --index $INDEX --max-tokens 500 --output throughput.constrained.500.$MODEL.log
python throughput.py --model $MODEL --index $INDEX --max-tokens 1000 --output throughput.constrained.1000.$MODEL.log

