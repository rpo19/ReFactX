#!/bin/bash

source env.sh

INDEX=$1
MODEL=$2
DATASET=$3
ADDITIONAL_ARGS=$4

cd ..
CURRENT_COMMIT=$(git rev-parse --short HEAD)
CURRENT_DATE=$(date +"%Y%m%d_%H%M%S")
LOGDIR=logs_${CURRENT_DATE}_${CURRENT_COMMIT}
mkdir -p $LOGDIR

python eval.py --index $INDEX --model $MODEL --dataset $DATASET $ADDITIONAL_ARGS --log-dir $LOGDIR
