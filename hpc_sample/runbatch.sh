#!/bin/bash

source env.sh

INDEX=$1
MODEL=$2
DATASET=$3
ADDITIONAL_ARGS=$4

cd ../..
python eval.py --index $INDEX --model $MODEL --dataset $DATASET $ADDITIONAL_ARGS
