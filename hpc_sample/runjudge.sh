#!/bin/bash

source env.sh

JUDGE_MODEL=$1
PREDICTION_FILE=$2
DEVICE_MAP=$3
BATCH_SIZE=$4
ADDITIONAL_ARGS=$5

cd ..
python llm_as_a_judge.py --model $JUDGE_MODEL --predictions $PREDICTION_FILE --device-map $DEVICE_MAP --batch-size $BATCH_SIZE $ADDITIONAL_ARGS
