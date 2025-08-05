#!/bin/bash
set -e
set -o pipefail

: "${ENV_FILE:=env.sh}"

mkdir -p logs

source "$ENV_FILE"

DATASET=$1
ADDITIONAL_ARGS_ALL="${@:2}"

for i in "${!MODELS[@]}"; do
  MODEL=${MODELS[$i]}
  INDEX=${INDEXES[$i]}
  JOB_NAME="$(basename $MODEL)_$(basename $DATASET)"
  GPUS_NUM=${GPUS_NUM_PER_MODEL[$i]}
  NODES_NUM=${NODES_NUM_PER_MODEL[$i]}
  MEM=${MEM_PER_MODEL[$i]}
  TIME_LIMIT=${TIME_LIMIT_PER_MODEL[$i]}
  ADDITIONAL_ARGS="${ADDITIONAL_ARGS_ALL} ${ADDITIONAL_ARGS_PER_MODEL[$i]}"

  SBATCH_ARGS="--job-name=$JOB_NAME \
        --output=logs/${JOB_NAME}_%j_out.log \
        --error=logs/${JOB_NAME}_%j_err.log \
        --partition=$PARTITION \
        --gres=gpu:$GPUS_NUM \
        --nodes=$NODES_NUM \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=$MEM \
        --time=$TIME_LIMIT \
        runeval.sh \
        $INDEX \
        $MODEL \
        $DATASET \
        $ADDITIONAL_ARGS" # e.g. --wandb

  if [ "$DEBUG" == "true" ]; then
    echo sbatch $SBATCH_ARGS
  else
    # Run the task
    sbatch $SBATCH_ARGS
  fi
done

# # Print to file a log of the variables used
LOG_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Models: ${MODELS[@]}" > logs/runall_$LOG_DATE.log
echo "Indexes: ${INDEXES[@]}" >> logs/runall_$LOG_DATE.log
echo "GPUs per model: ${GPUS_NUM_PER_MODEL[@]}" >> logs/runall_$LOG_DATE.log
echo "Nodes per model: ${NODES_NUM_PER_MODEL[@]}" >> logs/runall_$LOG_DATE.log
echo "Memory per model: ${MEM_PER_MODEL[@]}" >> logs/runall_$LOG_DATE.log
echo "Time limit per model: ${TIME_LIMIT_PER_MODEL[@]}" >> logs/runall_$LOG_DATE.log
echo "Partition: $PARTITION" >> logs/runall_$LOG_DATE.log
echo "Dataset: $DATASET" >> logs/runall_$LOG_DATE.log
echo "Additional args: $ADDITIONAL_ARGS" >> logs/runall_$LOG_DATE.log
echo "Job name: $JOB_NAME" >> logs/runall_$LOG_DATE.log

if [ "$DEBUG" != "true" ]; then
  echo "${#MODELS[@]} jobs submitted for dataset $DATASET."
  squeue --me
fi
