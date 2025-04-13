#!/bin/bash
set -e
set -o pipefail

mkdir -p logs

source env.sh

PREDICTION_FILES=$@

for i in "${!PREDICTION_FILES[@]}"; do
  PREDICTION_FILE=${PREDICTION_FILES[$i]}
  if [ ! -f "$PREDICTION_FILE" ]; then
    echo "Error: Prediction file $PREDICTION_FILE does not exist!" >&2
    exit 1
  fi
  JOB_NAME="judge_$PREDICTION_FILE"
  SBATCH_ARGS="--job-name=$JOB_NAME \
        --output=logs/$JOB_NAME_%j_out.log \
        --error=logs/$JOB_NAME_%j_err.log \
        --partition=$PARTITION \
        --gres=gpu:$JUDGE_NUM_GPUS \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=1 \
        --mem=$JUDGE_MEM \
        --time=$JUDGE_TIME_LIMIT \
        runjudge.sh \
        $JUDGE_MODEL \
        $PREDICTION_FILE \
        $JUDGE_DEVICE_MAP \
        $JUDGE_BATCH_SIZE \
        ${JUDGE_ADDITIONAL_ARGS_PER_MODEL[$i]}" # e.g. --wandb

  if [ "$DEBUG" == "true" ]; then
    echo sbatch $SBATCH_ARGS
  else
    # Run the task
    sbatch $SBATCH_ARGS
  fi

  if [ "$DEBUG" == "true" ]; then
    echo "Debug mode enabled. Breaking after first iteration."
    break
  fi
done

# Print to file a log of the variables used
LOG_DATE=$(date +"%Y%m%d_%H%M%S")
echo "Prediction files: ${PREDICTION_FILES[@]}" > logs/runall_$LOG_DATE.log
echo "Partition: $PARTITION" >> logs/runall_$LOG_DATE.log
echo "Judge GPUs: $JUDGE_NUM_GPUS" >> logs/runall_$LOG_DATE.log
echo "Judge memory: $JUDGE_MEM" >> logs/runall_$LOG_DATE.log
echo "Judge time limit: $JUDGE_TIME_LIMIT" >> logs/runall_$LOG_DATE.log
echo "Judge model: $JUDGE_MODEL" >> logs/runall_$LOG_DATE.log
echo "Judge device map: $JUDGE_DEVICE_MAP" >> logs/runall_$LOG_DATE.log
echo "Judge batch size: $JUDGE_BATCH_SIZE" >> logs/runall_$LOG_DATE.log
echo "Additional args per model: ${JUDGE_ADDITIONAL_ARGS_PER_MODEL[@]}" >> logs/runall_$LOG_DATE.log

if [ "$DEBUG" != "true" ]; then
  echo "${#PREDICTION_FILES[@]} jobs submitted."
  squeue --me
fi
