#!/bin/bash
# Parent-matched reasoning ablation on fixed (v3) templates.
set -uo pipefail
cd /workspace/notebooks/refactx_clean
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs/reason3

train() {
  local tag="$1"
  rm -rf "sft_output_08b_${tag}"
  accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
    --config "configs/sft_config_qwen35_08b_${tag}.json" \
    --output-dir "sft_output_08b_${tag}" > "logs/reason3/train_${tag}.log" 2>&1
}

# Evaluate one adapter on all four 100-ID sets, two datasets per GPU at a time.
evaluate() {
  local tag="$1"
  for ds in kqa 2wiki webqsp cwq; do
    sed "s#\"adapter_path\": \"[^\"]*\"#\"adapter_path\": \"sft_output_08b_${tag}/checkpoint-100\"#;
         s#\"experiment_name\": \"[^\"]*\"#\"experiment_name\": \"eval100_${ds}_${tag}\"#" \
      "configs/eval100_${ds}.json" > "configs/eval100_${ds}_${tag}.json"
  done
  CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config "configs/eval100_kqa_${tag}.json" --flush > "logs/reason3/eval_kqa_${tag}.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config "configs/eval100_2wiki_${tag}.json" --flush > "logs/reason3/eval_2wiki_${tag}.log" 2>&1 &
  wait
  CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config "configs/eval100_webqsp_${tag}.json" --flush > "logs/reason3/eval_webqsp_${tag}.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python -m utils.eval --config "configs/eval100_cwq_${tag}.json" --flush > "logs/reason3/eval_cwq_${tag}.log" 2>&1 &
  wait
}

train reason3_plain
train reason3_nothink
evaluate reason3_plain
evaluate reason3_nothink
echo "ALL DONE" > logs/reason3/DONE
