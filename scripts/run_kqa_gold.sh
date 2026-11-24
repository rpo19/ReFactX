#!/bin/bash
set -uo pipefail
cd /workspace/notebooks/refactx_clean
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
mkdir -p logs/kqa_gold
rm -rf sft_output_08b_kqa_gold
accelerate launch --multi_gpu --num_processes 2 -m utils.sft_train \
  --config configs/sft_config_qwen35_08b_kqa_gold.json \
  --output-dir sft_output_08b_kqa_gold > logs/kqa_gold/train.log 2>&1
sed 's#"adapter_path": "[^"]*"#"adapter_path": "sft_output_08b_kqa_gold/checkpoint-100"#;
     s#"experiment_name": "[^"]*"#"experiment_name": "eval100_kqa_kqa_gold"#' \
  configs/eval100_kqa.json > configs/eval100_kqa_kqa_gold.json
CUDA_VISIBLE_DEVICES=0 python -m utils.eval --config configs/eval100_kqa_kqa_gold.json --flush > logs/kqa_gold/eval.log 2>&1
echo DONE > logs/kqa_gold/DONE
