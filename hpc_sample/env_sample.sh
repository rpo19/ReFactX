# env.sh is source in both runbatch.sh and runall.sh
# custom environment variables
export HF_HUB_CACHE=/path/to/hf/hub

# do not define MODEL and INDEX here

MODELS=(llama_1B_model llama_8B_model llama_70B_model phi4_model qwen25_05_model qwen25_1B_model qwen25_3B_model qwen25_7B_model qwen25_14B_model qwen25_32B_model qwen25_72B_model)

# # postgres indexes
# INDEXES=(llama_index llama_index llama_index phi4_index qwen25_index qwen25_index qwen25_index qwen25_index qwen25_index qwen25_index qwen25_index)

# http indexes
INDEXES=(http_index_llama http_index_llama http_index_llama http_index_phi4 http_index_qwen http_index_qwen http_index_qwen http_index_qwen http_index_qwen http_index_qwen http_index_qwen)

ADDITIONAL_ARGS_PER_MODEL=(
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
)

# # HPC partition
PARTITION=your_partition_name

# # Define the number of GPUs for each model
GPUS_NUM_PER_MODEL=(1 1 4 2 1 1 1 1 2 3 4)

# # Define the number of nodes for each model
NODES_NUM_PER_MODEL=(1 1 1 1 1 1 1 1 1 1 1)

# # Define the memory required for each model (in GB)
MEM_PER_MODEL=(20G 40G 80G 40G 20G 20G 40G 40G 40G 80G 80G)

# # Define the time limit for each model (in hours)
TIME_LIMIT_PER_MODEL=(1:00:00 2:00:00 8:00:00 4:00:00 1:00:00 1:00:00 2:00:00 3:00:00 4:00:00 5:00:00 8:00:00)

# LLM as a judge
JUDGE_MODEL=meta-llama/Llama-3.3-70B-Instruct
JUDGE_BATCH_SIZE=4
JUDGE_DEVICE_MAP=auto
JUDGE_NUM_GPUS=2
JUDGE_MEM=80G
JUDGE_TIME_LIMIT=1:00:00

JUDGE_ADDITIONAL_ARGS_PER_MODEL=(
  "" # --fix-predictions --split-pattern '<\|end_of_text\|>'
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
  ""
)

# Assert that the lengths of MODELS and INDEXES are equal
if [ "${#MODELS[@]}" -ne "${#INDEXES[@]}" ]; then
  echo "Error: The lengths of MODELS and INDEXES arrays are not equal!" >&2
  exit 1
fi