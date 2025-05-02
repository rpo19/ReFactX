from llama_model_config import LlamaModelConfig

model_config = LlamaModelConfig(
    model_name = 'meta-llama/Llama-3.3-70B-Instruct',
    device_map='auto',
)
model_config.batch_size=4
