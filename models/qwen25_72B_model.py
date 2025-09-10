from qwen25_model_config import QwenModelConfig

model_config = QwenModelConfig(
    model_name = 'Qwen/Qwen2.5-72B-Instruct',
    device_map = 'auto'
)
model_config.batch_size = 4
