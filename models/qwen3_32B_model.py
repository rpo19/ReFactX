from qwen25_model_config import QwenModelConfig

model_config = QwenModelConfig(
    model_name = 'Qwen/Qwen3-32B',
    device_map = 'auto'
)
model_config.batch_size = 4
