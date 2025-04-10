from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'Qwen/QwQ-32B',
    switch_pattern = [17417, 25], # "Fact:" after newline
    newline_token = 198,
    device_map = 'auto'
)
model_config.batch_size = 4

