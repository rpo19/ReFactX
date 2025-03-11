from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'Qwen/Qwen2.5-3B-Instruct',
    switch_pattern = [17417, 25], # "Fact:" after newline
    newline_token = 198,
)
