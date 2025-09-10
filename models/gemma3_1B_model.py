from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'google/gemma-3-1b-it',
    switch_pattern = [27711, 236787], # "Fact:" after newline
    newline_token = 107,
    device_map = 'cuda'
)
