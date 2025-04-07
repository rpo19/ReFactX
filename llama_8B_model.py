from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct',
    switch_pattern = [17873, 25], # "Fact:" after newline
    newline_token = 198,
    device_map='auto'
)
