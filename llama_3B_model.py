from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'meta-llama/Llama-3.2-3B-Instruct',
    switch_pattern = [17873, 25], # "Fact:" after newline
    newline_token = 198,
)
