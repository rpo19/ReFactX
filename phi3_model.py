from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'microsoft/Phi-3-mini-128k-instruct',
    switch_pattern = [20738, 29901], # "Fact:" after newline
    newline_token = 13,
)
