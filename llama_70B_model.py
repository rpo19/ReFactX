from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'meta-llama/Llama-3.3-70B-Instruct',
    switch_pattern = [17873, 25], # "Fact:" after newline
    newline_token = 198,
    device_map='auto',
)
model_config.batch_size=4
