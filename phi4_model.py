from base_model_config import ModelConfig

model_config = ModelConfig(
    model_name = 'microsoft/phi-4',
    switch_pattern = [17873, 25], # "Fact:" after newline
    newline_token = 198,
    # load_model = False
)

# model_config.model = load_model_with_quantization() # TODO

# self.quantization_config = dict(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type='nf4',
#     bnb_4bit_compute_dtype='bfloat16')
# self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
#     quantization_config=BitsAndBytesConfig(**self.quantization_config),
#     low_cpu_mem_usage=True,
#     # attn_implementation="flash_attention_2"
#     )
