from .base_model_config import ModelConfig

SWITCH_PATTERN = {17417:{25:{}},36712:{25:{}},33110:{25:{}},2097:{25:{}}} # 'Fact:' and ' Fact:' for qwen. also 'fact:' and ' fact:'

NEWLINE_TOKEN = 198

class QwenModelConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('switch_pattern', SWITCH_PATTERN)
        kwargs.setdefault('newline_token', NEWLINE_TOKEN)
        super().__init__(*args, **kwargs)