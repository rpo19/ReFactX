from base_model_config import ModelConfig

SWITCH_PATTERN = {17873: {25: {}}, 34210: {25: {}}, 37812: {25: {}}, 2144: {25: {}}}
NEWLINE_TOKEN = 198

class LlamaModelConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('switch_pattern', SWITCH_PATTERN)
        kwargs.setdefault('newline_token', NEWLINE_TOKEN)
        super().__init__(*args, **kwargs)