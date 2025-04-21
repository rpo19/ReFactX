from base_model_config import ModelConfig

SWITCH_PATTERN = {27: {33110: {29: {}}}, 366: {33110: {29: {}}}} # <fact> or _<fact>
NEWLINE_TOKEN = 198

class QwenModelConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('switch_pattern', SWITCH_PATTERN)
        kwargs.setdefault('newline_token', NEWLINE_TOKEN)
        super().__init__(*args, **kwargs)