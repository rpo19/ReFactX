from base_model_config import ModelConfig

SWITCH_PATTERN = {27: {34210: {29: {}}}, 366: {34210: {29: {}}}} # <fact> or _<fact>
NEWLINE_TOKEN = 198

class LlamaModelConfig(ModelConfig):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('switch_pattern', SWITCH_PATTERN)
        kwargs.setdefault('newline_token', NEWLINE_TOKEN)
        super().__init__(*args, **kwargs)