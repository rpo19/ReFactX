# dummy model for testing without gpu

from base_model_config import ModelConfig
import time

class DummyOutput():
    def __init__(self):
        self.past_key_values = []

class DummyModel():
    def __init__(self):
        self.device = 'cpu'

    def eval(self):
        pass

    def generate(self, *args, **kwargs):
        time.sleep(2)
        return [[12]*40] # dummy output

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwds):
        return DummyOutput()

model_config = ModelConfig(
    model_name = 'Qwen/Qwen2.5-3B-Instruct', # for the tokenizer
    switch_pattern = [17417, 25], # "Fact:" after newline
    newline_token = 198,
    model_class=DummyModel,
)

model_config.batch_size = 1