# dummy model for testing without gpu

from base_model_config import ModelConfig
import time
import numpy as np  # Import numpy for array handling

class DummyOutput():
    def __init__(self, output: list):
        self.output = output
        self.past_key_values = []
        self.shape = [len(output)]
        if self.shape[0] > 0:
            if isinstance(output[0], list):
                self.shape.append(len(output[0]))

    def __iter__(self):
        # Make the class iterable by returning an iterator over numpy arrays
        for item in self.output:
            yield np.array(item)  # Convert each item to a numpy array

    def __getitem__(self, index):
        # Allow indexing into the output and return a numpy array
        if isinstance(index, slice):
            return np.array(self.output[index])  # Convert the slice to a numpy array
        return np.array(self.output[index])  # Convert the single item to a numpy array

class DummyModel():
    def __init__(self):
        self.device = 'cpu'

    def eval(self):
        pass

    def generate(self, *args, **kwargs):
        # time.sleep(2)
        return DummyOutput([[12] * 40])  # dummy output

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def __call__(self, *args, **kwds):
        return DummyOutput([12])

model_config = ModelConfig(
    model_name='Qwen/Qwen2.5-3B-Instruct',  # for the tokenizer
    switch_pattern=[17417, 25],  # "Fact:" after newline
    newline_token=198,
    model_class=DummyModel,
)

model_config.batch_size = 1