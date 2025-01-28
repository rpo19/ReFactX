import json
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, tokenizer, prompt_template, device = "cpu", path = '/workspace/data/mintaka_test.json'):
        self.path = path
        print(f'Loading {self.path}')
        self.prompt_template = prompt_template
        self.device = device
        self.tokenizer = tokenizer
        with open(self.path) as fd:
            self.dataset = json.load(fd)
        self.encodings = self.tokenize()
        self.encodings = self.encodings.to(self.device)

    def __len__(self):
        return self.encodings['input_ids'].size(0)

    def __getitem__(self, idx):
        # Return a dictionary directly with tensor slices for the given index
        return {key: tensor[idx] for key, tensor in self.encodings.items()}

    def tokenize(self):
        questions = list(self.questions())
        return self.tokenizer(
                    questions,              # Input list of questions
                    padding=True,           # Pad to the longest sequence in the batch
                    truncation=True,        # Truncate to the model's maximum input length
                    return_tensors="pt"     # Return PyTorch tensors (use 'tf' for TensorFlow)
                )

    def questions(self, iter_function = iter):
        for sample in iter_function(self.dataset):
            yield self.prompt_template.format(sample['question'])
