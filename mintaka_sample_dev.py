import json
from torch.utils.data import Dataset
import random

class QADataset(Dataset):
    def __init__(self):
        self.path = '/workspace/data/mintaka_dev.json'
        print(f'Loading {self.path}')
        with open(self.path) as fd:
            self.dataset = json.load(fd)

        # sample
        self.random_seed = 1234
        random.seed(self.random_seed)
        self.dataset = random.sample(self.dataset, 50)

        self.skip_serialize = set(['skip_serialize','dataset'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Return a dictionary directly with tensor slices for the given index
        sample = self.dataset[idx]
        return sample

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def dump_config(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

class QuestionsDataset(QADataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, idx):
        return super().__getitem__(idx)['question']

    def __iter__(self):
        for sample in super().__iter__():
            yield sample['question']