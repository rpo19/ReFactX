from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

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

    def questions_dataset(self):
        return QuestionsDataset(self.dataset, self.config)

    def get_answer(self, i):
        return self.dataset[i]['answer']

class QuestionsDataset(QADataset):
    def __init__(self, dataset, config):
        super().__init__(dataset, config)

    def __getitem__(self, idx):
        return super().__getitem__(idx)['question']

    def __iter__(self):
        for sample in super().__iter__():
            yield sample['question']
