from torch.utils.data import Dataset
from dotenv import load_dotenv
import os

load_dotenv()

class QADataset(Dataset):
    def preprocess(self, dataset):
        return dataset

    def __init__(self, dataset, config, preprocess=True):
        self.dataset = dataset
        if preprocess:
            self.dataset = self.preprocess(self.dataset)
        self.config = config

        self.skip_serialize = set(['skip_serialize','dataset'])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.dataset[key], self.config)
        else:
            return self.dataset[key]

    def __iter__(self):
        for sample in self.dataset:
            yield sample

    def dump_config(self):
        keys = set(self.__dict__.keys())
        keys.difference_update(self.skip_serialize)
        for key in keys:
            yield (key, self.__dict__[key])

    def questions_dataset(self):
        # Dynamically create a subclass of the calling class
        class DynamicQuestionsDataset(self.__class__):
            def __init__(dataset_self, dataset, config, preprocess=False):
                super().__init__(dataset, config, preprocess)

            def __getitem__(dataset_self, idx):
                return dataset_self.get_question(idx)

            def __iter__(dataset_self):
                for sample in super().__iter__():
                    yield dataset_self.get_question_from_sample(sample)

        # Return an instance of the dynamically created class
        return DynamicQuestionsDataset(self.dataset, self.config, preprocess=False)

    """
    Array of correct answers.
    """
    def get_answer(self, i) -> list:
        return [self.dataset[i]['answer']]

    def get_question_from_sample(self, sample) -> str:
        return sample['question']

    def get_question(self, i) -> str:
        sample = self.dataset[i]
        return self.get_question_from_sample(sample)

    @staticmethod
    def get_dataset_path(dataset_name):
        DATA_PATH = os.getenv('DATA_PATH', './data')
        path = os.path.join(DATA_PATH, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}. Please check the path.")
        return path
