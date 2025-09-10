from torch.utils.data import Dataset
from dotenv import load_dotenv
import os
from refactx.prompt_base import PROMPT_TEMPLATE

load_dotenv()

class QADataset(Dataset):
    def dump(self):
        return self.dataset

    def preprocess(self, dataset):
        return dataset

    def __init__(self, dataset, config, preprocess=True, name='QADataset'):
        self.dataset = dataset
        self.config = config
        if preprocess:
            self.dataset = self.preprocess(self.dataset)
            # now self.dataset should be a list
            dataset_start_from = int(os.environ.get('DATASET_START_FROM', 0))
            if dataset_start_from:
                print(f"Skipping first {dataset_start_from} samples from dataset. Previous length: {len(self.dataset)}, current length: {len(self.dataset) - dataset_start_from}")
                self.dataset = self.dataset[dataset_start_from:]
                config['start_from'] = dataset_start_from

        self.prompt_template = PROMPT_TEMPLATE

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
                if isinstance(idx, slice):
                    config = self.config.copy()
                    config['start_from'] = idx.start
                    return DynamicQuestionsDataset(dataset_self.dataset[idx], dataset_self.config, preprocess=False)
                else:
                    return dataset_self.get_question(idx)

            def __iter__(dataset_self):
                for sample in super().__iter__():
                    yield dataset_self.get_question_from_sample(sample)

        # Return an instance of the dynamically created class
        return DynamicQuestionsDataset(self.dataset, self.config, preprocess=False)

    def get_answer_type(self, sample) -> str:
        return 'simple'

    def get_answer(self, i) -> str:
        sample = self.dataset[i]
        return self.get_answer_from_sample(sample)

    def get_answer_from_sample(self, sample) -> str:
        return sample['answer']

    def get_question_from_sample(self, sample) -> str:
        return sample['question']

    def get_question(self, i) -> str:
        sample = self.dataset[i]
        return self.get_question_from_sample(sample)

    def get_question_type(self, sample):
        return 'unknown'

    '''
    Usage:
    ```
        from datasets import Dataset
        ds = Dataset.from_generator(QADataset.to_verl(prompt_fn))
        ds.to_parquet(path)
    ```
    '''
    def to_verl(self, prompt_fn=None, name=None):
        if prompt_fn is None:
            prompt_fn = lambda question: question

        if name is None:
            name = self.name

        for i in range(len(self)):
            question = self.get_question(i)
            prompt = prompt_fn(question)
            # TODO system prompt
            data = {
                "data_source": name,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": self.get_answer(i)
                },
                "extra_info": {
                    'index': i,
                    "question": question,
                }
            }
            yield data

    @staticmethod
    def get_dataset_path(dataset_name):
        DATA_PATH = os.getenv('DATA_PATH', './data')
        path = os.path.join(DATA_PATH, dataset_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}. Please check the path.")
        return path
